import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        # 计算Dice系数
        dice = metric.binary.dc(pred, gt)
        # 计算Hausdorff距离
        hd95 = metric.binary.hd95(pred, gt)
        # 计算IoU (mIoU)
        iou = metric.binary.jc(pred, gt)
        # 计算精确率 (Precision)
        precision = metric.binary.precision(pred, gt)
        # 计算召回率 (Recall)
        recall = metric.binary.recall(pred, gt)
        # 计算F1分数 (使用precision和recall计算)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return dice, hd95, iou, precision, recall, f1
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 1, 0, 0
    else:
        return 0, 0, 0, 0, 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    # 修改维度处理
    image = image.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    
    # 确保图像维度正确
    if len(image.shape) == 4:  # [B, C, H, W]
        image = image[0]  # 取第一个batch
    if len(label.shape) == 3:  # [B, H, W]
        label = label[0]  # 取第一个batch
    
    # 如果是3通道图像，转换为单通道
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.mean(image, axis=0)  # 转换为灰度图
    
    # 确保输入维度正确
    if len(image.shape) == 2:
        image = image[np.newaxis, ...]  # 添加通道维度
    
    # 进行预测
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        outputs = net(input)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
    
    # 计算评估指标
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    
    # 保存预测结果
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    
    return metric_list