import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
import pandas as pd


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # 获取图像尺寸
        if len(image.shape) == 3:  # 如果是3通道图像
            h, w, _ = image.shape
        else:  # 如果是单通道图像
            h, w = image.shape
            
        # 调整图像大小
        if h != self.output_size[0] or w != self.output_size[1]:
            if len(image.shape) == 3:
                image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w, 1), order=3)
            else:
                image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w), order=3)
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
        
        # 转换为tensor
        if len(image.shape) == 3:
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        else:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, split, list_dir):
        self.base_dir = base_dir
        self.split = split
        self.list_dir = list_dir
        
        # 读取数据列表
        if split == 'test':
            # 使用test目录
            self.image_dir = os.path.join(base_dir, 'images')
            self.mask_dir = os.path.join(base_dir, 'masks')
            self.image_list = sorted(os.listdir(self.image_dir))
        else:
            # 使用train目录
            self.image_dir = os.path.join(base_dir, 'images')
            self.mask_dir = os.path.join(base_dir, 'masks')
            self.image_list = sorted(os.listdir(self.image_dir))
        
        # 添加数据增强
        self.transform = RandomGenerator(output_size=[224, 224])
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # 获取图像和掩码路径
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '_mask.png'))
        
        # 读取图像和掩码
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 数据增强
        if self.split == 'train':
            sample = {'image': image, 'label': mask}
            sample = self.transform(sample)
            image = sample['image']
            mask = sample['label']
        else:
            # 转换为tensor
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        # 获取文件名（不包含路径和扩展名）
        case_name = os.path.splitext(img_name)[0]
        
        return {
            'image': image,
            'label': mask,
            'case_name': case_name
        }
