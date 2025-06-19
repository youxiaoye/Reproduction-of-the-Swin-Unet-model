import os
import random
import numpy as np
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
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
        
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class ISIC_dataset(Dataset):
    def __init__(self, base_dir, split, list_dir):
        self.base_dir = base_dir
        self.split = split
        self.list_dir = list_dir
        
        # 读取数据列表
        self.sample_list = pd.read_csv(os.path.join(list_dir, self.split + '.txt'))
        
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        # 获取图像和掩码路径
        img_path = self.sample_list.iloc[idx]['image_path']
        mask_path = self.sample_list.iloc[idx]['mask_path']
        
        # 读取图像和掩码
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 转换为tensor
        image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        # 获取文件名（不包含路径和扩展名）
        case_name = os.path.splitext(os.path.basename(img_path))[0]
        
        return {
            'image': image,
            'label': mask,
            'case_name': case_name
        } 