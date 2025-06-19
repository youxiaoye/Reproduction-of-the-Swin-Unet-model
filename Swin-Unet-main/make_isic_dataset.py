import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
from tqdm import tqdm

def process_isic_dataset(data_root, output_dir):
    """
    处理ISIC-2018数据集
    data_root: ISIC-2018数据集根目录
    output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    
    # 获取所有图像文件
    image_dir = os.path.join(data_root, 'train', 'jpgimages')
    mask_dir = os.path.join(data_root, 'train', 'masks')
    
    # 获取所有图像文件名（不包含扩展名）
    image_files = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]
    mask_files = [f.split('.')[0] for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    # 准备数据列表
    data_list = []
    
    # 创建图像文件名到掩码文件名的映射
    # 假设掩码文件名是图像文件名加1000
    mask_map = {str(int(img_name) + 1000): img_name for img_name in image_files}
    
    for img_name in tqdm(image_files):
        # 图像路径
        img_path = os.path.join(image_dir, f'{img_name}.jpg')
        # 掩码路径（使用映射关系）
        mask_name = str(int(img_name) + 1000)
        mask_path = os.path.join(mask_dir, f'{mask_name}.png')
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            # 读取并预处理图像
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            
            # 读取并预处理掩码
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (224, 224))
            mask = (mask > 127).astype(np.uint8)  # 二值化
            
            # 保存处理后的图像和掩码
            img_save_path = os.path.join(output_dir, 'images', f'{img_name}.png')
            mask_save_path = os.path.join(output_dir, 'masks', f'{img_name}.png')
            
            cv2.imwrite(img_save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_save_path, mask)
            
            data_list.append({
                'image_path': img_save_path,
                'mask_path': mask_save_path,
                'num_classes': 2  # 二分类问题
            })
    
    # 划分训练集、验证集和测试集
    train_list, temp_list = train_test_split(data_list, test_size=0.3, random_state=42)
    val_list, test_list = train_test_split(temp_list, test_size=0.5, random_state=42)
    
    # 保存数据集列表
    os.makedirs('./lists/ISIC2018', exist_ok=True)
    pd.DataFrame(train_list).to_csv('./lists/ISIC2018/train.txt', index=False)
    pd.DataFrame(val_list).to_csv('./lists/ISIC2018/val.txt', index=False)
    pd.DataFrame(test_list).to_csv('./lists/ISIC2018/test.txt', index=False)
    
    print(f"数据集划分完成：")
    print(f"训练集：{len(train_list)}张")
    print(f"验证集：{len(val_list)}张")
    print(f"测试集：{len(test_list)}张")

if __name__ == '__main__':
    # 使用相对路径
    data_root = './datasets/ISIC2018'  # 原始数据集路径
    output_dir = './processed/ISIC2018'  # 处理后的数据保存路径
    process_isic_dataset(data_root, output_dir) 