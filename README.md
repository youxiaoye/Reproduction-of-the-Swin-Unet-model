# Reproduction-of-the-Swin-Unet-model

在 ISIC-2018 数据集上对 Swin-Unet 模型进行的复现。初步结果尚可，经 200 轮训练，模型 Dice Loss 收敛于 0.225 左右。

## 相关链接
- **Swin-Unet 论文链接**：[Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation](https://arxiv.org/abs/2105.05537)  
- **ISIC-2018 数据集下载地址**：[ISIC Challenge 2018](https://challenge.isic-archive.com/data/#2018)  
 

## 数据集特征
- 包含 10,000+ 张皮肤病变图像，涵盖 7 类疾病（如黑色素瘤、基底细胞癌）。  
- 图像具有高分辨率、多类别，包含复杂纹理和颜色变化等特征。  
- 由专业医生标注病变区域边界，分割任务包含 2594 张训练图像。

## 主要改动
1. **数据预处理**：
   - 将原始三维 `.h5` 数据转为二维图像格式（ISIC-2018：RGB 彩色图），统一缩放到 224×224 分辨率。  
   - 图像采用双三次插值（`order=3`），标签使用最近邻插值（`order=0`）保持二值性。  
   - 识别输入维度，将 RGB 数据转为 3×H×W，并归一化至 `[0,1]`。

2. **优化器调整**：
   - 使用 AdamW 替代原 SGD（`lr=3e-4`, `weight_decay=1e-5`），适配 Transformer 的梯度更新特性。  
   - 配合余弦退火学习率调度。

## 训练结果
![Training Result](https://github.com/user-attachments/assets/808a378d-e521-4a29-a0f5-a7532cc4a7ea)
