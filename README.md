# Reproduction-of-the-Swin-Unet-model
在ISIC-2018数据集上对Swin-Unet模型进行的复现。初步结果尚可，经200轮训练模型Dice loss收敛于0.225左右。
Swin-Unet论文链接：[https://arxiv.org/abs/2105.05537](https://doi.org/10.48550/arXiv.2105.05537)
ISIC-2018数据集下载地址：[https://challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data#2019)
数据集特征：包含 10,000+张皮肤病变图像，涵盖 7 类疾病（如黑色素瘤、基底细胞癌），具有高分辨率、多类别，包含复杂纹理和颜色变化等特征；专业医生标注病变区域边界，分割任务含 2594 张训练图像。
主要改动：
1.将原始三维.h5 数据转为二维图像格式（ISIC-2018：RGB 彩色图），统一缩放到 224×224 分辨率，图像采用双三次插值（order=3），标签使用最近邻插值（order=0）保持二值性；识别输入维度，RGB 数据转为 3×H×W，并归一化至[0,1]。
2.优化器调整：以 AdamW 替代原 SGD（lr=3e-4, weight_decay=1e-5），适配Transformer 的梯度更新特性，配合余弦退火学习率调度。
![image](https://github.com/user-attachments/assets/808a378d-e521-4a29-a0f5-a7532cc4a7ea)

