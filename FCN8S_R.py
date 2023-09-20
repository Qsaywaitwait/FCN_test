import numpy as np
import torch
from torchsummary import summary
from torchvision import transforms
from torch.utils import data
from PIL import Image
from torchvision.models import vgg19
import torch.nn.functional as F
from torch import nn, optim


# 使用预训练的VGG19网络：
model_vgg19 = vgg19(pretrained = True).cuda()
summary(model_vgg19, input_size = (3,320, 480))


# 自定义FCN-8s:
class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.base_model = model_vgg19.features  # 去除全连接层

        self.ConvTrans1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )

        self.ConvTrans2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )

        self.ConvTrans3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)  # 1x1卷积， 在像素级别进行分类
        # 将对应的池化层存入字典，方便到时候提取该层的特征进行求和：
        self.layers = {'18': 'maxpool_3', '27': 'maxpool_4', '36': 'maxpool_5', }

    def forward(self, x):
        output = {}  # 用来保存中间层的特征
        # 首先利用预训练的VGG19提取特征：
        # _modules.items()遍历item()来实现层层监视，
        # 即如果这一层是我们需要保存的特征，我们就可以使用字典结构将这一层的特征提取并保存起来，在反卷积操作的时候再提取出参与融合
        for name, layer in self.base_model._modules.items():
            x = layer(x)

            # 如果当前层的特征需要被保存：
            if name in self.layers:
                output[self.layers[name]] = x
        x5 = output['maxpool_5']  # 原图的H/32, W/32
        x4 = output['maxpool_4']  # 原图的H/16, W/16
        x3 = output['maxpool_3']  # 原图的H/ 8, W/ 8

        # 对特征进行相关转置卷积操作，逐渐恢复到原图大小:
        score = self.ConvTrans1(x5)  # 提取maxpool_5的特征，转置卷积进行上采样，激活函数输出
        score = self.ConvTrans2(score + x4)  # 上采样后的特征再与maxpool_4的特征相加，并进行归一化操作
        score = self.ConvTrans3(score + x3)  # score
        score = self.classifier(score)

        return score