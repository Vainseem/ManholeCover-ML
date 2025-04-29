import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image
import random
import shutil
from PIL import Image
import pandas as pd
import numpy as np
#引入了残差模块的CNN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.res=nn.Conv2d(in_channels,out_channels,1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.res(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out+residual)
        return out
#没有池化层的传统CNN
class Regular_CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Regular_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
#带有池化层的传统CNN
class MaxPooling_CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaxPooling_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.pool= nn.MaxPool2d(2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.relu(x)
        return x
class Traditional_Network1(nn.Module):
    def __init__(self):
        super(Traditional_Network1, self).__init__()
        self.conv1 = MaxPooling_CNN(3,32)
        self.conv2 = MaxPooling_CNN(32, 32)
        self.conv3 = MaxPooling_CNN(32, 32)
        self.conv4 = MaxPooling_CNN(32, 32)
        self.conv5 = MaxPooling_CNN(32, 64)
        self.conv6 = MaxPooling_CNN(64, 64)
        self.conv7 = Regular_CNN(64, 64)
        self.conv8 = Regular_CNN(64, 80)
        self.conv9 = Regular_CNN(80, 80)
        self.conv10 = Regular_CNN(80, 80)
        self.conv11 = Regular_CNN(80, 96)
        self.conv12 = Regular_CNN(96, 128)
        self.fc1 = nn.Linear(2048, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
class Resnet_Network1(nn.Module):
    def __init__(self):
        super(Resnet_Network1, self).__init__()
        self.conv1 = MaxPooling_CNN(3, 32)
        self.conv2 = MaxPooling_CNN(32, 32)
        self.conv3 = MaxPooling_CNN(32, 32)
        self.conv4 = MaxPooling_CNN(32, 32)
        self.conv5 = MaxPooling_CNN(32, 64)
        self.conv6 = MaxPooling_CNN(64, 64)
        self.conv7 = ResidualBlock(64, 64)
        self.conv8 = ResidualBlock(64, 80)
        self.conv9 = ResidualBlock(80, 80)
        self.conv10 = ResidualBlock(80, 80)
        self.conv11 = ResidualBlock(80, 96)
        self.conv12 = ResidualBlock(96, 128)
        self.fc1 = nn.Linear(2048, 5)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
class Resnet_Network2(nn.Module):
    def __init__(self):
        super(Resnet_Network2, self).__init__()
        self.conv1 = MaxPooling_CNN(3, 32)
        self.conv2 = MaxPooling_CNN(32, 32)
        self.conv3 = MaxPooling_CNN(32, 32)
        self.conv4 = MaxPooling_CNN(32, 32)
        self.conv5 = ResidualBlock(32, 48)
        self.conv6 = ResidualBlock(48, 48)
        self.conv7 = ResidualBlock(48, 48)
        self.conv8 = MaxPooling_CNN(48, 48)
        self.conv9 = ResidualBlock(48, 64)
        self.conv10 = ResidualBlock(64, 64)
        self.conv11 = ResidualBlock(64, 64)
        self.conv12 = MaxPooling_CNN(64, 64)
        self.conv13 = ResidualBlock(64, 96)
        self.conv14 = ResidualBlock(96, 96)
        self.conv15 = ResidualBlock(96, 128)
        self.conv16 = ResidualBlock(128, 128)
        self.fc1 = nn.Linear(2048, 5)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
class Resnet_Network3(nn.Module):
    def __init__(self):
        super(Resnet_Network3, self).__init__()
        self.conv1 = MaxPooling_CNN(3, 32)
        self.conv2 = MaxPooling_CNN(32, 32)
        self.conv3 = MaxPooling_CNN(32, 32)
        self.conv4 = MaxPooling_CNN(32, 32)
        self.conv5 = ResidualBlock(32, 48)
        self.conv6 = ResidualBlock(48, 48)
        self.conv7 = ResidualBlock(48, 48)
        self.conv8 = ResidualBlock(48, 48)
        self.conv9 = MaxPooling_CNN(48, 48)
        self.conv10 = ResidualBlock(48, 64)
        self.conv11 = ResidualBlock(64, 64)
        self.conv12 = ResidualBlock(64, 64)
        self.conv13 = ResidualBlock(64, 64)
        self.conv14 = MaxPooling_CNN(64, 64)
        self.conv15 = ResidualBlock(64, 80)
        self.conv16 = ResidualBlock(80, 80)
        self.conv17 = ResidualBlock(80, 80)
        self.conv18 = ResidualBlock(80, 96)
        self.conv19 = ResidualBlock(96, 96)
        self.conv20 = ResidualBlock(96, 128)
        self.fc1 = nn.Linear(2048, 5)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# 真实标签和预测的标签
y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
y_scores = [0.1, 0.4, 0.8, 0.8, 0.13, 0.55, 0.25, 0.8, 0.9, 0.5]
y=[]
s=[]
for i in range(0,10000):
    if i<5000:
        y.append(0)
    else:
        y.append(1)
for i in range(0,10000):
    p=random.random()
    if(i<5000):
        s.append(random.uniform(0,0.51))
    else:
        s.append((random.uniform(0.49,1)))
# 计算精确率和召回率以及阈值
precision, recall, thresholds = precision_recall_curve(y, s)

# 绘制PR曲线
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
