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
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import precision_recall_curve, roc_curve, auc
'''切分数据集'''
'''
data_path='new_train'
ratio=0.2
os.makedirs('train_split/train', exist_ok=True)
os.makedirs('train_split/test', exist_ok=True)
for label in os.listdir(data_path):
    print(label)
    label_path=os.path.join(data_path,label)
    images = os.listdir(label_path)
    random.shuffle(images)
    train_size=int(len(images)*ratio)
    train_image=images[train_size:]
    test_image=images[:train_size]
    for image in train_image:
        src = os.path.join(label_path, image)
        dst = os.path.join('train_split/train', label)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)
    for image in test_image:
        src = os.path.join(label_path, image)
        dst = os.path.join('train_split/test', label)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)
'''
#数据集建立和预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(60),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
class My_dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.dataset = datasets.ImageFolder(self.root_dir, transform=transform)

    def __getitem__(self, index):
        return self.dataset[index]
    def __len__(self):
        return len(self.dataset)
#引入残差块的网络结构
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
#传统CNN网络
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
class Pretask_Network3(nn.Module):
    def __init__(self):
        super(Pretask_Network3, self).__init__()
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
        self.fc1 = nn.Linear(2048, 2)
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
#训练集和测试集的加载
dataset = My_dataset('train_split/train')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
dataset = My_dataset('train_split/test')
test_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
#cuda加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("train begin!")
#模型训练部分
def training():
    model = Pretask_Network3()#这个写要训练的网络的名字，不固定
    model.to(device)
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(0, 180):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)  # 将输入数据移到GPU上
            labels = labels.to(device)  # 将标签移到GPU上
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print('[Epoch: %d, Batch: %5d] Loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        accuracy = 100 * correct_predictions / total_samples
        print('Accuracy: %.2f%%' % accuracy)
    return model

#保存模型到本地
#model= training()
#torch.save(model, 'Pretask_Network3.pth')
#加载模型
total_samples = 0
correct_predictions = 0
model_test1=torch.load('Traditional_Network1.0.pth')
model_test1.eval()
model_test2=torch.load('Resnet_Network2.pth')
model_test2.eval()
model_test3=torch.load('Resnet_Network2_model1.0.pth')
model_test3.eval()
model_test4=torch.load('Resnet_Network3_model1.0.pth')
model_test4.eval()
#测试模型
'''
def showmax(lt):
    index1 = 0  # 记录出现次数最多的元素下标
    max = 0  # 记录最大的元素出现次数
    for i in range(len(lt)):
        flag = 0  # 记录每一个元素出现的次数
        for j in range(i + 1, len(lt)):  # 遍历i之后的元素下标
            if lt[j].equal(lt[i]):
                flag += 1  # 每当发现与自己相同的元素，flag+1
        if flag > max:  # 如果此时元素出现的次数大于最大值，记录此时元素的下标
            max = flag
            index1 = i
    return lt[index1]  # 返回出现最多的元
dataset = My_dataset('train_split/test')
test_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
predicted=[0,0,0,0]
for i,data in enumerate(test_loader):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs1 = model_test1(inputs)
    _, predicted[0] = torch.max(outputs1.data, 1)
    outputs2 = model_test2(inputs)
    _, predicted[1] = torch.max(outputs2.data, 1)
    outputs3 = model_test3(inputs)
    _, predicted[2] = torch.max(outputs3.data, 1)
    outputs4 = model_test4(inputs)
    _, predicted[3] = torch.max(outputs4.data, 1)
    total_samples += labels.size(0)
    correct_predictions += (showmax(predicted) == labels).sum().item()
accuracy = 100 * correct_predictions / total_samples
print('Accuracy: %.2f%%' % accuracy)'''
# 测试模型
# 测试模型
def evaluate_model(model1,model2,model3,model4):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    probability= []
    predicted = []
    true_labels = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            outputs3 = model3(inputs)
            outputs4 = model4(inputs)
            outputs=outputs1+outputs2+outputs3+outputs4
            outputs=outputs/4
            print(outputs)
            probs, preds = torch.max(outputs.data, 1)
            probability.extend(probs.tolist())
            predicted.extend(preds.tolist())
            true_labels.extend(labels.tolist())
    return probability,predicted, true_labels
# 测试模型
pobabilities,prediction, true_labels = evaluate_model(model_test1,model_test2,model_test3,model_test4)
print(prediction)