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
prediction=[]
new_name={}
new_name['broke']=0
new_name['circle']=0
new_name['good']=0
new_name['lose']=0
new_name['uncovered']=0
print("预测")
dataset= My_dataset('prediction')
prediction_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
for i,image in enumerate(prediction_loader):
    inputs, labels = image
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model_test(inputs)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
    predicted=predicted.tolist()
    if predicted[0]==0:
        new_name['broke']+=1
        prediction.append('broke'+str(new_name['broke'])+'.jpg')
    if predicted[0]==1:
        new_name['circle']+=1
        prediction.append('circle'+str(new_name['circle'])+'.jpg')
    if predicted[0]==2:
        new_name['good']+=1
        prediction.append('good'+str(new_name['good'])+'.jpg')
    if predicted[0]==3:
        new_name['lose']+=1
        prediction.append('lose'+str(new_name['lose'])+'.jpg')
    if predicted[0]==4:
        new_name['uncovered']+=1
        prediction.append('uncovered'+str(new_name['uncovered'])+'.jpg')
print(prediction)
images = os.listdir('prediction/new_train')
files = os.listdir('prediction/new_train')
image_files = [f for f in files]
# 遍历文件列表和重命名列表，进行重命名
for i, image_file in enumerate(image_files):
    new_name = prediction[i]
    old_path = os.path.join('prediction/new_train', image_file)
    new_path = os.path.join('prediction/new_train', new_name)
    os.rename(old_path, new_path)
'''