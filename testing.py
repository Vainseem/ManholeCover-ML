import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from model_training import Resnet_Network3,MaxPooling_CNN,ResidualBlock,Regular_CNN,Traditional_Network1,Resnet_Network1,Resnet_Network2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the test dataset
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.root_dir)[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return img_name, image

test_dataset = TestDataset(r"C:\Users\19588\Desktop\2024服创大赛A03井盖数据完整\井盖测试集\测试集图片", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the trained models
model1 = torch.load('Resnet_Network2_model2.0.pth')
model2 = torch.load('Resnet_Network2_model1.0.pth')
model3 = torch.load('Resnet_Network3_model1.0.pth')
model4 = torch.load('Traditional_Network1.0.pth')

# Set the models to evaluation mode
model1.eval()
model2.eval()
model3.eval()
model4.eval()

# Perform inference and generate predictions using voting
results = []
for img_name, img in test_loader:
    with torch.no_grad():
        img = img.to(device)
        output1 = model1(img)
        output2 = model2(img)
        output3 = model3(img)
        output4 = model4(img)

        # Perform voting based on predictions from each model
        predictions = [torch.argmax(output1).item(), torch.argmax(output2).item(),
                       torch.argmax(output3).item(), torch.argmax(output4).item()]
        vote_result = Counter(predictions).most_common(1)[0][0]

        results.append((img_name, vote_result))

# Write the results to a text file
with open('voting_predictions.txt', 'w') as f:
    for result in results:
        line = f"{result[0]} {result[1]}\n"
        f.write(line)