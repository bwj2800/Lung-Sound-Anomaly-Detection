import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
import numpy as np
import random

sys.path.append(os.path.abspath('./'))
from model.cnn_lstm import CNN_LSTM

# 데이터셋 경로
image_dir = './data_4gr/mel_image_cnn_lstm_2class'
model_save_path = './checkpoint/cnn_lstm_2class_1.pth'

# 라벨 매핑
label_map = {'normal': 0, 'crackle': 1, 'wheeze': 1, 'both': 1}

seed = 42  # 원하는 시드 값으로 설정
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Custom Dataset class definition
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label_name, label_idx in label_map.items():
            image_folder = os.path.join(image_dir, label_name)
            for img_file in os.listdir(image_folder):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Include only image files
                    self.image_paths.append(os.path.join(image_folder, img_file))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create the dataset
dataset = CustomDataset(image_dir=image_dir, transform=transform)
print("Dataset ready")

# 데이터셋 분할
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
print("Dataset split")

# 모델 초기화
num_class = 2
model = CNN_LSTM(num_class=num_class)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.load_state_dict(torch.load(model_save_path, weights_only=True))
model.eval()
correct = 0
total = 0
avg_cm = [[0, 0], [0, 0]]

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 혼동 행렬 계산
        for i in range(num_class):
            avg_cm[labels[i]][predicted[i]] += 1

    print(f'Accuracy on test set: {100 * correct / total}%')

    # 클래스별 성능 계산    
    S_e=avg_cm[1][1]/(avg_cm[1][0]+avg_cm[1][1])
    S_p=avg_cm[0][0]/(avg_cm[0][0]+avg_cm[0][1])
    S_c=(S_p+S_e)/2
    print("S_p: {}, S_e: {}, Score: {}".format(S_p, S_e, S_c))
