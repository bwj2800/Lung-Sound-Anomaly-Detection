import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 데이터셋 경로
data_dir = 'data_4gr/mel_image'
model_path='best_resnet18_3.pth'

# 라벨 매핑
label_map = {'normal': 0, 'crackle': 1, 'wheeze': 2, 'both': 3}

class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(path)
        return sample, target
    
# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 데이터셋 생성
# dataset = MelImageDataset(data_dir, transform=transform)

def get_class_from_path(path):
    return label_map[os.path.basename(os.path.dirname(path))]

dataset = CustomImageFolder(root=data_dir, transform=transform,
                            target_transform=lambda x: get_class_from_path(x))
dataset.classes = [0, 1, 2, 3]


# 랜덤 시드 값 설정
seed = 42
train_size = int(0.6 * len(dataset))
test_size = len(dataset) - train_size

# 랜덤 시드 값을 가진 generator 생성
generator = torch.Generator().manual_seed(seed)

# 훈련 데이터와 테스트 데이터 분할
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# train_dataset, test_dataset = train_test_split(dataset, test_size=0.4, random_state=42)

print("dataset loaded")

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("data loader ready")

# ResNet 모델 사용
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(label_map))

# 모델 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)
model.to(device)

model.load_state_dict(torch.load(model_path))
model.eval()


print(model_path+" model loaded")

correct = 0
total = 0
avg_cm = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

with torch.no_grad():
    for data in tqdm(test_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 혼동 행렬 계산
        for i in range(len(labels)):
            avg_cm[labels[i]][predicted[i]] += 1

    print(f'Accuracy on test set: {100 * correct / total}%')

    # 클래스별 성능 계산
    s_crackle = avg_cm[1][1] / (avg_cm[1][0] + avg_cm[1][1] + avg_cm[1][2] + avg_cm[1][3])
    s_wheezle = avg_cm[2][2] / (avg_cm[2][0] + avg_cm[2][1] + avg_cm[2][2] + avg_cm[2][3])
    s_both = avg_cm[3][3] / (avg_cm[3][0] + avg_cm[3][1] + avg_cm[3][2] + avg_cm[3][3])
    
    S_e=(avg_cm[1][1]+avg_cm[2][2]+avg_cm[3][3] )/\
                      (avg_cm[1][0] + avg_cm[1][1] + avg_cm[1][2] + avg_cm[1][3]
                      +avg_cm[2][0] + avg_cm[2][1] + avg_cm[2][2] + avg_cm[2][3]
                      +avg_cm[3][0] + avg_cm[3][1] + avg_cm[3][2] + avg_cm[3][3])
    S_p=avg_cm[0][0]/(avg_cm[0][0]+avg_cm[0][1]+avg_cm[0][2]+avg_cm[0][3])
    S_c=(S_p+S_e)/2

    print(f'Crackle Accuracy: {s_crackle:.2%}')
    print(f'Wheeze Accuracy: {s_wheezle:.2%}')
    print(f'Both Accuracy: {s_both:.2%}')
    print("S_p: {}, S_e: {}, Score: {}".format(S_p, S_e, S_c))