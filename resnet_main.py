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
data_dir = 'data_4gr/mel_image_new'

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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 데이터셋 생성
def get_class_from_path(path):
    return label_map[os.path.basename(os.path.dirname(path))]

dataset = CustomImageFolder(root=data_dir, transform=transform,
                            target_transform=lambda x: get_class_from_path(x))
dataset.classes = [0, 1, 2, 3]

# 랜덤 시드 값 설정
seed = 42
train_size = int(0.6 * len(dataset))
valid_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - valid_size

# 랜덤 시드 값을 가진 generator 생성
generator = torch.Generator().manual_seed(seed)

# 훈련, 검증, 테스트 데이터 분할
train_dataset, valid_test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size + test_size], generator=generator)
valid_dataset, test_dataset = torch.utils.data.random_split(valid_test_dataset, [valid_size, test_size], generator=generator)

print("dataset loaded")

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("data loader ready")

# ResNet 모델 사용
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(label_map))

print("model loaded")

# 모델 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
best_loss = float('inf')
best_accuracy = 0.0
best_model = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss / len(train_loader)}, Valid Accuracy: {valid_accuracy}%')

    # 모델 저장
    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        best_model = model.state_dict()
        torch.save(best_model, 'best_resnet18_4.pth')
        print("Model saved")

# 최적의 모델 저장
model.load_state_dict(best_model)
model.eval()
correct = 0
total = 0
avg_cm = np.zeros((4, 4))

with torch.no_grad():
    for data in test_loader:
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
    for i, cls in enumerate(['normal', 'crackle', 'wheeze', 'both']):
        s = avg_cm[i][i] / np.sum(avg_cm[i])
        print(f'{cls}: {s:.2%}')
