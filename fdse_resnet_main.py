import os
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import KernelPCA
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

# 데이터셋 경로
image_dir = 'data_4gr/mel_image'
source_dir = './mat_462/'

# 라벨 매핑
label_map = {'normal': 0, 'crackle': 1, 'wheeze': 2, 'both': 3}

# 추가 feature 읽기 및 전처리 함수
def load_and_preprocess_features():
    normal_features = sio.loadmat(os.path.join(source_dir, 'normal_462.mat'))['normal']
    crackle_features = sio.loadmat(os.path.join(source_dir, 'crackle_462.mat'))['crackle']
    wheeze_features = sio.loadmat(os.path.join(source_dir, 'wheeze_462.mat'))['wheeze']
    both_features = sio.loadmat(os.path.join(source_dir, 'both_462.mat'))['both']

    normal_X = normal_features[:, :-1]
    crackle_X = crackle_features[:, :-1]
    wheeze_X = wheeze_features[:, :-1]
    both_X = both_features[:, :-1]

    normal_y = normal_features[:, -1]
    crackle_y = crackle_features[:, -1]
    wheeze_y = wheeze_features[:, -1]
    both_y = both_features[:, -1]

    X = np.concatenate((normal_X, crackle_X, wheeze_X, both_X), axis=0)
    y = np.concatenate((normal_y, crackle_y, wheeze_y, both_y), axis=0)

    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    joblib.dump(min_max_scaler, 'scaler.pkl')
    print("scaler saved")

    transformer = KernelPCA(n_components=184, kernel='linear')
    X = transformer.fit_transform(X)
    joblib.dump(transformer, 'transformer.pkl')
    print("transformer saved")

    return X, y

# 추가 feature 로드
X_features, y = load_and_preprocess_features()
print("mat file loaded")

# 커스텀 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, image_dir, features, labels, transform=None):
        self.image_dir = image_dir
        self.features = features
        self.labels = labels
        self.transform = transform
        self.image_paths = []
        for label in label_map.keys():
            image_folder = os.path.join(image_dir, label)
            for img_file in os.listdir(image_folder):
                self.image_paths.append(os.path.join(image_folder, img_file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = label_map[os.path.basename(os.path.dirname(image_path))]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        feature = self.features[index]
        feature = torch.tensor(feature, dtype=torch.float32)

        return image, feature, label

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 데이터셋 생성
dataset = CustomDataset(image_dir=image_dir, features=X_features, labels=y, transform=transform)
print("dataset ready")

# 데이터셋 분할
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("dataset split")

# ResNet 모델 수정
class MultiInputResNet(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(MultiInputResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(num_features + feature_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x_image, x_features):
        x = self.resnet(x_image)
        x = torch.cat((x, x_features), dim=1)
        x = nn.ReLU()(self.fc1(x))
        x = nn.Dropout(0.5)(x)
        x = self.fc2(x)
        return x

# 모델 초기화
feature_dim = 184  # PCA로 축소된 추가 특징의 차원
num_classes = len(label_map)
model = MultiInputResNet(num_classes, feature_dim)

# 모델 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    for i, (images, features, labels) in tqdm(enumerate(train_loader)):
        images, features, labels = images.to(device), features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            images, features, labels = data
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            outputs = model(images, features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss / len(train_loader)}, Val Accuracy: {accuracy}%')

    # 모델 저장
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model.state_dict()
        torch.save(best_model, 'best_resnet_fdse_1.pth')
        print("Model saved")

model.load_state_dict(best_model)
model.eval()
correct = 0
total = 0
avg_cm = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

with torch.no_grad():
    for data in test_loader:
        images, features, labels = data
        images, features, labels = images.to(device), features.to(device), labels.to(device)
        outputs = model(images, features)
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
