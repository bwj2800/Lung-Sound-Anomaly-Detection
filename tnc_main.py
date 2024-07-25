import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

# Define the residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        if self.skip_connection:
            x = self.skip_connection(x)
        return self.relu(x + res)

# Define the TCN network
class TCNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TCNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.pool1 = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.15)
        self.pool2 = nn.AvgPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.2)
        self.pool3 = nn.AvgPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.25)

        self.conv6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout6 = nn.Dropout(0.25)

        self.residual_blocks = nn.ModuleList()
        dilation_bases = [2, 3]
        for dilation_base in dilation_bases:
            for i in range(3):
                dilation_rate = dilation_base ** i
                self.residual_blocks.append(ResidualBlock(512, 512, dilation_rate))

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.pool3(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.dropout6(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

# Define your training and evaluation functions
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return np.array(y_true), np.array(y_pred)

# Define hyperparameters and initialize model
input_size = feature_size  # Adjust based on your input size
num_classes = 4
model = TCNModel(input_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate the model
best_test_acc = 0.0
for i in range(5):
    # Split the dataset into training, validation, and test sets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Train the model
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    
    # Evaluate the model
    y_true, y_pred = evaluate_model(model, test_loader, device)
    test_acc = np.mean(y_true == y_pred)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), f'./model_seed/LN_ML_TCN/60_40/Sequence/Dup_Trun/model_Aug_{i}.pth')

    # Compute the confusion matrix and classification metrics
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    accuracy = np.diag(cm) / np.sum(cm, axis=1)
    specificity = np.diag(cm) / (np.diag(cm) + np.sum(cm, axis=0) - np.diag(cm))

    print(f"Fold {i+1} Metrics:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Specificity: {specificity}")

# Print overall scores
print("\nAverage scores:")
print(f"Best Test Accuracy: {best_test_acc:.4f}")

################################################################################################################

################################################################################################################

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
    s_normal = avg_cm[0][0] / (avg_cm[0][0] + avg_cm[0][1] + avg_cm[0][2] + avg_cm[0][3])
    s_crackle = avg_cm[1][1] / (avg_cm[1][0] + avg_cm[1][1] + avg_cm[1][2] + avg_cm[1][3])
    s_wheezle = avg_cm[2][2] / (avg_cm[2][0] + avg_cm[2][1] + avg_cm[2][2] + avg_cm[2][3])
    s_both = avg_cm[3][3] / (avg_cm[3][0] + avg_cm[3][1] + avg_cm[3][2] + avg_cm[3][3])

    print(f'normal: {s_normal:.2%}')
    print(f'Crackle: {s_crackle:.2%}')
    print(f'Wheezle: {s_wheezle:.2%}')
    print(f'Both: {s_both:.2%}')
