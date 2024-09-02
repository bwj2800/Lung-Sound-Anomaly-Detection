import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

# from model.tcn import MultilevelTCNModel
from tcn_resnet50 import MultilevelTCNModel

# 데이터셋 경로
# image_dir = 'data_4gr/mel_image_new'
image_dir = './Dataset_ICBHI_Log-Melspec/Dataset_Task_1/Dataset_1_2'
model_save_path = './checkpoint/tcn_resnet50.pth'

# 라벨 매핑
label_map = {'normal': 0, 'crackle': 1, 'wheeze': 2, 'both': 3}

# 시드 고정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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

def train_and_evaluate():
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create the dataset
    dataset = CustomDataset(image_dir=image_dir, transform=transform)
    print("Dataset ready")

    seed=42
    set_seed(seed)

    # 이미지 경로와 라벨 리스트를 가져옵니다.
    image_paths = dataset.image_paths
    labels = dataset.labels

    # 전체 데이터셋을 60:40 비율로 train과 test로 나눕니다.
    train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.4, random_state=seed, stratify=labels)

    # train 데이터셋의 10%를 validation 데이터로 나눕니다.
    train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.1, random_state=seed, stratify=train_labels)

    # 각 서브셋에 대해 CustomDataset을 만듭니다.
    train_dataset = CustomDataset(image_dir=image_dir, transform=transform)
    val_dataset = CustomDataset(image_dir=image_dir, transform=transform)
    test_dataset = CustomDataset(image_dir=image_dir, transform=transform)

    # 필요한 경우 각 데이터셋의 이미지 경로와 라벨을 설정해줍니다.
    train_dataset.image_paths, train_dataset.labels = train_paths, train_labels
    val_dataset.image_paths, val_dataset.labels = val_paths, val_labels
    test_dataset.image_paths, test_dataset.labels = test_paths, test_labels

    # 데이터셋 분할
    # train_size = int(0.6 * len(dataset))
    # val_size = int(0.2 * len(dataset))
    # test_size = len(dataset) - train_size - val_size
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    print("Dataset split")

    # 모델 초기화
    num_classes = len(label_map)
    model = MultilevelTCNModel(num_classes)

    # 모델 학습
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    best_loss = float('inf')
    best_accuracy = 0.0
    best_model = None

    print("Start training")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss / len(train_loader)}, Val Accuracy: {accuracy}%')

        # 모델 저장
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.state_dict()
            torch.save(best_model, model_save_path)
            print("Model saved")


    model.load_state_dict(best_model)
    model.eval()
    correct = 0
    total = 0
    avg_cm = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    with torch.no_grad():
        for images, labels in test_loader:
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

if __name__ == '__main__':
    train_and_evaluate()