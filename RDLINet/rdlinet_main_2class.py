import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
# from model.rdlinet import RDLINet  # Import the RDLINet model
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from rdlinet import RDLINet  # Import the RDLINet model
import random
import numpy as np
from sklearn.model_selection import train_test_split

# 데이터셋 경로
# image_dir = 'data_4gr/mel_image'
image_dir = './Aug/Task1_1'
# image_dir = './Dataset_ICBHI_Log-Melspec/Dataset_Task_1/Dataset_1_2'
model_save_path = './checkpoint/rdlinet_binary_augmented.pth'

# 라벨 매핑 - Binary Classification
# label_map = {'normal': 0, 'crackle': 1, 'wheeze': 1, 'both': 1}
label_map = {'normal': 0, 'abnormal': 1}

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

    # 데이터셋 분할
    seed = 42  # 원하는 시드 값으로 설정
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

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    

    # 모델 초기화
    num_classes = 1  # Binary classification이므로 클래스 수를 1로 설정
    model = RDLINet(num_classes)

    # 모델 학습
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # BinaryCrossEntropyLoss 사용
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    best_accuracy = 0.0
    best_model = None

    # 학습 기록 저장
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    print("Start training")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs)).detach()  # 이진 분류의 경우 0.5를 기준으로 예측
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
            f"Train Precision: {precision:.4f}, Train Recall: {recall:.4f}, Train F1 Score: {f1:.4f}")

        # 검증 루프
        model.eval()
        val_running_loss = 0.0
        val_all_preds = []
        val_all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs)).detach()
                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = accuracy_score(val_all_labels, val_all_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_all_labels, val_all_preds, average='binary')

        val_loss_history.append(val_epoch_loss)
        val_acc_history.append(val_epoch_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, "
            f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1 Score: {val_f1:.4f}")

        # 모델 저장
        if val_epoch_acc > best_accuracy:
            best_accuracy = val_epoch_acc
            best_model = model.state_dict()
            torch.save(best_model, model_save_path)
            print(f"Model saved at epoch {epoch+1}")

    # 최적의 모델 로드
    model.load_state_dict(best_model)
    model.eval()
    correct = 0
    total = 0
    avg_cm = np.zeros((2, 2))  # 이진 분류를 위한 2x2 혼동 행렬

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images).squeeze(1)
            preds = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            cm = confusion_matrix(labels.cpu(), preds.cpu())
            avg_cm += cm

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

    # 성능 지표 계산
    s_normal = avg_cm[0, 0] / avg_cm[0].sum()
    s_abnormal = avg_cm[1, 1] / avg_cm[1].sum()
    
    S_e = s_abnormal
    S_p = s_normal
    S_c = (S_p + S_e) / 2

    f1 = 2 * (S_p * S_e) / (S_p + S_e)

    print(f'Normal Sensitivity: {s_normal:.2%}')
    print(f'Abnormal Sensitivity: {s_abnormal:.2%}')
    print(f"S_p: {S_p:.4f}, S_e: {S_e:.4f}, S_c: {S_c:.4f}")
    
    # 학습 결과 시각화 및 저장
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss_history, label='Train Loss')
    plt.plot(epochs, val_loss_history, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_acc_history, label='Train Accuracy')
    plt.plot(epochs, val_acc_history, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == '__main__':
    train_and_evaluate()