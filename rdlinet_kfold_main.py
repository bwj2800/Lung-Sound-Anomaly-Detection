import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model.rdlinet import RDLINet  # Import the RDLINet model
import random

# 데이터셋 경로
image_dir = 'data_4gr/mel_image'
model_save_path = './checkpoint/rdlinet_100_melonly.pth'

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
        transforms.Resize((64,38)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create the dataset
    dataset = CustomDataset(image_dir=image_dir, transform=transform)
    print("Dataset ready")

    # 데이터셋 분할
    seed = 42  # 원하는 시드 값으로 설정
    set_seed(seed)

    # Stratified K-Fold 설정
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    X = np.arange(len(dataset))
    y = np.array(dataset.labels)
    # acc, s_crackle, s_wheezle, s_both, S_e, S_p, S_c, f1score
    fold_metrics = []
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{skf.n_splits}")

        # Subset을 사용하여 train, val, test 데이터셋을 만듭니다.
        train_dataset = Subset(dataset, train_index)
        test_dataset = Subset(dataset, test_index)

        # 다시 Stratified K-Fold로 train과 val로 나눕니다.
        val_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(val_split.split(train_index, y[train_index]))

        val_dataset = Subset(dataset, val_idx)
        train_dataset = Subset(dataset, train_idx)

        # 데이터 로더 생성
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
        print("Dataset split")

        # 모델 초기화
        num_classes = len(label_map)
        model = RDLINet(num_classes)

        # 모델 학습
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
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
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

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
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    val_all_preds.extend(preds.cpu().numpy())
                    val_all_labels.extend(labels.cpu().numpy())

            val_epoch_loss = val_running_loss / len(val_loader)
            val_epoch_acc = accuracy_score(val_all_labels, val_all_preds)
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_all_labels, val_all_preds, average='weighted')

            val_loss_history.append(val_epoch_loss)
            val_acc_history.append(val_epoch_acc)

            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, "
                  f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1 Score: {val_f1:.4f}")

            # 모델 저장
            if val_epoch_acc > best_accuracy:
                best_accuracy = val_epoch_acc
                best_model = model.state_dict()
                torch.save(best_model, model_save_path)
                print(f"{epoch+1} Model saved")

        # 최적의 모델 로드
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
        # Fold 마다 지표 계산
        accuracy = 100 * correct / total
        s_crackle = avg_cm[1][1] / (avg_cm[1][0] + avg_cm[1][1] + avg_cm[1][2] + avg_cm[1][3])
        s_wheezle = avg_cm[2][2] / (avg_cm[2][0] + avg_cm[2][1] + avg_cm[2][2] + avg_cm[2][3])
        s_both = avg_cm[3][3] / (avg_cm[3][0] + avg_cm[3][1] + avg_cm[3][2] + avg_cm[3][3])
        
        S_e=(avg_cm[1][1]+avg_cm[2][2]+avg_cm[3][3] )/\
                        (avg_cm[1][0] + avg_cm[1][1] + avg_cm[1][2] + avg_cm[1][3]
                        +avg_cm[2][0] + avg_cm[2][1] + avg_cm[2][2] + avg_cm[2][3]
                        +avg_cm[3][0] + avg_cm[3][1] + avg_cm[3][2] + avg_cm[3][3])
        S_p=avg_cm[0][0]/(avg_cm[0][0]+avg_cm[0][1]+avg_cm[0][2]+avg_cm[0][3])
        S_c=(S_p+S_e)/2

        fold_metrics.append([accuracy, s_crackle, s_wheezle, s_both, S_e, S_p, S_c, f1])
        print(f'Accuracy on test set for fold {fold+1}: {accuracy:.2f}%')
        print(f'Crackle Accuracy: {s_crackle:.2%}')
        print(f'Wheeze Accuracy: {s_wheezle:.2%}')
        print(f'Both Accuracy: {s_both:.2%}')
        print("S_p: {}, S_e: {}, Score: {}".format(S_p, S_e, S_c))

    # 전체 Fold에 대한 성능 평균 출력

    # Fold별 평균 성능 계산
    fold_metrics = np.array(fold_metrics)
    mean_metrics = np.mean(fold_metrics, axis=0)

    # avg_accuracy = np.mean(fold_metrics)
    print(f'\nAverage Accuracy across all folds: {mean_metrics[0]:.2f}%')
    print(f"Average Crackle Sensitivity: {mean_metrics[1]:.4f}")
    print(f"Average Wheeze Sensitivity: {mean_metrics[2]:.4f}")
    print(f"Average Both Sensitivity: {mean_metrics[3]:.4f}")
    print(f"Average Sensitivity (S_e): {mean_metrics[4]:.4f}")
    print(f"Average Specificity (S_p): {mean_metrics[5]:.4f}")
    print(f"Average Score (S_c): {mean_metrics[6]:.4f}")
    print(f"Average F1 Score: {mean_metrics[7]:.4f}")

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
