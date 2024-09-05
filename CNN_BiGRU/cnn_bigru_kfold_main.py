import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import sys
sys.path.append(os.path.abspath('./'))
from model.cnn_bigru import CNN_BiGRU

# 데이터셋 경로
# image_dir = 'data_4gr/mel_image_cnn_lstm'
image_dir = './data_4gr/0822/Task2_2'
model_save_dir = './checkpoint/'

# 라벨 매핑
# label_map = {'normal': 0, 'crackle': 1, 'wheeze': 2, 'both': 3}
label_map = {'Healthy': 0, 'Chronic': 1, 'Non-Chronic': 2}

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
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
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
        # transforms.Resize((64, 64)),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Create the dataset
    dataset = CustomDataset(image_dir=image_dir, transform=transform)
    print("Dataset ready")

    # 데이터셋 분할
    seed = 42
    set_seed(seed)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    X = np.arange(len(dataset))
    y = np.array(dataset.labels)
    fold_metrics = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{skf.n_splits}")

        # Train and test split
        train_dataset = Subset(dataset, train_index)
        test_dataset = Subset(dataset, test_index)

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
        print("Dataset split")

        # Model initialization
        num_class = 3
        # num_class = 4
        model = CNN_BiGRU(input_channels=80, num_branches=3, num_layers_per_branch=3, dilation_base=[2,3,4], hidden_dim1=80, hidden_dim2=32, num_classes=num_class)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=1e-5)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 100
        best_accuracy = 0.0
        best_model = None

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

            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

            # Validation loop
            model.eval()
            val_running_loss = 0.0
            val_all_preds = []
            val_all_labels = []

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    val_all_preds.extend(preds.cpu().numpy())
                    val_all_labels.extend(labels.cpu().numpy())

            val_epoch_loss = val_running_loss / len(test_loader)
            val_epoch_acc = accuracy_score(val_all_labels, val_all_preds)

            val_loss_history.append(val_epoch_loss)
            val_acc_history.append(val_epoch_acc)

            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

            # Save the model if it is the best so far
            if val_epoch_acc > best_accuracy:
                best_accuracy = val_epoch_acc
                best_model = model.state_dict()
                torch.save(best_model, os.path.join(model_save_dir, f'cnn_bigru_fold_{fold+1}_0822.pth'))
                print(f"Best model saved for fold {fold+1}")

        # Load the best model for the fold
        model.load_state_dict(best_model)
        model.eval()
        correct = 0
        total = 0
        # avg_cm = np.zeros((4, 4), dtype=int)  # Reset confusion matrix for each fold
        avg_cm = np.zeros((3, 3), dtype=int)  # Reset confusion matrix for each fold

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Compute confusion matrix
                # cm = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1, 2, 3])
                cm = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1, 2])
                avg_cm += cm

        # Fold metrics calculation
        accuracy = 100 * correct / total
        # s_crackle = avg_cm[1][1] / avg_cm[1].sum() if avg_cm[1].sum() > 0 else 0
        # s_wheeze = avg_cm[2][2] / avg_cm[2].sum() if avg_cm[2].sum() > 0 else 0
        # s_both = avg_cm[3][3] / avg_cm[3].sum() if avg_cm[3].sum() > 0 else 0

        # S_e = (avg_cm[1][1] + avg_cm[2][2] + avg_cm[3][3]) / np.sum(avg_cm[1:4, :])
        # S_p = avg_cm[0][0] / np.sum(avg_cm[0, :]) if np.sum(avg_cm[0, :]) > 0 else 0
        s_crackle = avg_cm[1][1] / (avg_cm[1][0] + avg_cm[1][1] + avg_cm[1][2])
        s_wheeze = avg_cm[2][2] / (avg_cm[2][0] + avg_cm[2][1] + avg_cm[2][2])
        
        S_e=(avg_cm[1][1]+avg_cm[2][2])/\
                        (avg_cm[1][0] + avg_cm[1][1] + avg_cm[1][2]
                        +avg_cm[2][0] + avg_cm[2][1] + avg_cm[2][2])
        S_p=avg_cm[0][0]/(avg_cm[0][0]+avg_cm[0][1]+avg_cm[0][2])
        
        S_c = (S_p + S_e) / 2

        # fold_metrics.append([accuracy, s_crackle, s_wheeze, s_both, S_e, S_p, S_c])
        fold_metrics.append([accuracy, s_crackle, s_wheeze, 0, S_e, S_p, S_c])
        print(f'Accuracy on test set for fold {fold+1}: {accuracy:.2f}%')
        print(f'Crackle Sensitivity: {s_crackle:.4f}')
        print(f'Wheeze Sensitivity: {s_wheeze:.4f}')
        # print(f'Both Sensitivity: {s_both:.4f}')
        print(f"S_p: {S_p:.4f}, S_e: {S_e:.4f}, Score: {S_c:.4}")
        
    # Average metrics across all folds
    fold_metrics = np.array(fold_metrics)
    mean_metrics = np.mean(fold_metrics, axis=0)

    print(f'\nAverage Accuracy across all folds: {mean_metrics[0]:.2f}%')
    print(f"Average Crackle Sensitivity: {mean_metrics[1]:.4f}")
    print(f"Average Wheeze Sensitivity: {mean_metrics[2]:.4f}")
    # print(f"Average Both Sensitivity: {mean_metrics[3]:.4f}")
    print(f"Average Sensitivity (S_e): {mean_metrics[4]:.4f}")
    print(f"Average Specificity (S_p): {mean_metrics[5]:.4f}")
    print(f"Average Score (S_c): {mean_metrics[6]:.4f}")
    
if __name__ == '__main__':
    train_and_evaluate()