import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

# from multi_input_to_backbone_tcn_ import MultilevelTCNModel
from multi_input_tcn import MultilevelTCNModel

# 데이터셋 경로
mel_dir = 'data_4gr/mel_image'
# mel_dir = './Dataset_ICBHI_Log-Melspec/Dataset_Task_1/Dataset_1_2'
chroma_dir = 'data_4gr/chroma_image'
mfcc_dir = 'data_4gr/mfcc_image'
model_save_path = './checkpoint/tcn_vgg19_pt_ml.pth'

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

# Custom Dataset class 정의
class CustomDataset(Dataset):
    def __init__(self, mel_dir, chroma_dir, mfcc_dir, transform=None):
        self.mel_dir = mel_dir
        self.chroma_dir = chroma_dir
        self.mfcc_dir = mfcc_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label_name, label_idx in label_map.items():
            mel_folder = os.path.join(mel_dir, label_name)
            chroma_folder = os.path.join(chroma_dir, label_name)
            mfcc_folder = os.path.join(mfcc_dir, label_name)
            for img_file in os.listdir(mel_folder):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    mel_path = os.path.join(mel_folder, img_file)
                    chroma_path = os.path.join(chroma_folder, img_file)
                    mfcc_path = os.path.join(mfcc_folder, img_file)
                    if os.path.exists(chroma_path) and os.path.exists(mfcc_path):
                        self.image_paths.append((mel_path, chroma_path, mfcc_path))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        mel_path, chroma_path, mfcc_path = self.image_paths[index]
        label = self.labels[index]
        mel_image = Image.open(mel_path).convert('RGB')
        chroma_image = Image.open(chroma_path).convert('RGB')
        mfcc_image = Image.open(mfcc_path).convert('RGB')
        if self.transform:
            mel_image = self.transform(mel_image)
            chroma_image = self.transform(chroma_image)
            mfcc_image = self.transform(mfcc_image)
        return mel_image, chroma_image, mfcc_image, label

def train_and_evaluate():
    final_epoch = 0
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create the dataset
    dataset = CustomDataset(mel_dir=mel_dir, chroma_dir=chroma_dir, mfcc_dir=mfcc_dir, transform=transform)
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
    train_dataset = CustomDataset(mel_dir=mel_dir, chroma_dir=chroma_dir, mfcc_dir=mfcc_dir, transform=transform)
    val_dataset = CustomDataset(mel_dir=mel_dir, chroma_dir=chroma_dir, mfcc_dir=mfcc_dir, transform=transform)
    test_dataset = CustomDataset(mel_dir=mel_dir, chroma_dir=chroma_dir, mfcc_dir=mfcc_dir, transform=transform)

    # 필요한 경우 각 데이터셋의 이미지 경로와 라벨을 설정해줍니다.
    train_dataset.image_paths, train_dataset.labels = train_paths, train_labels
    val_dataset.image_paths, val_dataset.labels = val_paths, val_labels
    test_dataset.image_paths, test_dataset.labels = test_paths, test_labels

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # train_size = int(0.6 * len(dataset))
    # val_size = int(0.2 * len(dataset))
    # test_size = len(dataset) - train_size - val_size
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # # 데이터 로더 생성
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    # print("Dataset split")

    
    # 데이터셋 분할 후, 각 데이터셋의 크기 출력
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    # 모델 초기화
    num_classes = len(label_map)
    model = MultilevelTCNModel(num_classes)

    # 모델 학습
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    num_epochs = 100
    best_loss = float('inf')
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
        correct = 0
        total = 0

        for i, (mel_images, chroma_images, mfcc_images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            mel_images, chroma_images, mfcc_images, labels = mel_images.to(device), chroma_images.to(device), mfcc_images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mel_images, chroma_images, mfcc_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_acc = 100 * correct / total
        train_loss_history.append(running_loss / len(train_loader))
        train_acc_history.append(epoch_acc)

        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        val_all_labels = []
        val_all_predictions = []
        model.eval()
        with torch.no_grad():
            for mel_images, chroma_images, mfcc_images, labels in tqdm(val_loader, desc="Validating"):
                mel_images, chroma_images, mfcc_images, labels = mel_images.to(device), chroma_images.to(device), mfcc_images.to(device), labels.to(device)
                outputs = model(mel_images, chroma_images, mfcc_images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_all_labels.extend(labels.cpu().numpy())
                val_all_predictions.extend(predicted.cpu().numpy())

        
        accuracy = 100 * val_correct / val_total
        val_acc_history.append(accuracy)
        val_acc_history.append(val_running_loss / len(val_loader))

        precision = precision_score(val_all_labels, val_all_predictions, average='weighted')
        recall = recall_score(val_all_labels, val_all_predictions, average='weighted')
        f1 = f1_score(val_all_labels, val_all_predictions, average='weighted')
        conf_matrix = confusion_matrix(val_all_labels, val_all_predictions)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss / len(train_loader)}, Val Accuracy: {accuracy}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

        # 모델 저장
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.state_dict()
            torch.save(best_model, model_save_path)
            print(f"Model saved, best at {epoch+1}")
            final_epoch = epoch+1

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
    plt.savefig('training_results_tcn.png')
    plt.show()

    model.load_state_dict(best_model)
    model.eval()
    correct = 0
    total = 0
    avg_cm = np.zeros((4, 4))

    with torch.no_grad():
        all_labels = []
        all_predictions = []
        for mel_images, chroma_images, mfcc_images, labels in tqdm(test_loader, desc="Testing"):
            mel_images, chroma_images, mfcc_images, labels = mel_images.to(device), chroma_images.to(device), mfcc_images.to(device), labels.to(device)
            outputs = model(mel_images, chroma_images, mfcc_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # 혼동 행렬 계산
            for i in range(len(labels)):
                avg_cm[labels[i]][predicted[i]] += 1

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f'Accuracy on test set: {accuracy}%')
    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

    # # Confusion Matrix 시각화
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix')
    # plt.savefig('confusion_matrix.png')
    # plt.show()

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
    print(f'Best Val Accuracy recorded at {final_epoch}!')

if __name__ == '__main__':
    train_and_evaluate()

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from PIL import Image
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from model.multi_input_tcn import MultilevelTCNModel

# # 데이터셋 경로
# mel_dir = 'data_4gr/mel_image'
# chroma_dir = 'data_4gr/chroma_image'
# mfcc_dir = 'data_4gr/mfcc_image'
# model_save_path = './checkpoint/tcn_vgg19_pt_ml.pth'

# # 라벨 매핑
# label_map = {'normal': 0, 'crackle': 1, 'wheeze': 2, 'both': 3}

# # Custom Dataset class 정의
# class CustomDataset(Dataset):
#     def __init__(self, mel_dir, chroma_dir, mfcc_dir, transform=None):
#         self.mel_dir = mel_dir
#         self.chroma_dir = chroma_dir
#         self.mfcc_dir = mfcc_dir
#         self.transform = transform
#         self.image_paths = []
#         self.labels = []
#         for label_name, label_idx in label_map.items():
#             mel_folder = os.path.join(mel_dir, label_name)
#             chroma_folder = os.path.join(chroma_dir, label_name)
#             mfcc_folder = os.path.join(mfcc_dir, label_name)
#             for img_file in os.listdir(mel_folder):
#                 if img_file.endswith(('.png', '.jpg', '.jpeg')):
#                     mel_path = os.path.join(mel_folder, img_file)
#                     chroma_path = os.path.join(chroma_folder, img_file)
#                     mfcc_path = os.path.join(mfcc_folder, img_file)
#                     if os.path.exists(chroma_path) and os.path.exists(mfcc_path):
#                         self.image_paths.append((mel_path, chroma_path, mfcc_path))
#                         self.labels.append(label_idx)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         mel_path, chroma_path, mfcc_path = self.image_paths[index]
#         label = self.labels[index]
#         mel_image = Image.open(mel_path).convert('RGB')
#         chroma_image = Image.open(chroma_path).convert('RGB')
#         mfcc_image = Image.open(mfcc_path).convert('RGB')
#         if self.transform:
#             mel_image = self.transform(mel_image)
#             chroma_image = self.transform(chroma_image)
#             mfcc_image = self.transform(mfcc_image)
#         return mel_image, chroma_image, mfcc_image, label

# # Data preprocessing
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Create the dataset
# dataset = CustomDataset(mel_dir=mel_dir, chroma_dir=chroma_dir, mfcc_dir=mfcc_dir, transform=transform)
# print("Dataset ready")

# # 데이터셋 분할
# train_size = int(0.6 * len(dataset))
# val_size = int(0.2 * len(dataset))
# test_size = len(dataset) - train_size - val_size
# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# # 데이터 로더 생성
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
# print("Dataset split")

# # 모델 초기화
# num_classes = len(label_map)
# model = MultilevelTCNModel(num_classes)

# # 모델 학습
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 100
# best_loss = float('inf')
# best_accuracy = 0.0
# best_model = None

# train_losses = []
# val_accuracies = []

# print("Start training")

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for i, (mel_images, chroma_images, mfcc_images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
#         mel_images, chroma_images, mfcc_images, labels = mel_images.to(device), chroma_images.to(device), mfcc_images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(mel_images, chroma_images, mfcc_images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     train_losses.append(running_loss / len(train_loader))

#     correct = 0
#     total = 0
#     all_labels = []
#     all_predictions = []
#     model.eval()
#     with torch.no_grad():
#         for mel_images, chroma_images, mfcc_images, labels in tqdm(val_loader, desc="Validating"):
#             mel_images, chroma_images, mfcc_images, labels = mel_images.to(device), chroma_images.to(device), mfcc_images.to(device), labels.to(device)
#             outputs = model(mel_images, chroma_images, mfcc_images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             all_labels.extend(labels.cpu().numpy())
#             all_predictions.extend(predicted.cpu().numpy())

#     accuracy = 100 * correct / total
#     val_accuracies.append(accuracy)
#     precision = precision_score(all_labels, all_predictions, average='weighted')
#     recall = recall_score(all_labels, all_predictions, average='weighted')
#     f1 = f1_score(all_labels, all_predictions, average='weighted')
#     conf_matrix = confusion_matrix(all_labels, all_predictions)

#     print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss / len(train_loader)}, Val Accuracy: {accuracy}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

#     # 모델 저장
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_model = model.state_dict()
#         torch.save(best_model, model_save_path)
#         print("Model saved")

# # Loss 시각화
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Train Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.legend()
# plt.savefig('train_loss.png')
# # plt.show()

# # Accuracy 시각화
# plt.figure(figsize=(10, 5))
# plt.plot(val_accuracies, label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.title('Validation Accuracy')
# plt.legend()
# plt.savefig('val_accuracy.png')
# # plt.show()

# model.load_state_dict(best_model)
# model.eval()
# correct = 0
# total = 0
# avg_cm = np.zeros((4, 4))

# with torch.no_grad():
#     all_labels = []
#     all_predictions = []
#     for mel_images, chroma_images, mfcc_images, labels in tqdm(test_loader, desc="Testing"):
#         mel_images, chroma_images, mfcc_images, labels = mel_images.to(device), chroma_images.to(device), mfcc_images.to(device), labels.to(device)
#         outputs = model(mel_images, chroma_images, mfcc_images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         all_labels.extend(labels.cpu().numpy())
#         all_predictions.extend(predicted.cpu().numpy())

#         # 혼동 행렬 계산
#         for i in range(len(labels)):
#             avg_cm[labels[i]][predicted[i]] += 1

#     accuracy = 100 * correct / total
#     precision = precision_score(all_labels, all_predictions, average='weighted')
#     recall = recall_score(all_labels, all_predictions, average='weighted')
#     f1 = f1_score(all_labels, all_predictions, average='weighted')

#     print(f'Accuracy on test set: {accuracy}%')
#     print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

#     # Confusion Matrix 시각화
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.savefig('confusion_matrix.png')
#     # plt.show()
    
#     # 클래스별 성능 계산
#     s_normal = avg_cm[0][0] / (avg_cm[0][0] + avg_cm[0][1] + avg_cm[0][2] + avg_cm[0][3])
#     s_crackle = avg_cm[1][1] / (avg_cm[1][0] + avg_cm[1][1] + avg_cm[1][2] + avg_cm[1][3])
#     s_wheezle = avg_cm[2][2] / (avg_cm[2][0] + avg_cm[2][1] + avg_cm[2][2] + avg_cm[2][3])
#     s_both = avg_cm[3][3] / (avg_cm[3][0] + avg_cm[3][1] + avg_cm[3][2] + avg_cm[3][3])

#     print(f'normal: {s_normal:.2%}')
#     print(f'Crackle: {s_crackle:.2%}')
#     print(f'Wheezle: {s_wheezle:.2%}')
#     print(f'Both: {s_both:.2%}')