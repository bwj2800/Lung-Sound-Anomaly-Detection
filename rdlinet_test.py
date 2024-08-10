import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from model.rdlinet import RDLINet  # RDLINet 모델 임포트

# 데이터셋 경로
image_dir = 'data_4gr/mel_image'
model_save_path = './checkpoint/rdlinet_100_melonly.pth'

# 라벨 매핑
label_map = {'normal': 0, 'crackle': 1, 'wheeze': 2, 'both': 3}
label_names = list(label_map.keys())

# Custom Dataset class definition (기존 코드 사용)
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

def test():
    # Data preprocessing (기존 코드 사용)
    transform = transforms.Compose([
        transforms.Resize((64, 38)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 데이터셋 생성
    dataset = CustomDataset(image_dir=image_dir, transform=transform)
    # 데이터셋 분할
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 모델 불러오기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RDLINet(num_classes=len(label_map))
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()

    # 추론 및 성능 평가
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Test'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 혼동 행렬 계산
    avg_cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(avg_cm)

    # 클래스별 정확도 계산 - 아래 Accuracy랑 같음
    # class_accuracy = avg_cm.diagonal() / avg_cm.sum(axis=1)
    # for i, class_name in enumerate(label_names):
    #     print(f"Accuracy for {class_name}: {class_accuracy[i]*100:.2f}%")

    # 성능 지표 계산
    # 클래스별 성능 계산
    # Sensitivity (Se) = TP / (TP + FN)
    # Specificity (Sp) = TN / (FP + TN)
    s_crackle = avg_cm[1][1] / (avg_cm[1][0] + avg_cm[1][1] + avg_cm[1][2] + avg_cm[1][3])
    s_wheezle = avg_cm[2][2] / (avg_cm[2][0] + avg_cm[2][1] + avg_cm[2][2] + avg_cm[2][3])
    s_both = avg_cm[3][3] / (avg_cm[3][0] + avg_cm[3][1] + avg_cm[3][2] + avg_cm[3][3])

    S_e=(avg_cm[1][1]+avg_cm[2][2]+avg_cm[3][3] )/\
                        (avg_cm[1][0] + avg_cm[1][1] + avg_cm[1][2] + avg_cm[1][3]
                        +avg_cm[2][0] + avg_cm[2][1] + avg_cm[2][2] + avg_cm[2][3]
                        +avg_cm[3][0] + avg_cm[3][1] + avg_cm[3][2] + avg_cm[3][3])
    S_p=avg_cm[0][0]/(avg_cm[0][0]+avg_cm[0][1]+avg_cm[0][2]+avg_cm[0][3])
    S_c=(S_p+S_e)/2

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Crackle Accuracy: {s_crackle:.2%}')
    print(f'Wheeze Accuracy: {s_wheezle:.2%}')
    print(f'Both Accuracy: {s_both:.2%}')
    print("S_p: {}, S_e: {}, Score: {}".format(S_p, S_e, S_c))

    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    print(f"F1 Score (weighted): {f1:.4f}")


    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == '__main__':
    test()