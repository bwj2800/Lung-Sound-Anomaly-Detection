import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
import numpy as np
import random
import time
from thop import profile
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath('./'))
from model.cnn_lstm import CNN_LSTM

# 데이터셋 경로
# image_dir = './data_4gr/mel_image_cnn_lstm_2class'
# model_save_path = './checkpoint/cnn_lstm_2class.pth'
image_dir = './data_4gr/0822/Task2_1'
model_save_path = './checkpoint/cnn_lstm_Task2_1.pth'

# 라벨 매핑
# label_map = {'normal': 0, 'crackle': 1, 'wheeze': 1, 'both': 1}
# label_map = {'normal': 0, 'abnormal': 1}
label_map = {'Healthy': 0, 'Unhealthy': 1}

seed = 42  # 원하는 시드 값으로 설정
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)
random.seed(seed)
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

# Data preprocessing
transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create the dataset
dataset = CustomDataset(image_dir=image_dir, transform=transform)
print("Dataset ready")

# # 데이터셋 분할
# train_size = int(0.6 * len(dataset))
# val_size = int(0.2 * len(dataset))
# test_size = len(dataset) - train_size - val_size
# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

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

# 데이터셋 분할 후, 각 데이터셋의 크기 출력
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# 모델 초기화
num_class = 2
model = CNN_LSTM(num_class=num_class)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.load_state_dict(torch.load(model_save_path, weights_only=True))
model.eval()
correct = 0
total = 0
avg_cm = [[0, 0], [0, 0]]


# Count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f'Trainable Parameters: {num_params / 1e6:.2f} M')

# Calculate MACs and FLOPs
input = torch.randn(1, 3, 128, 128).to(device)
macs, params = profile(model, inputs=(input,))
print(f'MACs per single image: {macs / 1e9:.2f} G')
print(f'FLOPs per single image: {macs * 2 / 1e9:.2f} G')

# Estimate memory usage
def get_model_memory(model, input_size):
    input = torch.randn(1, *input_size).to(device)
    model = model.to(device)
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]  # Take the first element if the output is a tuple
        module.memory = output.nelement() * output.element_size()
    
    hooks = []
    for layer in model.modules():
        if len(list(layer.children())) == 0:
            hooks.append(layer.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        _ = model(input)
    
    total_memory = sum([layer.memory for layer in model.modules() if hasattr(layer, 'memory')])
    
    for hook in hooks:
        hook.remove()

    return total_memory

input_size = (3, 128, 128)
memory_usage = get_model_memory(model, input_size)
print(f'Model Memory Usage: {memory_usage / 1024**3:.2f} G  ({memory_usage / 1024**2:.2f} MB)')


with torch.no_grad():
    start_time = time.time()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 혼동 행렬 계산
        for i in range(num_class):
            avg_cm[labels[i]][predicted[i]] += 1

    end_time = time.time()
    avg_time = (end_time - start_time) / len(test_loader)
    print(f'Total Inference Time: {(end_time-start_time)* 1000:.2f} ms   Average Inference Time per Image: {avg_time*1000:.2f} ms')

    print(f'Accuracy on test set: {100 * correct / total}%')

    # 클래스별 성능 계산    
    S_e=avg_cm[1][1]/(avg_cm[1][0]+avg_cm[1][1])
    S_p=avg_cm[0][0]/(avg_cm[0][0]+avg_cm[0][1])
    S_c=(S_p+S_e)/2
    print("S_p: {}, S_e: {}, Score: {}".format(S_p, S_e, S_c))
