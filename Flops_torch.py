# python Flops_torch.py --model_kind RDLINet --model_path checkpoint\rdlinet.pth --image_path data_4gr\mel_image_old\both\image_1015.jpg --input_size 3 64 64
# python Flops_torch.py --model_kind RDLINet --model_path checkpoint\rdlinet.pth --image_path data_4gr\mel_image_old 3 64 64
import torch
import time
import argparse
from torchvision import transforms
from PIL import Image
from thop import profile
from model.rdlinet import RDLINet  # Import the RDLINet model
from torchsummary import summary
import random
import numpy as np
from RDLINet.rdlinet_main import CustomDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from tqdm import tqdm

label_map = {'normal': 0, 'crackle': 1, 'wheeze': 2, 'both': 3}
# 데이터셋 경로
image_dir = './Dataset_ICBHI_Log-Melspec/Dataset_Task_1/Dataset_1_2'
# image_dir = './data_4gr/mel_image_old'

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

def load_image(image_path, input_size):
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지가 RGB라면 정규화
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    return image

def print_model_statistics(model_kind, model_path, image_path, input_size):
    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_kind == 'RDLINet':
        model = torch.load(model_path)
        model = RDLINet(num_classes=len(label_map))
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    
    # Trainable parameters 계산
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # MACs와 FLOPs 계산
    input_tensor = torch.randn(1, *input_size).to('cuda')  # 입력 텐서 생성
    macs, flops = profile(model, inputs=(input_tensor,))
    macs, flops = macs / 1e6, flops / 1e6  # 단위를 Giga로 변환
    print(f"MACs: {macs:.3f} M")
    print(f"FLOPs: {flops:.3f} M")

    # 메모리 크기 계산
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    model_memory_size = (param_size + buffer_size) / 1024**2
    print(f"Model Size: {model_memory_size:.3f} MB")

    # Inference 시간 측정
    
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        # transforms.Normalize((0.3416, 0.1199, 0.3481]), (0.2769, 0.1272, 0.1512))
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    seed = 42  # 원하는 시드 값으로 설정
    set_seed(seed)

    # Create the dataset
    dataset = CustomDataset(image_dir=image_dir, transform=transform)
    print("Dataset ready")

    # 이미지 경로와 라벨 리스트를 가져옵니다.
    image_paths = dataset.image_paths
    labels = dataset.labels

    train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.4, random_state=seed, stratify=labels)
    train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.1, random_state=seed, stratify=train_labels)
    test_dataset = CustomDataset(image_dir=image_dir, transform=transform)
    # 필요한 경우 각 데이터셋의 이미지 경로와 라벨을 설정해줍니다.
    test_dataset.image_paths, test_dataset.labels = test_paths, test_labels
    # 데이터 로더 생성
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    model.eval()

    # 추론 및 성능 평가
    all_preds = []
    all_labels = []

    with torch.no_grad():
        start_time = time.time()
        for images, labels in tqdm(test_loader, desc=f'Test'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # 밀리초로 변환


    # inference_time, all_labels, all_preds = measure_inference_time(model, image_tensor)
    print(f"Inference Time: {inference_time:.2f} ms")
    print(summary(model, (3, 64, 64)))

    # 혼동 행렬 계산
    avg_cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(avg_cm)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Statistics and Inference Time")
    parser.add_argument("--model_kind", type=str, required=True, help="Model kind(TCN, RDLINet, etc...)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--input_size", type=int, nargs='+', default=[3, 64, 64], help="Input size of the model (C, H, W)")

    args = parser.parse_args()
    
    # 모델 통계 및 추론 시간 출력
    print_model_statistics(model_kind=args.model_kind, model_path=args.model_path, image_path=args.image_path, input_size=tuple(args.input_size))
