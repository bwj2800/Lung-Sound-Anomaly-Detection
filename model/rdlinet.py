import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MFLIBlock(nn.Module):
    def __init__(self, in_channels):
        super(MFLIBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.conv2 = DepthwiseSeparableConv(in_channels, 48, kernel_size=3, padding=1)
        self.conv3 = DepthwiseSeparableConv(in_channels, 48, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        x1 = self.leaky_relu(self.conv1(x))
        x2 = self.leaky_relu(self.conv2(x))
        x3 = self.leaky_relu(self.conv3(x))
        x4 = self.leaky_relu(self.conv4(self.maxpool(x)))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out

class GLUClassifierModule(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GLUClassifierModule, self).__init__()
        self.fc1 = nn.Linear(in_channels, 15)
        self.fc2 = nn.Linear(in_channels, 15)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(15, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.sigmoid(self.fc2(x))
        x = x1 * x2
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class RDLINet(nn.Module):
    def __init__(self, num_classes=4):
        super(RDLINet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.depthwise_conv = DepthwiseSeparableConv(16, 32, kernel_size=3, padding=1)
        self.pointwise_conv = nn.Conv2d(16, 32, kernel_size=1)
        self.mfli1 = MFLIBlock(32)
        self.mfli2 = MFLIBlock(64)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.glu_classifier = GLUClassifierModule(64, num_classes)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.leaky_relu(self.depthwise_conv(x))
        x = self.leaky_relu(self.pointwise_conv(x))
        x = self.maxpool(x)
        x = self.mfli1(x)
        x = self.mfli2(x)
        x = self.maxpool(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.glu_classifier(x)
        return x

if __name__ == "__main__":
    # 모델 초기화 및 테스트
    model = RDLINet(num_classes=4)
    print(model)

    # 임의의 입력 데이터로 테스트
    input_data = torch.randn(1, 3, 64, 38)
    output = model(input_data)
    print(output.shape)
