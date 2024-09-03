import torch
import torch.nn as nn
import torchvision.models as models

# Define the residual block for TCN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding=dilation_rate)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding=dilation_rate)
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

# Define the overall model
class MultilevelTCNModel(nn.Module):
    def __init__(self, num_classes):
        super(MultilevelTCNModel, self).__init__()
        # Pre-trained VGG19 model for feature extraction
        self.vgg19 = models.vgg19(weights='VGG19_Weights.DEFAULT').features
        
        # Multilevel TCN blocks
        self.residual_blocks1 = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 2), ResidualBlock(512, 512, 4)])
        self.residual_blocks2 = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 3), ResidualBlock(512, 512, 9)])
        self.residual_blocks2 = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 4), ResidualBlock(512, 512, 16)])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 80)
        self.fc2 = nn.Linear(80, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Extract features using VGG19
        x = self.vgg19(x)

        # Reshape from (batch_size, channels, height, width) to (batch_size, channels, height * width)
        x = x.view(x.size(0), x.size(1), -1)

        # Apply TCN blocks (Assuming x is of shape [batch_size, channels, time_steps])
        x1 = x.clone()
        x2 = x.clone()
        
        for block in self.residual_blocks1:
            x1 = block(x1)
        
        for block in self.residual_blocks2:
            x2 = block(x2)

        # Concatenate the outputs from both TCN blocks
        x = torch.cat((x1, x2), dim=1)

        # Global average pooling and fully connected layers
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x
