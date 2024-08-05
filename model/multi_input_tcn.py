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
        self.vgg19_chroma = models.vgg19(weights='VGG19_Weights.DEFAULT').features
        self.vgg19_mfcc = models.vgg19(weights='VGG19_Weights.DEFAULT').features
        self.vgg19_mel = models.vgg19(weights='VGG19_Weights.DEFAULT').features

        # Multilevel TCN blocks
        self.residual_blocks_chroma1 = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 2), ResidualBlock(512, 512, 3)])
        self.residual_blocks_chroma2 = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 3), ResidualBlock(512, 512, 9)])
        
        self.residual_blocks_mfcc1 = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 2), ResidualBlock(512, 512, 3)])
        self.residual_blocks_mfcc2 = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 3), ResidualBlock(512, 512, 9)])
        
        self.residual_blocks_mel1 = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 2), ResidualBlock(512, 512, 3)])
        self.residual_blocks_mel2 = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 3), ResidualBlock(512, 512, 9)])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 6, 128)  # 512 channels * 6 TCN blocks * 3 inputs
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, chroma, mfcc, mel):
        # Extract features using VGG19
        chroma_features = self.vgg19_chroma(chroma)
        mfcc_features = self.vgg19_mfcc(mfcc)
        mel_features = self.vgg19_mel(mel)

        # Reshape from (batch_size, channels, height, width) to (batch_size, channels, height * width)
        chroma_features = chroma_features.view(chroma_features.size(0), chroma_features.size(1), -1)
        mfcc_features = mfcc_features.view(mfcc_features.size(0), mfcc_features.size(1), -1)
        mel_features = mel_features.view(mel_features.size(0), mel_features.size(1), -1)

        # Apply TCN blocks to chroma features
        chroma1 = chroma_features.clone()
        chroma2 = chroma_features.clone()
        for block in self.residual_blocks_chroma1:
            chroma1 = block(chroma1)
        for block in self.residual_blocks_chroma2:
            chroma2 = block(chroma2)
        chroma_out = torch.cat((chroma1, chroma2), dim=1)

        # Apply TCN blocks to mfcc features
        mfcc1 = mfcc_features.clone()
        mfcc2 = mfcc_features.clone()
        for block in self.residual_blocks_mfcc1:
            mfcc1 = block(mfcc1)
        for block in self.residual_blocks_mfcc2:
            mfcc2 = block(mfcc2)
        mfcc_out = torch.cat((mfcc1, mfcc2), dim=1)

        # Apply TCN blocks to mel features
        mel1 = mel_features.clone()
        mel2 = mel_features.clone()
        for block in self.residual_blocks_mel1:
            mel1 = block(mel1)
        for block in self.residual_blocks_mel2:
            mel2 = block(mel2)
        mel_out = torch.cat((mel1, mel2), dim=1)

        # Concatenate the outputs from all TCN blocks
        combined_out = torch.cat((chroma_out, mfcc_out, mel_out), dim=1)

        # Global average pooling and fully connected layers
        combined_out = self.global_avg_pool(combined_out).squeeze(-1)
        combined_out = self.flatten(combined_out)
        combined_out = self.fc1(combined_out)
        combined_out = self.fc2(combined_out)

        return combined_out
