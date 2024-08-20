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
        self.vgg19 = models.vgg19(pretrained=True).features

        # TCN blocks for each input
        self.residual_blocks_mel = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 2), ResidualBlock(512, 512, 4)])
        self.residual_blocks_chroma = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 2), ResidualBlock(512, 512, 4)])
        self.residual_blocks_mfcc = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 2), ResidualBlock(512, 512, 4)])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 3, 128)  # Combine the output from all three TCN blocks
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, chroma, mfcc, mel):
        # Extract features using VGG19
        chroma_features = self.vgg19(chroma)
        mfcc_features = self.vgg19(mfcc)
        mel_features = self.vgg19(mel)

        # Reshape to (batch_size, channels, height * width)
        chroma_features = chroma_features.view(chroma_features.size(0), chroma_features.size(1), -1)
        mfcc_features = mfcc_features.view(mfcc_features.size(0), mfcc_features.size(1), -1)
        mel_features = mel_features.view(mel_features.size(0), mel_features.size(1), -1)

        # Apply TCN blocks separately for each feature
        for block in self.residual_blocks_mel:
            mel_features = block(mel_features)
        for block in self.residual_blocks_chroma:
            chroma_features = block(chroma_features)
        for block in self.residual_blocks_mfcc:
            mfcc_features = block(mfcc_features)

        # Global average pooling
        mel_out = self.global_avg_pool(mel_features).squeeze(-1)
        chroma_out = self.global_avg_pool(chroma_features).squeeze(-1)
        mfcc_out = self.global_avg_pool(mfcc_features).squeeze(-1)

        # Concatenate the outputs from all three TCN blocks
        combined_out = torch.cat((mel_out, chroma_out, mfcc_out), dim=1)
        combined_out = self.flatten(combined_out)

        # Fully connected layers
        combined_out = self.fc1(combined_out)
        combined_out = self.fc2(combined_out)

        return combined_out
