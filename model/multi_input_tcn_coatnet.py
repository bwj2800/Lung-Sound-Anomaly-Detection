import torch
import torch.nn as nn
import torchvision.models as models
import timm

# Define the residual block for TCN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding=dilation_rate)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding=dilation_rate)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        #print('x shape: ', x.shape)
        #batch_size, channels, seq_length = x.shape
        #height = width = int(seq_length ** 0.5)  # Assuming the original height and width were equal (7x7)
        #x = x.view(batch_size, channels, height, width)  # Reshape to (batch_size, channels, height, width)

        #print('x shape: ', x.shape)
        res = self.conv1(x)
        #print('conv1: ', res.shape)
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
        coatnet = timm.create_model('coatnet_rmlp_2_rw_224', pretrained=True)
        self.coatnet_features = nn.Sequential(*list(coatnet.children())[:-2])  # Remove the last FC layer and pooling
    

        
        self.conv1x1 = nn.Conv2d(1024, 512, kernel_size=1)  # 1x1 convolution to reduce channels from 2048 to 512
        #coatnet_out_channels = coatnet.num_features  # Get the output channels from CoAtNet
        #print('channel: ',coatnet_out_channels)

        # TCN blocks for each input
        self.residual_blocks_mel = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 2), ResidualBlock(512, 512, 4)])
        self.residual_blocks_chroma = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 2), ResidualBlock(512, 512, 4)])
        self.residual_blocks_mfcc = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 2), ResidualBlock(512, 512, 4)])
        
        self.residual_blocks_mel2 = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 3), ResidualBlock(512, 512, 9)])
        self.residual_blocks_chroma2 = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 3), ResidualBlock(512, 512, 9)])
        self.residual_blocks_mfcc2 = nn.ModuleList([ResidualBlock(512, 512, 1), ResidualBlock(512, 512, 3), ResidualBlock(512, 512, 9)])

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 3, 256)  # Combine the output from all three TCN blocks
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, chroma, mfcc, mel):
        # Extract features using VGG19
        chroma_features = self.coatnet_features(chroma)
        mfcc_features = self.coatnet_features(mfcc)
        mel_features = self.coatnet_features(mel)

        # 1x1 convolution으로 채널 크기 줄이기
        chroma_features = self.conv1x1(chroma_features)
        mfcc_features = self.conv1x1(mfcc_features)
        mel_features = self.conv1x1(mel_features)
        #print('chroma shape: ', chroma_features.shape)
        #batch_size, channels, height_width = chroma_features.shape
        #height = width = int(height_width ** 0.5)  # Assuming it's square (e.g., 49 = 7x7)

        #chroma_features = chroma_features.view(batch_size, channels, height, width)
        #mfcc_features = mfcc_features.view(batch_size, channels, height, width)
        #mel_features = mel_features.view(batch_size, channels, height, width)
        # Reshape to (batch_size, channels, height * width)
        #chroma_features = chroma_features.view(chroma_features.size(0), chroma_features.size(1), -1)
        #mfcc_features = mfcc_features.view(mfcc_features.size(0), mfcc_features.size(1), -1)
        #mel_features = mel_features.view(mel_features.size(0), mel_features.size(1), -1)

        mel_features1 = mel_features.clone()
        mel_features2 = mel_features.clone()
        mfcc_features1 = mfcc_features.clone()
        mfcc_features2 = mfcc_features.clone()
        chroma_features1 = chroma_features.clone()
        chroma_features2 = chroma_features.clone()

        # Apply TCN blocks separately for each feature
        for block in self.residual_blocks_mel:
            mel_features1 = block(mel_features1)
        for block in self.residual_blocks_chroma:
            chroma_features1 = block(chroma_features1)
        for block in self.residual_blocks_mfcc:
            mfcc_features1 = block(mfcc_features1)

        for block in self.residual_blocks_mel2:
            mel_features2 = block(mel_features2)
        for block in self.residual_blocks_chroma2:
            chroma_features2 = block(chroma_features2)
        for block in self.residual_blocks_mfcc2:
            mfcc_features2 = block(mfcc_features2)

        # Concatenate the outputs from both TCN blocks
        mel_features = torch.cat((mel_features1, mel_features2), dim=2)
        mfcc_features = torch.cat((mfcc_features1, mfcc_features2), dim=2)
        chroma_features = torch.cat((chroma_features1, chroma_features2), dim=2)

        # Global average pooling
        mel_out = self.global_avg_pool(mel_features).squeeze(-1)
        chroma_out = self.global_avg_pool(chroma_features).squeeze(-1)
        mfcc_out = self.global_avg_pool(mfcc_features).squeeze(-1)

        # Concatenate the outputs from all three TCN blocks
        combined_out = torch.cat((mel_out, chroma_out, mfcc_out), dim=1)
        combined_out = self.flatten(combined_out)

        # Fully connected layers
        x = self.fc1(combined_out)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x