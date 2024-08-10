import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        
        # First CNN Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second CNN Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third CNN Layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(96)
        
        # Max Pooling Layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=29, hidden_size=64, num_layers=1, batch_first=True)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64*288, 128)
        self.fc2 = nn.Linear(128, 4)
    
    def forward(self, x):
        # First CNN Layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool(x)
        
        # Second CNN Layer
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool(x)
        
        # Third CNN Layer
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.pool(x)
        # Reshape for LSTM
        x = x.reshape(x.size(0), 96 * 3, 29)
        
        # LSTM Layer
        x, _ = self.lstm(x)
        
        # Fully Connected Layers
        x=torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)