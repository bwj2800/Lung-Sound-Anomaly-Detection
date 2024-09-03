import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ResidualBlock, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        # self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=dilation, dilation=dilation)
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=(1,1))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

class TCNBranch(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, dilation_base):
        super(TCNBranch, self).__init__()
        self.conv = nn.Conv2d(3, 80, stride=1, kernel_size=3, padding=1)
        layers = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            layers.append(ResidualBlock(in_channels, out_channels, dilation))
            in_channels = out_channels
        self.tcn_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x=self.conv(x)
        return self.tcn_layers(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_BiGRU(nn.Module):
    def __init__(self, input_channels, num_branches, num_layers_per_branch, dilation_base, hidden_dim1, hidden_dim2, num_classes):
        super(CNN_BiGRU, self).__init__()
        self.branches = nn.ModuleList([])
        for i in range(num_branches):
            self.branches.append(TCNBranch(input_channels, 80, num_layers_per_branch, dilation_base[i]))
        # self.branches = nn.ModuleList([
        #     TCNBranch(input_channels, 80, num_layers_per_branch, dilation_base)
        #     for _ in range(num_branches)
        # ])
        # self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Define the MLP classifier with 3 layers
        self.classifier = MLPClassifier(num_branches * 80, hidden_dim1, hidden_dim2, num_classes)

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        # print("len(branch_outputs):",len(branch_outputs), " branch_outputs[0].shape:", branch_outputs[0].shape)
        
        # Concatenate along the sequence length dimension
        x = torch.cat(branch_outputs, dim=1)
        # print("x.shape:", x.shape) # x.shape: torch.Size([1, 240, 64, 64])
        x = self.global_avg_pool(x).squeeze(-1).squeeze(-1) # x.shape: torch.Size([1, 240])
        # print("x.shape:", x.shape)
        out = self.classifier(x) 
        return out # out.shape: torch.Size([1, 4])
    
# if __name__ == "__main__":
#     model = CNN_BiGRU(input_channels=80, num_branches=3, num_layers_per_branch=3, dilation_base=[2,3,4], hidden_dim1=80, hidden_dim2=32, num_classes=4)
#     x = torch.randn((1, 3, 64, 64))
#     y = model(x)
    
#     print(y.shape)