import torch
import torch.nn as nn

class ECG_CNN(nn.Module):
    def __init__(self, num_classes=1, num_leads=8):
        super(ECG_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_leads, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        # Camada de pooling adaptativa para lidar com diferentes comprimentos de entrada
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x