import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """Matches the developerPratik/mnist-cnn-classifier checkpoint:
    4x (Conv+BN+ReLU) with 2 pooling stages, then 3x FC+BN+Dropout.
    28x28 -> pool -> 14x14 -> pool -> 7x7, 128*7*7 = 6272.
    """

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)                      # 28x28 -> 14x14

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)                      # 14x14 -> 7x7

        x = x.flatten(1)
        x = self.dropout(F.relu(self.bn5(self.fc1(x))))
        x = self.dropout(F.relu(self.bn6(self.fc2(x))))
        return self.fc3(x)