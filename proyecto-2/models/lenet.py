import torch.nn as nn
import torch.nn.functional as functional


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 224 -> 220
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # 220 -> 110
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # 110 -> 106
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # 106 -> 53
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5) # 53 -> 49

        self.fc1 = nn.Linear(120*49*49, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(functional.tanh(self.conv1(x)))
        x = self.pool2(functional.tanh(self.conv2(x)))
        x = functional.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = functional.tanh(self.fc1(x))
        x = self.fc2(x)
        return x