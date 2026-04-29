import torch
import torch.nn as nn

class BrainTumorCNN(nn.Module):

    def __init__(self):
        super(BrainTumorCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 🔥 fixed to 3 channels
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),  # 🔥 adjusted for 128x128 input
            nn.ReLU(),
            nn.Linear(128, 4)  # 🔥 fixed to 4 classes
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x