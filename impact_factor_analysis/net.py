import torch
from torch import nn, optim


class Net01(nn.Module):
    def __init__(self):
        super(Net01, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(11, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8,1),
        )

    def forward(self, x):
        # input: NV 结构
        return self.layer(x)