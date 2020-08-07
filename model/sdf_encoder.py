import torch
import torch.nn as nn

class SDFEncoder(nn.Module):
    def __init__(self):
        super(SDFEncoder, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv3d(1, 16, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(), 
            nn.Conv3d(16, 32, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 64, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 128, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 256, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.Conv3d(256, 512, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
        )

    def forward(self, x):
        return self.stage1(x)