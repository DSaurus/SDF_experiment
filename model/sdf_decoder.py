import torch
import torch.nn as nn

class SDFDecoder(nn.Module):
    def __init__(self):
        super(SDFDecoder, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv1d(512 + 3, 256, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv1d(256, 128, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv1d(128, 64, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv1d(64, 32, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv1d(32, 1, 1, 1, 0),
            nn.Sigmoid()
        )

        self.pos_stage = nn.Sequential(
            nn.Conv1d(512 + 3 * 128, 512, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv1d(512, 256, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv1d(256, 128, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv1d(128, 64, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv1d(64, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x, pos_encoding=False):
        if pos_encoding:
            return self.pos_stage(x)
        return self.stage1(x)