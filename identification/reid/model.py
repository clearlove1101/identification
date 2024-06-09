import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * stride, 3, stride, 1, bias=False),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channel * stride, out_channel, 1, 1, 0, bias=False),
            nn.LeakyReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        )

        self.shortcut = nn.Identity() \
            if stride == 1 and in_channel == out_channel else \
            nn.Conv2d(in_channel, out_channel, 3, stride, 1)

        self.norm = nn.Sequential(
            nn.LeakyReLU(True),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        out = self.norm(out)
        return out

