import torch
import torch.nn as nn
import torch.nn.functional as F

# Heightmap Denoising Network (HDN)
class HeightmapDenoiser(nn.Module):
    def __init__(self, in_channels=1, num_layers=10, num_feat=64):
        super(HeightmapDenoiser, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, num_feat, kernel_size=7,padding=3))
        layers.append(nn.BatchNorm2d(num_feat))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_feat, num_feat, kernel_size=7,padding=3))
            layers.append(nn.BatchNorm2d(num_feat))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.BatchNorm2d(num_feat))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_feat, in_channels, kernel_size=7,
                                padding=3))
        self.net = nn.Sequential(*layers)


    def forward(self, h):
        # h: (B, 1, k, k) noisy heightmap
        return self.net(h)  # (B, 1, k, k) denoised heightmap