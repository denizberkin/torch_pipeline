from typing import Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, f: int=3, padding: int=1):
        """
        ### Params: f, padding -> (3, 1) defaults from UNet paper
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=f, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=f, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self.conv(x)
        x = self.pool(residual)
        return x, residual
    

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(2 * out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        # residual shape integrity
        dh = skip.size()[2] - x.size()[2]
        dw = skip.size()[3] - x.size()[3]

        x = nn.functional.pad(x, [dw // 2, dw - dw // 2,
                                  dh // 2, dh - dh // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
