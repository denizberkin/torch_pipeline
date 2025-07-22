import torch 
import torch.nn as nn

from models.segmentation.unet_modules.block import UpSampleBlock, DownSampleBlock, ConvBlock


class Encoder(nn.Module):
    def __init__(self, in_channels: int, blocks: list = [64, 128, 256, 512]):
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        blocks = [in_channels] + blocks
        self.encoder_blocks = nn.ModuleList([
            DownSampleBlock(blocks[i], blocks[i + 1]) for i in range(len(blocks) - 1)
        ])

    def forward(self, x):
        features = []
        for block in self.encoder_blocks:
            x, residual = block(x)
            features.append(residual)
        return x, features.reverse()
    

class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, blocks: list = [1024, 512, 256, 128, 64]):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([
            UpSampleBlock(blocks[i], blocks[i + 1]) for i in range(len(blocks) - 1)
        ])

    def forward(self, x: torch.Tensor, features: list) -> torch.Tensor:
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, features[i])
        return x


class UNetModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.bottleneck = Bottleneck(512, 1024)
        self.decoder = Decoder()
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x, enc_features = self.encoder(x)
        bottleneck_output = self.bottleneck(x)
        dec_output = self.decoder(bottleneck_output, enc_features)
        return self.out(dec_output)