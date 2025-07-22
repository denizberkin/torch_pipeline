import torch
import torch.nn as nn

from models.classification.fcn import OuterLinearLayer
from models.segmentation.unet_modules.network import UNetModule
from models.base import BaseModel


class UNet(BaseModel):
    def __init__(self, **kwargs):
        """
        UNet implementation, paper: https://arxiv.org/pdf/1505.04597
        ### Arguments:
            - kwargs: dict containing `in_channels` and `out_channels`
                - in_channels: number of input channels; RGB, grayscale or else (default: 3)
                - out_channels: number of output channels; number of output classes (default: 1)
        """
        super().__init__()
        defaults = {"in_channels": kwargs.get("in_channels", 3), "out_channels": kwargs.get("out_channels", 1)}
        self.unet = UNetModule(**defaults)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unet(x)
        return x

    def get_alias(self): return "unet"



if __name__ == "__main__":
    device = torch.device("cuda")
    model = UNet(in_channels=3, out_channels=1)
    model.to(device)
    input_tensor = torch.randn(1, 3, 572, 572).to(device)
    out = model(input_tensor)

    print(f"input shape: {input_tensor.shape}")
    print(f"output shape: {out.shape}")
