import torch
import torchvision
from torch import nn


class Block(nn.Module):
    """Block of the proposed network architecture"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        param:
            - x: input tensor
        return:
            - output tensor
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    """Encoder of the proposed network architecture"""

    def __init__(self, chs: tuple = (3, 64, 128, 256, 512, 1024)) -> None:
        """
        Encoder
        param:
            - chs: number of channels for each block
        """
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass
        param:
            - x: input tensor
        return:
            - output list of encoded features
        """
        enc_features = []
        for block in self.enc_blocks:
            x = block(x)
            enc_features.append(x)
            x = self.pool(x)
        return enc_features


class Decoder(nn.Module):
    """Decoder of the proposed network architecture"""
    
    def __init__(self, chs: tuple = (1024, 512, 256, 128, 64)) -> None:
        """
        Decoder
        param:
            - chs: number of channels for each block
        """
        super().__init__()
        self.chs = chs
        self.upconv = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], kernel_size=2, stride=2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x: torch.Tensor, enc_features: list) -> torch.Tensor:
        """
        Forward pass
        param:
            - x: input tensor
            - enc_features: list of tensors
        return:
            - output tensor
        """
        for i in range(len(self.chs) - 1):
            x = self.upconv[i](x)
            enc_ftrs = self.crop(x, enc_features[i])
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, x: torch.Tensor, enc_ftrs: list) -> torch.Tensor:
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs  # type: ignore


class NlosNet(nn.Module):

    def __init__(self, enc_channels: tuple = (6, 64, 128, 256, 512, 1024), dec_channels: tuple = (1024, 512, 256, 128, 64), num_class: int = 1, retain_dim: bool = False, out_size: tuple = (512, 512)) -> None:
        """
        NLOS Net
        param:
            - enc_channels: number of channels for each block of the encoder
            - dec_channels: number of channels for each block of the decoder
            - num_class: number of classes
            - retain_dim: if True the output tensor will have the same dimension of the input tensor
            - out_size: output size of the network
        """
        super().__init__()
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        self.head = nn.Conv2d(dec_channels[-1], num_class, kernel_size=1)
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        param:
            - x: input tensor
        return:
            - output tensor
        """
        enc_features = self.encoder(x)
        out = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = nn.functional.interpolate(out, size=self.out_size)
        return out
