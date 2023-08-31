import torch
import torchvision
from torch import nn
from math import sqrt
from ..utils import itof2depth
from .StraightThroughEstimators import StraightThroughEstimator


class Block(nn.Module):
    """Block of the proposed network architecture"""

    def __init__(self, in_ch: int, out_ch: int, pad: int) -> None:
        """
        Single block of the proposed network architecture
        param:
            - in_ch: number of input channels
            - out_ch: number of output channels
            - pad: padding
        """

        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=pad)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=pad)

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

    def __init__(self, chs: tuple = (3, 64, 128, 256, 512, 1024), pad: int = 0) -> None:
        """
        Encoder
        param:
            - chs: number of channels for each block
            - pad: padding for the block
        """

        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1], pad) for i in range(len(chs) - 1)]
        )
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

    def __init__(self, chs: tuple = (1024, 512, 256, 128, 64), pad: int = 0) -> None:
        """
        Decoder
        param:
            - chs: number of channels for each block
            - pad: padding for the block
        """

        super().__init__()
        self.chs = chs
        self.upconv = nn.ModuleList(
            [
                nn.ConvTranspose2d(chs[i], chs[i + 1], kernel_size=2, stride=2)
                for i in range(len(chs) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1], pad) for i in range(len(chs) - 1)]
        )

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

    def crop(self, x: torch.Tensor, enc_ftrs: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class FinalConv(nn.Module):
    """Final convolutional layers of the proposed network architecture"""

    def __init__(
        self, chs: tuple = (8, 4, 2), additional_layers: int = 0, pad: int = 1
    ) -> None:
        """
        Final convolutional layers
        param:
            - n_layer: number of layers
            - additional_layers: number of additional CNN layers
            - pad: padding
        """

        super().__init__()
        self.n_layers = len(chs) + additional_layers - 1
        channels = list(chs)
        for _ in range(additional_layers):
            channels = [chs[0]] + channels
        self.conv = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=pad)
                for i in range(self.n_layers)
            ]
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        param:
            - x: input tensor
        return:
            - output tensor
        """

        for i in range(self.n_layers):
            x = self.conv[i](x)
            if i != self.n_layers - 1:
                x = self.relu(x)
        return x


class NlosNetItof(nn.Module):
    """
    NLOS Net\n
    Proposet Neural Network for NLOS imaging implemented in PyTorch
    """

    def __init__(
        self,
        enc_channels: tuple = (6, 64, 128, 256, 512, 1024),
        dec_channels: tuple = (1024, 512, 256, 128, 64),
        pad: int = 1,
        num_class: int = 1,
        additional_cnn_layers: int = 0,
    ) -> None:
        """
        NLOS Net
        param:
            - enc_channels: number of channels for each block of the encoder
            - dec_channels: number of channels for each block of the decoder
            - pad: padding for the blocks
            - num_class: number of classes for the last UNet layer (power of 2)
            - additional_cnn_layers: number of additional CNN layers
        """

        super().__init__()
        self.encoder = Encoder(enc_channels, pad)  # Initialize the encoder
        self.decoder = Decoder(dec_channels, pad)  # Initialize the decoder
        self.head = nn.Conv2d(
            dec_channels[-1], num_class, kernel_size=1
        )  # Initialize the head (last layer of the UNet reduce the features layer to the one set by num_class)

        # Final layers
        chs = [num_class]
        n_final_layers = (
            round(sqrt(num_class)) - 1
        )  # Number of layers for the final layers
        for i in range(
            n_final_layers
        ):  # Initialize the number of channels for the final layers
            chs.append(int(round(num_class / (2 ** (i + 1)))))
        self.itof_estiamtor = FinalConv(
            chs=tuple(chs), additional_layers=additional_cnn_layers
        )  # Initialize the itof data estimator

        # Initialize the straight through estimators
        self.st_clean = StraightThroughEstimator(task="clean")
        self.st_hard = StraightThroughEstimator(task="threshold", threshold=0)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Forward pass
        param:
            - x: input tensor
        return:
            - output tensor
        """

        # Run the encoder
        enc_features = self.encoder(x)
        # Run the decoder
        out = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        # Run the head to move back to the number of classes
        out = self.head(out)
        # Run the final two branches
        itof = self.itof_estiamtor(out)

        # Compute the related depthmap and return a hard mask
        clean_itof = self.st_clean(itof)
        depth = itof2depth(clean_itof, 20e06)
        mask = self.st_hard(clean_itof)

        return itof, depth, mask
