import torch
from torch import nn


class NlosNet(nn.Module):

    def __init__(self) -> None:
        """
        NLOS Net
        param:
            - n_freq: number of frequencies used by the iToF sensor
        """
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        param:
            - x: input tensor
        return:
            - output tensor
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):
    """Encoder of the proposed network architecture"""

    def __init__(self) -> None:
        """
        Encoder
        """
        super().__init__()


class Decoder(nn.Module):
    """Decoder of the proposed network architecture"""
    
    def __init__(self) -> None:
        """
        Decoder
        """
        super().__init__()