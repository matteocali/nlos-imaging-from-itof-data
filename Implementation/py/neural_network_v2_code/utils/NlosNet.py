import torch
import torchvision
from torch import nn


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
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], pad) for i in range(len(chs) - 1)])
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
        self.upconv = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], kernel_size=2, stride=2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], pad) for i in range(len(chs) - 1)])

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


class Regressor(nn.Module):
    """Upsampler of the proposed network architecture"""

    def __init__(self) -> None:
        """
        Upsampler
        """

        super().__init__()
        features = 320 * 240
        self.dense = nn.Linear(features, features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        param:
            - x: input tensor
        return:
            - output tensor
        """
        
        x = self.dense(x)
        x = self.relu(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dense(x)
        return x


class FinalConv(nn.Module):
    """Final convolutional layers of the proposed network architecture"""

    def __init__(self, ch: int = 2, n_layers: int = 5, pad: int = 1) -> None:
        """
        Final convolutional layers
        param:
            - n_layer: number of layers
            - pad: padding
        """

        super().__init__()
        self.n_layers = n_layers
        self.conv = nn.ModuleList([nn.Conv2d(ch, ch, kernel_size=3, padding=pad) for _ in range(n_layers)])
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


class NlosNet(nn.Module):

    def __init__(self, enc_channels: tuple = (6, 64, 128, 256, 512, 1024), dec_channels: tuple = (1024, 512, 256, 128, 64), pad: int = 1, num_class: int = 1, retain_dim: bool = False, out_size: tuple = (512, 512)) -> None:
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
        self.encoder = Encoder(enc_channels, pad)
        self.decoder = Decoder(dec_channels, pad)
        self.head = nn.Conv2d(dec_channels[-1], num_class, kernel_size=1)
        self.retain_dim = retain_dim
        self.out_size = out_size
        #self.depth_estiamtor = FinalConv(ch=1, n_layers=2)
        #self.alpha_estiamtor = FinalConv(ch=1, n_layers=2)
        self.final_cnn = FinalConv(n_layers=5)
        #self.regressor = Regressor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        param:
            - x: input tensor
        return:
            - output tensor
        """

        # Run the encoder
        enc_features = self.encoder(x)
        # Run the decoder
        out = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        # Run the head to move back to the number of classes
        out = self.head(out)
        # Run the final two branches
        #depth = self.depth_estiamtor(out[:, 0, :, :].unsqueeze(1))
        #alpha = self.alpha_estiamtor(out[:, 1, :, :].unsqueeze(1))
        out = self.final_cnn(out)

        #out = torch.cat([depth, alpha], dim=1)
        # Run the regressor
        #out = self.regressor(out.flatten(1))
        #out = out.view(320, 240)
        if self.retain_dim:
            out = nn.functional.interpolate(out, size=self.out_size)
        #return out.squeeze(1)
        return out
