import torch
import torch.nn.functional as F
import numpy as np


def sobel(window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Function to create the sobel filter of custom size (3, 5, 7)
    Example of matrix for window_size=3:
                matx=[[-3, 0,+3],
                          [-10, 0 ,+10],
                          [-3, 0,+3]]
                maty=[[-3, -10,-3],
                          [0, 0 ,0],
                          [3, 10,3]]
    For the window_size=5 and window_size=7 it will be generated a weighted sobel mask,
                where the weight is 1/r where r is the distance from the center of the mask.

    param:
        - window_size: size of the sobel filter
    return:
        - sobel filter in the x direction
        - sobel filter in the y direction
    """

    assert window_size % 2 != 0  # Check that the window size is odd
    ind = int(window_size / 2)

    # Initialize the sobel filter
    matx = []
    maty = []

    # Create the sobel filter
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gx_ij = 0
            else:
                gx_ij = i / float(i * i + j * j)
            row.append(gx_ij)
        matx.append(row)
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gy_ij = 0
            else:
                gy_ij = j / float(i * i + j * j)
            row.append(gy_ij)
        maty.append(row)

    # Multiply the matrix by the correct value
    if window_size == 3:
        mult = 2
    elif window_size == 5:
        mult = 20
    elif window_size == 7:
        mult = 780
    else:
        raise ValueError("The window size must be 3, 5 or 7")

    matx = np.array(matx) * mult
    maty = np.array(maty) * mult

    return torch.Tensor(matx), torch.Tensor(maty)


def create_window(window_size: int, channel: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Function to create the window for the sobel filter.
    Essebtially it creates the sobel filter for each channel and then concatenate them.
    param:
            - window_size: size of the sobel filter
            - channel: number of channels
    return:
            - sobel filter in the x direction
            - sobel filter in the y direction
    """

    windowx, windowy = sobel(window_size)
    windowx, windowy = windowx.unsqueeze(0).unsqueeze(0), windowy.unsqueeze(
        0
    ).unsqueeze(0)
    windowx = torch.Tensor(windowx.expand(channel, 1, window_size, window_size))
    windowy = torch.Tensor(windowy.expand(channel, 1, window_size, window_size))

    return windowx, windowy


def gradient(
    img: torch.Tensor, windowx: torch.Tensor, windowy: torch.Tensor, channel: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Function to compute the gradient of an image using the soble filter of custom size (3, 5, 7)
    param:
            - img: image to compute the gradient
            - windowx: sobel filter in the x direction
            - windowy: sobel filter in the y direction
            - channel: number of channels
    return:
            - gradient in the x direction
            - gradient in the y direction
    """

    # Define the padding to be used
    padding = (windowx.shape[2] - 1) // 2

    # Perform the convolution on each channel separately and then concatenate
    if channel > 1:
        # Initialize the gradient
        gradx = torch.ones(img.shape)
        grady = torch.ones(img.shape)

        # Compute the gradient
        for i in range(channel):
            gradx[:, i, :, :] = F.conv2d(
                img[:, i, :, :].unsqueeze(0), windowx, padding=padding, groups=1
            ).squeeze(
                0
            )  # fix the padding according to the kernel size
            grady[:, i, :, :] = F.conv2d(
                img[:, i, :, :].unsqueeze(0), windowy, padding=padding, groups=1
            ).squeeze(0)
    else:
        gradx = F.conv2d(img, windowx, padding=padding, groups=1)
        grady = F.conv2d(img, windowy, padding=padding, groups=1)

    return gradx, grady


class SobelGrad(torch.nn.Module):
    """
    Class to compute the gradient of an image using the soble filter of custom size (3, 5, 7)
    """

    def __init__(self, window_size: int = 3) -> None:
        super(SobelGrad, self).__init__()
        self.window_size = window_size
        self.channel = 1  # Number of out channels
        self.windowx, self.windowy = create_window(window_size, self.channel)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract the number of channe of the input image
        (_, channel, _, _) = img.size()

        # Check if the input image is on the GPU
        if img.is_cuda:
            self.windowx = self.windowx.cuda(img.get_device())
            self.windowx = self.windowx.type_as(img)
            self.windowy = self.windowy.cuda(img.get_device())
            self.windowy = self.windowy.type_as(img)

        # Compute the gradient
        pred_gradx, pred_grady = gradient(img, self.windowx, self.windowy, channel)

        return pred_gradx, pred_grady
