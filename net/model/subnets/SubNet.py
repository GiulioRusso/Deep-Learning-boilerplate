import torch
from torch import nn, Tensor


class SubNet(nn.Module):
    """
    A minimal example submodule block with Conv1d + ReLU.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param kernel_size: Convolution kernel size.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        :param x: Input tensor of shape (B, C_in, T).
        :return: Output tensor of shape (B, C_out, T).
        """
        return self.relu(self.conv(x))
