import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding1D(nn.Module):
    def __init__(self, sequence_length: int, encoding_dim: int) -> None:
        """
        Initializes a 1D positional encoding layer.

        Args:
        - sequence_length (int): The length of the input sequence.
        - encoding_dim (int): The dimension of the encoding vector.

        Returns:
        - None
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.pos_encoding = self.generate_positional_encoding()

    def generate_positional_encoding(self):
        """
        Generates the positional encoding matrix.

        Args:
        - None

        Returns:
        - pos_encoding (Tensor): The positional encoding matrix with shape (1, sequence_length, encoding_dim).
        """
        # Function to compute the angle for positional encoding
        def angle(pos, i):
            return pos / np.power(10000, (2 * (i // 2)) / self.encoding_dim)

        # Create the positional encoding matrix using the angle function
        pos_encoding = np.fromfunction(angle, (self.sequence_length, self.encoding_dim), dtype=np.float32)
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])

        # Convert the positional encoding matrix to a PyTorch tensor and add a batch dimension
        return torch.tensor(pos_encoding, dtype=torch.float32).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
        - x (Tensor): The input tensor with shape (batch_size, sequence_length, input_dim).

        Returns:
        - x (Tensor): The input tensor with positional encoding added, with shape (batch_size, sequence_length, input_dim).
        """
        # Add the positional encoding to the input tensor and return the result
        return x + self.pos_encoding.to(x.device)


class PositionalEncoding2D(nn.Module):
    def __init__(self, height: int, width: int, channels: int) -> None:
        """
        Initializes a 2D positional encoding layer.

        Args:
        - height (int): The height of the input image.
        - width (int): The width of the input image.
        - channels (int): The number of channels in the input image.

        Returns:
        - None
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.pos_encoding = self.generate_positional_encoding()

    def generate_positional_encoding(self):
        """
        Generates the positional encoding matrix.

        Args:
        - None

        Returns:
        - pos_encoding (Tensor): The positional encoding matrix with shape (1, height, width, channels).
        """
        # Function to compute the angle for positional encoding
        def angle(h, w, c):
            return (h + w) / np.power(10000, (2 * (c // 2)) / self.channels)

        # Create the positional encoding matrix using the angle function
        pos_encoding = np.fromfunction(angle, (self.height, self.width, self.channels), dtype=np.float32)
        pos_encoding[0::2] = np.sin(pos_encoding[0::2])
        pos_encoding[1::2] = np.cos(pos_encoding[1::2])

        # Convert the positional encoding matrix to a PyTorch tensor and add a batch dimension
        return torch.tensor(pos_encoding, dtype=torch.float32).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the 2D positional encoding layer.

        Args:
        - x (Tensor): The input tensor with shape (batch_size, channels, height, width).

        Returns:
        - Tensor: The output tensor with shape (batch_size, channels * height * width).
        """
        # Add the positional encoding to the input tensor and return the result
        return x + self.pos_encoding.to(x.device)
