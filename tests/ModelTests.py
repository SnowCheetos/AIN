import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.AdaptiveInteractionNetwork.AIN import Adaptive

def test_output_shape():
    batch_size = 16
    input_dim = 8
    output_dim = 4

    model = Adaptive(input_dim, output_dim)
    x = torch.randn(batch_size, input_dim)
    output = model(x)

    assert output.shape == (batch_size, output_dim), f"Expected output shape: {(batch_size, output_dim)}, got: {output.shape}"
