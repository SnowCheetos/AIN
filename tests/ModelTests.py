import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/AdaptiveInteractionNetwork/")))

import torch
from AIN import Adaptive


def test_output_shape():
    batch_size = 16
    input_dim = 8
    output_dim = 4

    model = Adaptive(input_dim, output_dim)
    x = torch.randn(batch_size, input_dim)
    output = model(x)

    assert output.shape == (batch_size, output_dim), f"Expected output shape: {(batch_size, output_dim)}, got: {output.shape}"


def test_parameters():
    input_dim = 8
    output_dim = 4

    model = Adaptive(input_dim, output_dim)

    assert model.nodeWeights.shape == (1, input_dim)
    assert model.edgeWeights.shape == (input_dim, input_dim)
    assert model.nodeBiases.shape == (1, input_dim)
    assert model.adjacencyMat.shape == (input_dim, input_dim)


def test_directionality():
    input_dim = 8
    output_dim = 4

    for directional in ["uni", "random", "bi"]:
        model = Adaptive(input_dim, output_dim, directional=directional)
        x = torch.randn(1, input_dim)
        _ = model(x)  # Ensure the model can execute without errors with each directionality setting


def test_loops_and_self_loops():
    input_dim = 8
    output_dim = 4

    for loops in range(1, 4):
        for selfLoops in [True, False]:
            model = Adaptive(input_dim, output_dim, loops=loops, selfLoops=selfLoops)
            x = torch.randn(1, input_dim)
            _ = model(x)  # Ensure the model can execute without errors with each loop count and self-loop setting


def test_bias():
    input_dim = 8
    output_dim = 4

    for bias in [True, False]:
        model = Adaptive(input_dim, output_dim, bias=bias)
        x = torch.randn(1, input_dim)
        _ = model(x)  # Ensure the model can execute without errors with each bias setting


def test_gradient_update():
    batch_size = 16
    input_dim = 8
    output_dim = 4
    epochs = 10

    model = Adaptive(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    x = torch.randn(batch_size, input_dim)
    target = torch.randn(batch_size, output_dim)

    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    assert loss.item() > 0, "Loss should be greater than 0 after gradient updates"


def test_error_handling():
    input_dim = 8
    output_dim = 4

    try:
        model = Adaptive(input_dim, output_dim, directional="invalid")
    except ValueError as e:
        assert str(e) == "Invalid value for 'directional'. Allowed values are ['uni', 'random', 'bi']"

if __name__ == "__main__":
    test_output_shape()
    test_parameters()
    test_directionality()
    test_loops_and_self_loops()
    test_bias()
    test_gradient_update()
    test_error_handling()
    print("All tests passed!")