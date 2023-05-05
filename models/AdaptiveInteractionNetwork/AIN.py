import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from Utils import adaptiveThresholding

class Adaptive(nn.Module):
    """
    Base class for the Adaptive Interaction Network (AIN) architecture.

    The AIN architecture is a graph-based neural network designed to handle feature interactions in a flexible and learnable way.
    It allows the connections between nodes to be learnable, enabling the model to find the most optimal structure for the given data.
    The architecture is suitable for a variety of data types, including tabular, image, and language data.
    
    Attributes:
        inputDim (int): The number of input features.
        outputDim (int): The number of output features.
        loops (int): The number of loops to perform during the forward pass. Defaults to 1.
        bias (bool): Whether or not to include biases in the model. Defaults to True.
        directional (str): The directionality of the graph. Can be "uni" (for unidirectional), "random" (for randomly directed), or "bi" (for bidirectional). Defaults to "uni".
        selfLoops (bool): Whether or not to allow self-loops in the graph. Defaults to True.
    """
    def __init__(self, inputDim: int, outputDim: int, loops: int=1, bias: bool=True, directional: str="uni", selfLoops: bool=True) -> None:
        """
        Initializes the AIN model.

        Args:
            inputDim (int): The number of input features.
            outputDim (int): The number of output features.
            loops (int, optional): The number of loops to perform during the forward pass. Defaults to 1.
            bias (bool, optional): Whether or not to include biases in the model. Defaults to True.
            directional (str, optional): The directionality of the graph. Can be "uni" (for unidirectional), "random" (for randomly directed), or "bi" (for bidirectional). Defaults to "uni".
            selfLoops (bool, optional): Whether or not to allow self-loops in the graph. Defaults to True.
        """
        super().__init__()
        
        # Check if the directional value is one of the allowed values
        allowed_directional = ["uni", "random", "bi"]
        if directional not in allowed_directional:
            raise ValueError(f"Invalid value for 'directional'. Allowed values are {allowed_directional}")

        # Store the input arguments as instance variables
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.loops = loops
        self.bias = bias
        self.directional = directional
        self.selfLoops = selfLoops

        # Initialize the learnable parameters of the model
        self.nodeWeights = nn.Parameter(torch.ones((1, self.inputDim)))
        self.edgeWeights = nn.Parameter(torch.normal(0, 1, size=(self.inputDim, self.inputDim)))
        self.nodeBiases = nn.Parameter(torch.normal(0, 1, size=(1, self.inputDim)))
        self.adjacencyMat = nn.Parameter(torch.normal(0, 1, size=(self.inputDim, self.inputDim)))
        if inputDim != outputDim:
            self.outputTransform = nn.Parameter(torch.normal(0, 1, size=(self.inputDim, self.outputDim)))

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the forward pass of the AIN base model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, inputDim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, outputDim).
        """
        
        # Apply the adaptive thresholding function to the adjacency matrix
        adjacencyMat = adaptiveThresholding(self.adjacencyMat)

        # Compute the weighted adjacency matrix based on the specified directionality
        if self.directional == "uni":
            weightedAdj = torch.triu(self.edgeWeights * adjacencyMat)
        elif self.directional == "random":
            weightedAdj = self.edgeWeights * adjacencyMat
        elif self.directional == "bi":
            weightedAdj = torch.triu(self.edgeWeights * adjacencyMat)
            weightedAdj = weightedAdj + weightedAdj.T
        
        # Adjust for proper selp loop condition
        if not self.selfLoops:
            weightedAdj = weightedAdj - torch.diag(weightedAdj.diag())
        
        # Perform the specified number of loops
        nodeVals = x * self.nodeWeights
        for _ in range(self.loops):
            # Compute the activations of the nodes
            nodeVals = F.relu(torch.matmul(nodeVals, weightedAdj))
            if self.bias:
                nodeVals = nodeVals * self.nodeWeights + self.nodeBiases
            else:
                nodeVals = nodeVals * self.nodeWeights
        if self.inputDim != self.outputDim:
            return torch.matmul(nodeVals, self.outputTransform)
        return nodeVals