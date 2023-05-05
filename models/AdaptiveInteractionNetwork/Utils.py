import torch
import torch.nn.functional as F
from torch import Tensor
import networkx as nx
import numpy as np


def adaptiveThresholding(adjacency: Tensor) -> Tensor:
    """
    Applies adaptive thresholding to the given adjacency matrix.

    The adaptive thresholding function applies a ReLU activation function to the output of a hyperbolic tangent function
    applied to the adjacency matrix. This function is commonly used in graph-based neural networks to promote sparsity
    in the connections between nodes.

    Args:
        adjacency (torch.Tensor): The input adjacency matrix of shape (n_nodes, n_nodes).

    Returns:
        torch.Tensor: The output adjacency matrix of shape (n_nodes, n_nodes).
    """
    return F.relu(torch.tanh(adjacency))


def countConnections(model) -> int:
    """
    Computes the number of connections in the graph represented by the model.
    
    Args:
    - model: a GraphConvolutionalNetwork instance
    
    Returns:
    - the number of connections in the graph
    
    """
    # Extract the adjacency matrix from the model
    adjacencyMat = adaptiveThresholding(model.adjacencyMat).detach().cpu().numpy()

    # Modify the adjacency matrix based on the graph directionality
    if model.directional == "bi":
        adjacencyMat = np.triu(adjacencyMat) + np.triu(adjacencyMat).T
    elif model.directional == "uni":
        adjacencyMat = np.triu(adjacencyMat)
        
    # Adjust for proper selp loop condition
    if not model.selfLoops:
        adjacencyMat = adjacencyMat - np.diag(adjacencyMat.diagonal())

    # Count the number of nonzero entries in the adjacency matrix
    return np.where(adjacencyMat > 0)[0].shape[0]


def exportGraph(model) -> nx.Graph:
    """
    Exports a graph from a given model, where nodes represent the model's neurons and edges represent the connections between them.

    Args:
        model (AIN_Base): An instance of the AIN_Base class.

    Returns:
        nx.Graph: A NetworkX graph object representing the model's graph.
    """
    
    # Convert the node weights and adjacency matrix to numpy arrays on the CPU
    nodeWeights = model.nodeWeights.detach().cpu().numpy()
    adjacencyMat = adaptiveThresholding(model.adjacencyMat).detach().cpu().numpy()

    # Create a networkx graph object based on the specified graph directionality
    if model.directional == "bi":
        # For a bidirectional graph, the adjacency matrix is made symmetric
        # and an undirected graph is created
        adjacencyMat = np.triu(adjacencyMat) + np.triu(adjacencyMat).T
        graph = nx.Graph()
    elif model.directional == "random":
        # For a randomly directed graph, a directed graph is created
        graph = nx.DiGraph()
    elif model.directional == "uni":
        # For a unidirectional graph, only the upper triangle of the
        # adjacency matrix is used, and a directed graph is created
        adjacencyMat = np.triu(adjacencyMat)
        graph = nx.DiGraph()

    # Adjust for proper selp loop condition
    if not model.selfLoops:
        adjacencyMat = adjacencyMat - np.diag(adjacencyMat.diagonal())

    # Get the indices and weights of the non-zero elements of the adjacency matrix
    edges = np.where(adjacencyMat > 0)
    edgeWeights = adjacencyMat[edges[0], edges[1]]

    # Normalize the edge weights and combine the indices and weights into a single array
    edgeWeightsNorm = edgeWeights / edgeWeights.max()
    weightedEdge = tuple(np.concatenate([
        edges[0].reshape(-1, 1), 
        edges[1].reshape(-1, 1),
        edgeWeightsNorm.reshape(-1, 1)
    ], 1))

    # Add the weighted edges to the graph and update node weights
    graph.add_weighted_edges_from(weightedEdge, weight="edge weight")
    for i, w in enumerate(nodeWeights):
        graph.nodes[i].update({"node weight": w})

    # Return the final graph
    return graph