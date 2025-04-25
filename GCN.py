import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNLayer(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim):
        super(GCNLayer, self).__init__()
        # Define GCN layers
        self.conv1 = GCNConv(g_dim, h1_dim)  # First GCN layer
        self.conv2 = GCNConv(h1_dim, h2_dim)  # Second GCN layer

    def forward(self, node_features, edge_index):
        # Apply the first GCN layer
        x = self.conv1(node_features, edge_index)
        x = nn.functional.leaky_relu(x)  # Apply LeakyReLU activation

        # Apply the second GCN layer
        x = self.conv2(x, edge_index)
        x = nn.functional.leaky_relu(x)  # Apply LeakyReLU activation

        return x
