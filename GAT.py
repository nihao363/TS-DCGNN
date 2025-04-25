import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GATLayer(nn.Module):
    def __init__(self, h2_dim, num_heads):
        super(GATLayer, self).__init__()
        # GAT layer: Applying multi-head attention
        self.gat = GATConv(h2_dim, h2_dim, heads=num_heads, concat=True)

    def forward(self, x, edge_index):
        # Apply the GAT layer (multi-head attention)
        x = self.gat(x, edge_index)
        return x
