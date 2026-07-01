import torch.nn as nn
from models.layers import GCNLayer

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()

        self.conv1 = GCNLayer(in_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, out_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight):

        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight)

        return x