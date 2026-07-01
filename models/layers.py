import torch
import torch.nn as nn
from models.base_conv import BaseConv


class GCNLayer(BaseConv):
    """GCN 图卷积层 —— message / aggregate / update 三段式实现。"""

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def message(self, x_j, edge_weight):
        # x_j = support[col], shape [E, in_c]
        return x_j * edge_weight.unsqueeze(1)

    def forward(self, x, edge_index, edge_weight):
        # 1. 线性变换
        support = x @ self.weight   # [N, out_c]

        # 2. message: 源节点 × 边权重
        row, col = edge_index
        messages = self.message(support[col], edge_weight)  # [E, out_c]

        # 3. aggregate: sum 到目标节点
        out = self.aggregate(messages, row, support.size(0))

        # 4. update: + bias
        out = self.update(out)
        if self.bias is not None:
            out = out + self.bias

        return out
