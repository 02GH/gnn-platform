import torch
import torch.nn as nn


class BaseConv(nn.Module):
    """GNN 卷积层抽象基类 —— 定义 message → aggregate → update 统一接口。

    子类只需实现 message()，aggregate() 和 update() 有默认实现。
    """

    def message(self, x_j, edge_weight):
        """对每条边计算消息。子类必须实现。

        Args:
            x_j: 源节点特征 [E, in_channels]
            edge_weight: 边权重 [E] 或 None

        Returns:
            messages: [E, out_channels]
        """
        raise NotImplementedError

    def aggregate(self, messages, index, dim_size):
        """聚合消息到目标节点。默认 sum 聚合。

        Args:
            messages: [E, out_channels]
            index: 目标节点索引 [E]
            dim_size: 节点总数

        Returns:
            aggr_out: [num_nodes, out_channels]
        """
        out = torch.zeros(dim_size, messages.size(1),
                          device=messages.device, dtype=messages.dtype)
        out.index_add_(0, index, messages)
        return out

    def update(self, aggr_out):
        """更新节点表示。默认恒等映射。

        Args:
            aggr_out: [num_nodes, out_channels]

        Returns:
            output: [num_nodes, out_channels]
        """
        return aggr_out
