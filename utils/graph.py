import torch


def add_self_loops(edge_index, num_nodes):
    """为边索引添加自环，返回 A + I。

    Returns:
        edge_index: [2, E + N]
        edge_weight: [E + N], 全 1
    """
    loop = torch.arange(num_nodes, dtype=edge_index.dtype, device=edge_index.device)
    loop = loop.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop], dim=1)
    edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    return edge_index, edge_weight


def degree(edge_index, edge_weight, num_nodes):
    """计算每个节点的度数。

    Returns:
        deg: [num_nodes]
    """
    row, _ = edge_index
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, edge_weight)
    return deg


def normalize_adj(edge_index, edge_weight, num_nodes):
    """对称归一化：D^{-1/2} A D^{-1/2}。

    Returns:
        edge_weight: 归一化后的边权重
    """
    row, col = edge_index
    deg = degree(edge_index, edge_weight, num_nodes)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_weight


def gcn_norm(edge_index, num_nodes, use_self_loops=True):
    """GCN 归一化：加自环 + 对称归一化 D^{-1/2} A D^{-1/2}。

    Returns:
        edge_index: [2, E']（含自环）
        edge_weight: [E'], 归一化权重
    """
    if use_self_loops:
        edge_index, edge_weight = add_self_loops(edge_index, num_nodes)
    else:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    edge_weight = normalize_adj(edge_index, edge_weight, num_nodes)
    return edge_index, edge_weight
