"""数据集构建器 —— 根据配置自动选择和加载数据集。"""

from datasets.planetoid import PlanetoidDataset
from datasets.nell import NELLDataset

PLANETOID_DATASETS = {"Cora", "CiteSeer", "PubMed"}


def build_dataset(name="Cora", root=None):
    """根据数据集名称构建并返回 (dataset, data) 元组。

    Args:
        name: 数据集名称 (Cora / CiteSeer / PubMed / NELL)
        root: 数据存储目录，默认为 data/{name}

    Returns:
        (dataset, data)
    """
    if name in PLANETOID_DATASETS:
        root = root or f"data/{name}"
        dataset = PlanetoidDataset(name=name, root=root)

    elif name == "NELL":
        root = root or "data/NELL"
        dataset = NELLDataset(root=root)

    else:
        raise ValueError(
            f"未知数据集: {name}。支持: {sorted(PLANETOID_DATASETS)} + NELL"
        )

    return dataset.get()
