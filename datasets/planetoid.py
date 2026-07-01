from torch_geometric.datasets import Planetoid
from datasets.base import BaseDataset


class PlanetoidDataset(BaseDataset):
    """加载 Planetoid 系列数据集 (Cora / CiteSeer / PubMed)。"""

    def __init__(self, name="Cora", root="data"):
        self.dataset = Planetoid(root=root, name=name)

    def get(self):
        data = self.dataset[0]
        return self.dataset, data
