from torch_geometric.datasets import NELL
from datasets.base import BaseDataset


class NELLDataset(BaseDataset):
    """加载 NELL 数据集"""

    def __init__(self, root="data/NELL"):
        self.dataset = NELL(root=root)

    def get(self):
        data = self.dataset[0]
        return self.dataset, data