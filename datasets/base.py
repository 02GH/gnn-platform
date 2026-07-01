from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """所有数据集的抽象基类。"""

    @abstractmethod
    def get(self):
        """返回 (dataset, data) 元组。"""
        raise NotImplementedError