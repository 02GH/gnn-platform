import torch
from utils.metrics import compute_accuracy


class Evaluator:
    """独立评估器 —— 加载检查点并对测试集进行评估。"""

    def __init__(self, model, data, edge_index, edge_weight,
                 task_type="single_label"):
        self.model = model
        self.data = data
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.task_type = task_type

    @torch.no_grad()
    def evaluate(self):
        """返回 train / val / test 三个 split 的准确率。"""
        self.model.eval()
        out = self.model(self.data.x, self.edge_index, self.edge_weight)

        results = {}
        for split, mask in [("train", self.data.train_mask),
                            ("val", self.data.val_mask),
                            ("test", self.data.test_mask)]:
            acc = compute_accuracy(out, self.data.y, mask, self.task_type)
            results[split] = acc

        return results

    def test(self):
        """仅返回测试集准确率。"""
        return self.evaluate()["test"]

    @torch.no_grad()
    def confusion_matrix(self):
        """返回测试集的混淆矩阵（向量化实现）。

        Returns:
            cm: [num_classes, num_classes] torch.Tensor
        """
        self.model.eval()
        out = self.model(self.data.x, self.edge_index, self.edge_weight)
        pred = out.argmax(dim=1)

        mask = self.data.test_mask
        y_true = self.data.y[mask]
        y_pred = pred[mask]

        num_classes = out.size(1)
        # 向量化：将 (y_true, y_pred) 对映射到一维索引
        index = y_true * num_classes + y_pred
        cm = torch.zeros(num_classes * num_classes, dtype=torch.long,
                         device=out.device)
        cm.scatter_add_(0, index, torch.ones_like(index, dtype=torch.long))
        return cm.view(num_classes, num_classes)
