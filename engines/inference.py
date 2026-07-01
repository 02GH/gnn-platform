import torch


class Inference:
    """推理器 —— 使用训练好的模型对新数据进行预测。"""

    def __init__(self, model, edge_index, edge_weight):
        self.model = model
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    @torch.no_grad()
    def predict(self, x):
        """对输入特征 x 做前向推理，返回类别预测。"""
        self.model.eval()
        out = self.model(x, self.edge_index, self.edge_weight)
        return out.argmax(dim=1)

    @torch.no_grad()
    def predict_proba(self, x):
        """返回 softmax 概率分布。"""
        self.model.eval()
        out = self.model(x, self.edge_index, self.edge_weight)
        return out.softmax(dim=1)
