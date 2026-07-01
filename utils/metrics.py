import torch


def detect_task_type(y):
    """根据标签张量形状自动判断任务类型。

    Args:
        y: 标签张量

    Returns:
        "single_label" 或 "multi_label"
    """
    if y.dim() == 1:
        return "single_label"
    elif y.dim() == 2 and y.size(1) > 1:
        return "multi_label"
    else:
        return "single_label"


def build_criterion(task_type):
    """根据任务类型返回对应的损失函数。"""
    if task_type == "multi_label":
        return torch.nn.BCEWithLogitsLoss()
    return torch.nn.CrossEntropyLoss()


def compute_accuracy(out, y, mask, task_type="single_label"):
    """单标签准确率计算。

    Args:
        out: 模型输出 logits, shape [N, C]
        y: 标签, shape [N]
        mask: 布尔掩码
        task_type: "single_label"（当前仅支持单标签）

    Returns:
        float: 准确率
    """
    if task_type == "multi_label":
        raise NotImplementedError(
            "多标签评估请使用 compute_f1()，暂未实现。"
        )
    pred = out.argmax(dim=1)
    correct = (pred[mask] == y[mask]).float().mean().item()
    return correct
