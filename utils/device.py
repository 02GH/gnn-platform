import torch


def get_device(prefer="cuda"):
    """获取可用设备，CUDA 不可用时自动回退 CPU。

    Args:
        prefer: "cuda" / "cpu"
    """
    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARNING] CUDA 不可用，回退到 CPU")
        return torch.device("cpu")

    if prefer != "cpu":
        print(f"[WARNING] 未知设备 '{prefer}'，回退到 CPU")
    return torch.device("cpu")


def print_gpu_info():
    """打印 GPU 信息，用于实验日志。"""
    if not torch.cuda.is_available():
        return "GPU: N/A"

    name = torch.cuda.get_device_name(0)
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    cuda_ver = torch.version.cuda
    return f"GPU: {name} | CUDA: {cuda_ver} | Memory: {mem_total:.0f} MB"


def reset_gpu_memory():
    """重置显存峰值统计。"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def log_gpu_memory():
    """返回 GPU 显存峰值 (MB)，仅在 CUDA 可用时有效。"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0
