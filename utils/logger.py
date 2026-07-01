import csv
import os
from datetime import datetime


class Logger:
    """工业级日志系统 —— 控制台 + log.txt + summary.csv 汇总。"""

    # 共享的全局汇总文件路径
    _global_summary = "runs/summary.csv"

    def __init__(self, exp_name="gcn"):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_name = exp_name
        self.exp_dir = f"runs/{exp_name}_{timestamp}"
        self.log_path = os.path.join(self.exp_dir, "log.txt")

        os.makedirs(self.exp_dir, exist_ok=True)

    def get_exp_dir(self):
        return self.exp_dir

    # ---------- 基础日志 ----------
    def _write(self, msg):
        print(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def log(self, msg):
        self._write(msg)

    def info(self, msg):
        self._write(f"[INFO] {msg}")

    def warning(self, msg):
        self._write(f"[WARNING] {msg}")

    def error(self, msg):
        self._write(f"[ERROR] {msg}")

    # ---------- 实验汇总 ----------
    # 固定列顺序，确保多次实验 CSV 对齐
    _SUMMARY_FIELDS = [
        "dataset", "model", "best_val_acc", "best_test_acc",
        "best_epoch", "gpu_memory_mb", "task_type",
        "exp_name", "exp_dir",
    ]

    def save_summary(self, metrics: dict):
        """将实验指标追加写入全局 summary.csv（首次写表头，后续追加）。

        Args:
            metrics: dict，键值会自动对齐到固定列，缺失列留空
        """
        metrics["exp_name"] = self.exp_name
        metrics["exp_dir"] = self.exp_dir

        os.makedirs("runs", exist_ok=True)
        file_exists = os.path.exists(self._global_summary)

        with open(self._global_summary, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._SUMMARY_FIELDS,
                                    extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

        self.log(f"Summary appended to {self._global_summary}")
