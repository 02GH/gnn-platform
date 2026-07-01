import os
import torch


class CheckpointManager:
    def __init__(self, exp_dir):
        self.path = os.path.join(exp_dir, "checkpoints")
        os.makedirs(self.path, exist_ok=True)

        self.best_path = os.path.join(self.path, "best.pt")
        self.last_path = os.path.join(self.path, "last.pt")

    # ---------- save ----------
    def save_best(self, model, optimizer, epoch, val_acc, config=None):
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
        }
        if config is not None:
            payload["config"] = config
        torch.save(payload, self.best_path)

    def save_last(self, model, optimizer, epoch, config=None):
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        if config is not None:
            payload["config"] = config
        torch.save(payload, self.last_path)

    # ---------- load ----------
    def load_best(self, model, optimizer=None):
        return self._load(self.best_path, model, optimizer)

    def load_last(self, model, optimizer=None):
        return self._load(self.last_path, model, optimizer)

    def _load(self, path, model, optimizer=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点不存在: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])

        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

        return {
            "epoch": ckpt.get("epoch", 0),
            "val_acc": ckpt.get("val_acc", 0.0),
            "config": ckpt.get("config", {}),
        }
