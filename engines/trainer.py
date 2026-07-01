import torch
from utils.metrics import compute_accuracy
from utils.device import reset_gpu_memory, log_gpu_memory
from utils.early_stopping import EarlyStopping


class Trainer:
    def __init__(self, model, optimizer, criterion, data,
                 edge_index, edge_weight, logger, ckpt,
                 scheduler=None, patience=50, writer=None,
                 task_type="single_label", config=None):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.data = data
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        self.logger = logger
        self.ckpt = ckpt
        self.scheduler = scheduler
        self.writer = writer
        self.task_type = task_type
        self.config = config

        self.best_val = 0
        self.best_epoch = 0
        self.best_test = 0
        self.early_stopping = EarlyStopping(patience=patience)

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(self.data.x, self.edge_index, self.edge_weight)

        y_train = self.data.y[self.data.train_mask]
        if self.task_type == "multi_label":
            y_train = y_train.float()

        loss = self.criterion(out[self.data.train_mask], y_train)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        out = self.model(self.data.x, self.edge_index, self.edge_weight)

        accs = []
        for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]:
            acc = compute_accuracy(out, self.data.y, mask, self.task_type)
            accs.append(acc)

        return accs

    def fit(self, epochs):
        reset_gpu_memory()

        for epoch in range(1, epochs + 1):

            loss = self.train_epoch()
            train_acc, val_acc, test_acc = self.evaluate()

            if self.scheduler is not None:
                self.scheduler.step(val_acc)

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.log(
                f"Epoch {epoch:3d} | Loss {loss:.4f} | "
                f"Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f} | "
                f"LR {current_lr:.6f}"
            )

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", loss, epoch)
                self.writer.add_scalar("Acc/train", train_acc, epoch)
                self.writer.add_scalar("Acc/val", val_acc, epoch)
                self.writer.add_scalar("Acc/test", test_acc, epoch)
                self.writer.add_scalar("LR", current_lr, epoch)

            self.ckpt.save_last(self.model, self.optimizer, epoch, self.config)

            if val_acc > self.best_val:
                self.best_val = val_acc
                self.best_epoch = epoch
                self.best_test = test_acc
                self.ckpt.save_best(self.model, self.optimizer, epoch, val_acc,
                                    self.config)

            if self.early_stopping.step(val_acc):
                self.logger.log(
                    f"Early stopping at epoch {epoch} "
                    f"(best val: {self.best_val:.4f}, Epoch: {self.best_epoch})"
                )
                break

        self.logger.log(
            f"训练结束，最佳验证准确率: {self.best_val:.4f} "
            f"(Epoch: {self.best_epoch})"
        )

        peak_mb = log_gpu_memory()
        if peak_mb > 0:
            self.logger.log(f"Peak GPU Memory: {peak_mb:.2f} MB")

        # 写入全局 summary.csv（字段对齐 Logger._SUMMARY_FIELDS）
        summary = {
            "dataset": (self.config.get("data", {}).get("name", "")
                        if self.config else ""),
            "model": (self.config.get("model", {}).get("name", "")
                      if self.config else ""),
            "best_val_acc": self.best_val,
            "best_test_acc": self.best_test,
            "best_epoch": self.best_epoch,
            "gpu_memory_mb": peak_mb,
            "task_type": self.task_type,
        }
        self.logger.save_summary(summary)
