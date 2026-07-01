class EarlyStopping:
    """当验证集指标不再提升时提前停止训练。

    可被 Trainer、GAT、GraphSAGE 等所有训练流程复用。
    """

    def __init__(self, patience=50, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.should_stop

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.should_stop = False
