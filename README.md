# GNN Experiment Platform

基于 PyTorch + PyG 的图神经网络实验系统，支持多数据集 (Cora/CiteSeer/PubMed/NELL) 的单标签节点分类任务。

## 特性

- **模块化架构** — `models/` → `engines/` → `datasets/` → `utils/` 清晰分层
- **配置驱动** — YAML 配置文件 + CLI 参数覆盖，无需改代码即可切换实验
- **自动任务检测** — 根据 `y.dim()` 自动选择 CrossEntropyLoss 或 BCEWithLogitsLoss
- **工业级日志** — 控制台 + log.txt + TensorBoard + summary.csv 汇总
- **模型管理** — best.pt / last.pt 含 config，可完整复现实验
- **显存追踪** — 每次训练自动记录 GPU 峰值显存
- **Message Passing 架构** — BaseConv 抽象基类，message → aggregate → update，便于扩展 GAT / GraphSAGE

## 项目结构

```
GNN_project/
├── main.py                     # 入口脚本
├── configs/                    # 实验配置文件
│   ├── cora.yaml
│   ├── citeseer.yaml
│   ├── pubmed.yaml
│   └── nell.yaml
├── datasets/                   # 数据层 (Factory Pattern)
│   ├── base.py                 # 抽象基类
│   ├── planetoid.py            # Cora / CiteSeer / PubMed
│   ├── nell.py                 # NELL
│   └── builder.py              # 自动构建
├── models/                     # 模型层
│   ├── base_conv.py            # BaseConv 抽象基类 (message→aggregate→update)
│   ├── layers.py               # GCNLayer
│   └── gcn.py                  # GCN (2层)
├── engines/                    # 训练/评估/推理
│   ├── trainer.py              # 训练器 + 显存追踪
│   ├── evaluator.py            # 独立评估 + 混淆矩阵
│   └── inference.py            # 推理器
└── utils/                      # 工具层
    ├── graph.py                # add_self_loops / degree / normalize / gcn_norm
    ├── device.py               # GPU 管理 + 显存统计
    ├── metrics.py              # 任务类型检测 + 准确率
    ├── logger.py               # 日志系统 (info/warning/error + summary.csv)
    ├── checkpoint.py           # 模型保存/加载 (含 config)
    ├── early_stopping.py       # 早停
    └── seed.py                 # 随机种子
```

## 快速开始

```bash
# 1. 安装依赖
pip install torch torch-geometric pyyaml numpy tensorboard

# 2. 训练
python main.py --config configs/cora.yaml      # Cora
python main.py --config configs/citeseer.yaml  # CiteSeer
python main.py --config configs/pubmed.yaml    # PubMed
python main.py --config configs/nell.yaml      # NELL

# 3. 覆盖超参
python main.py --config configs/cora.yaml --epochs 300 --hidden_dim 64 --lr 0.005

# 4. TensorBoard
tensorboard --logdir runs
```

## 实验结果

| 数据集 | 验证准确率 | 测试准确率 | Epoch | GPU 显存 |
|--------|-----------|-----------|-------|---------|
| Cora | 80.0% | 80.5% | 91 | ~45 MB |
| CiteSeer | 71.0% | 70.3% | 87 | ~38 MB |
| PubMed | 79.8% | 78.4% | 143 | ~52 MB |
| NELL | 53.0% | 46.0% | 15 | ~737 MB |

> NELL 训练节点仅 ~66 个，严重过拟合，结果符合预期。

## License

MIT
