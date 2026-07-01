import argparse
import torch
import yaml

from datasets.builder import build_dataset
from models.gcn import GCN
from engines.trainer import Trainer
from utils.logger import Logger
from utils.checkpoint import CheckpointManager
from utils.seed import set_seed
from utils.device import get_device, print_gpu_info
from utils.metrics import detect_task_type, build_criterion
from utils import gcn_norm


# ================= config =================
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="GNN 实验系统")
    parser.add_argument("--config", type=str, default="configs/cora.yaml",
                        help="配置文件路径")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)
    return parser.parse_args()


def merge_cli_overrides(cfg, args):
    """CLI 参数覆盖 YAML 配置（仅覆盖用户显式指定的值）。"""
    overrides = {
        "training": {
            "epochs": args.epochs, "lr": args.lr,
            "weight_decay": args.weight_decay, "patience": args.patience,
        },
        "model": {"hidden_dim": args.hidden_dim, "dropout": args.dropout},
        "data": {"name": args.data},
        "seed": args.seed,
        "device": args.device,
    }
    for section, values in overrides.items():
        if isinstance(values, dict):
            for k, v in values.items():
                if v is not None:
                    cfg.setdefault(section, {})[k] = v
        elif values is not None:
            cfg[section] = values
    return cfg


# ================= main =================
def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)

    # ---- seed ----
    set_seed(cfg["seed"])

    # ---- data ----
    dataset, data = build_dataset(cfg["data"]["name"], cfg["data"]["root"])
    device = get_device(cfg["device"])
    data = data.to(device)

    edge_index, edge_weight = gcn_norm(data.edge_index, data.num_nodes)
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

    # ---- task type (auto-detect) ----
    task_type = detect_task_type(data.y)

    # ---- model ----
    model = GCN(
        dataset.num_features,
        cfg["model"]["hidden_dim"],
        dataset.num_classes,
        cfg["model"]["dropout"],
    ).to(device)

    # ---- optim & criterion ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    criterion = build_criterion(task_type)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20
    ) if cfg["training"].get("scheduler", True) else None

    # ---- logging ----
    logger = Logger(f"gcn_{cfg['data']['name'].lower()}")
    ckpt = CheckpointManager(logger.get_exp_dir())

    # TensorBoard 仅在使用时才导入，避免未安装时报错
    use_tb = cfg["training"].get("tensorboard", True)
    if use_tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(logger.get_exp_dir())
    else:
        writer = None

    # ---- 实验信息 ----
    logger.info(f"Starting experiment: {cfg['data']['name']}")
    logger.info(print_gpu_info())
    logger.info(f"Config: {cfg}")
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Task type: {task_type}")

    # ---- train ----
    trainer = Trainer(
        model, optimizer, criterion, data,
        edge_index, edge_weight, logger, ckpt,
        scheduler=scheduler,
        patience=cfg["training"].get("patience", 50),
        writer=writer,
        task_type=task_type,
        config=cfg,
    )
    trainer.fit(cfg["training"]["epochs"])

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
