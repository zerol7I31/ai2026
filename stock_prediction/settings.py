import os

import torch


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_BASE = os.path.join(ROOT_DIR, "科大云盘", "A股数据")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

DAILY_DTYPE = {
    "ts_code": "category",
    "open": "float32",
    "high": "float32",
    "low": "float32",
    "close": "float32",
    "pre_close": "float32",
    "change": "float32",
    "pct_chg": "float32",
    "vol": "float32",
    "amount": "float32",
    "vwap": "float32",
}

DAILY_USECOLS = [
    "ts_code",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
    "vwap",
]

MLP_CONFIG = {
    "stock_pool": "hs300",
    "max_stocks": 200,
    "seq_len": 20,
    "pred_horizon": 1,
    "batch_size": 256,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "dropout": 0.3,
    "hidden_dims": [256, 128, 64],
    "train_start": "2016-01-01",
    "train_end": "2023-12-31",
    "val_start": "2024-01-01",
    "val_end": "2024-12-31",
    "test_start": "2025-01-01",
    "test_end": "2026-05-27",
    "initial_capital": 1000000.0,
    "commission_rate": 0.0003,
    "slippage": 0.001,
    "top_n_hold": 20,
    "daily_trade_n": 3,
    "competition_start": "2026-06-01",
    "competition_end": "2026-06-12",
    "patience": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

GRU_CONFIG = {
    "stock_pool": "hs300",
    "max_stocks": 300,
    "seq_len": 30,
    "pred_horizon": 5,
    "batch_size": 256,
    "epochs": 80,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "dropout": 0.4,
    "hidden_size": 128,
    "num_layers": 2,
    "embed_dim": 16,
    "train_start": "2016-01-01",
    "train_end": "2023-12-31",
    "val_start": "2024-01-01",
    "val_end": "2024-12-31",
    "test_start": "2025-01-01",
    "test_end": "2026-05-26",
    "initial_capital": 1000000.0,
    "commission_rate": 0.0003,
    "slippage": 0.001,
    "top_n_hold": 20,
    "daily_trade_n": 3,
    "competition_start": "2026-06-01",
    "competition_end": "2026-06-12",
    "patience": 12,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "label_type": "cs_rank",
    "ensemble_seeds": [42, 123, 456, 789, 2024],
    "ic_loss_weight": 0.5,
    "mixup_alpha": 0.2,
    "warmup_epochs": 5,
}

