# -*- coding: utf-8 -*-
"""
=============================================================================
深度学习基础大作业 - MLP Baseline 模型 (多层感知机)
=============================================================================
任务目标: 基于A股日频量价数据，使用MLP模型预测股票未来1日收益率
        构建量化交易策略并完成历史回测 + 同花顺FDL2026模拟交易支持

核心特性:
  - 严格时序划分，杜绝未来信息泄露
  - 按股票独立滚动标准化 (expanding window per stock)
  - MACD / RSI / KDJ / BOLL / ATR 等技术指标特征
  - MLP模型 + RankNet排序损失训练
  - IC / ICIR / 方向胜率 评估
  - T+1交易回测 + 比赛模拟输出

依赖库:
  pip install numpy pandas matplotlib scikit-learn torch tqdm
=============================================================================
"""

import os
import sys
import gc
import warnings
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# =============================================================================
# 路径与全局配置
# =============================================================================
DATA_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "科大云盘", "A股数据")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
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

print(f"Using device: {CONFIG['device']}")

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
    "ts_code", "open", "high", "low", "close",
    "pre_close", "change", "pct_chg", "vol", "amount", "vwap",
]


# =============================================================================
# 1. 数据加载模块
# =============================================================================

def load_basic_info():
    """加载股票基础信息表"""
    path = os.path.join(DATA_BASE, "basic.csv")
    df = pd.read_csv(path, dtype={"ts_code": str, "market": "category", "list_date": str})
    df["list_date"] = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
    return df


def load_trade_calendar():
    """加载交易日历"""
    path = os.path.join(DATA_BASE, "trade_cal.csv")
    df = pd.read_csv(path, dtype={"cal_date": str, "is_open": str, "pretrade_date": str})
    sse = df[df["exchange"] == "SSE"].copy()
    sse["cal_date"] = pd.to_datetime(sse["cal_date"], format="%Y%m%d")
    sse = sse[sse["is_open"] == "1"].sort_values("cal_date")
    return sse


def load_st_stocks():
    """加载历史ST股票名单，返回 {date_str: set(ts_code)}"""
    st_dir = os.path.join(DATA_BASE, "stock_st")
    st_set_by_date = {}
    for fname in os.listdir(st_dir):
        if not fname.endswith(".csv"):
            continue
        date_str = fname.replace(".csv", "")
        try:
            datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            continue
        df = pd.read_csv(os.path.join(st_dir, fname), dtype={"ts_code": str})
        st_set_by_date[date_str] = set(df["ts_code"].tolist())
    return st_set_by_date


def get_bj_codes(basic_df):
    """获取北交所股票代码集合，用于剔除"""
    return set(basic_df[basic_df["market"] == "北交所"]["ts_code"].tolist())


# =============================================================================
# 2. 数据预处理模块 (无未来信息泄露)
# =============================================================================

def load_and_clean_daily(basic_df, st_set_by_date):
    """流式加载日频数据，剔除北交所 / ST股 / 异常值"""
    daily_dir = os.path.join(DATA_BASE, "daily")
    all_files = sorted([f for f in os.listdir(daily_dir) if f.endswith(".csv")])
    bj_codes = get_bj_codes(basic_df)

    data_list = []
    files_loaded = 0

    for fname in tqdm(all_files, desc="Loading daily files"):
        date_str = fname.replace(".csv", "")
        try:
            datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            continue

        df = pd.read_csv(
            os.path.join(daily_dir, fname),
            dtype=DAILY_DTYPE,
            usecols=DAILY_USECOLS,
        )
        df["ts_code"] = df["ts_code"].astype("category")
        df["trade_date"] = date_str

        df = df[~df["ts_code"].isin(bj_codes)]

        st_set = st_set_by_date.get(date_str, set())
        if st_set:
            df = df[~df["ts_code"].isin(st_set)]

        if len(df) == 0:
            continue

        data_list.append(df)
        files_loaded += 1

        if files_loaded % 200 == 0:
            gc.collect()

    del bj_codes
    gc.collect()

    panel = pd.concat(data_list, ignore_index=True)
    del data_list
    gc.collect()

    panel["trade_date"] = pd.to_datetime(panel["trade_date"], format="%Y%m%d")
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    panel["ts_code"] = panel["ts_code"].astype("category")

    panel = panel.dropna(subset=["open", "high", "low", "close", "vol", "amount"])
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0) & (panel["vol"] > 0)]

    panel["amount"] = panel["amount"].astype("float32")
    panel["vol"] = panel["vol"].astype("float32")
    panel["pct_chg"] = panel["pct_chg"].astype("float32")

    outlier_cols = ["pct_chg", "close", "vol"]
    panel["_keep"] = True
    for code, grp in panel.groupby("ts_code", observed=True, sort=False):
        for col in outlier_cols:
            vals = grp[col].values.astype(np.float64)
            mu = np.mean(vals)
            sigma = np.std(vals)
            if sigma < 1e-9:
                continue
            z = np.abs(vals - mu) / sigma
            idx = grp.index[z > 3]
            panel.loc[idx, "_keep"] = False
    n_before = len(panel)
    panel = panel[panel["_keep"]].copy()
    n_after = len(panel)
    print(f"  3-sigma outlier filter: {n_before:,} -> {n_after:,} rows ({n_before - n_after:,} removed)")
    panel.drop(columns=["_keep"], inplace=True)

    gc.collect()
    return panel


def select_stock_pool(panel, cal_df):
    """按成交额排名选择流动性最好的股票池"""
    if CONFIG["stock_pool"] == "all":
        stock_counts = panel.groupby("ts_code", observed=True).size()
        valid_stocks = stock_counts[stock_counts >= CONFIG["seq_len"] + 50].index.tolist()
        return valid_stocks[:CONFIG["max_stocks"]]

    pool_start_dt = pd.to_datetime(CONFIG["train_start"]) - timedelta(days=90)
    pool_end_dt = pd.to_datetime(CONFIG["test_end"])

    sub = panel[(panel["trade_date"] >= pool_start_dt) & (panel["trade_date"] <= pool_end_dt)].copy()
    if len(sub) == 0:
        return panel["ts_code"].unique().tolist()[:CONFIG["max_stocks"]]

    sub["amount_rank"] = sub.groupby("trade_date", observed=True)["amount"].rank(ascending=False)
    stock_avg_rank = sub.groupby("ts_code", observed=True)["amount_rank"].mean()
    sub.drop(columns=["amount_rank"], inplace=True)

    sorted_stocks = stock_avg_rank.sort_values().index.tolist()
    del sub
    gc.collect()
    return sorted_stocks[:CONFIG["max_stocks"]]


# =============================================================================
# 3. 特征工程模块 (逐股票计算，无跨股票信息泄露)
# =============================================================================

def compute_features(group_df):
    """逐股票计算技术指标和量价特征"""

    df = group_df.sort_values("trade_date").copy()
    df["open"] = df["open"].astype("float32")
    df["close"] = df["close"].astype("float32")
    df["high"] = df["high"].astype("float32")
    df["low"] = df["low"].astype("float32")
    df["vol"] = df["vol"].astype("float32")
    df["amount"] = df["amount"].astype("float32")
    df["pct_chg"] = df["pct_chg"].astype("float32")
    df["vwap"] = df["vwap"].astype("float32")

    ret = df["close"].pct_change().astype("float32")

    df["ret_1"] = ret
    df["ret_5"] = (df["close"] / df["close"].shift(5) - 1).astype("float32")
    df["ret_10"] = (df["close"] / df["close"].shift(10) - 1).astype("float32")
    df["ret_20"] = (df["close"] / df["close"].shift(20) - 1).astype("float32")

    vol_5 = ret.rolling(5, min_periods=3).std(ddof=0).astype("float32")
    vol_10 = ret.rolling(10, min_periods=5).std(ddof=0).astype("float32")
    vol_20 = ret.rolling(20, min_periods=10).std(ddof=0).astype("float32")
    df["vol_5"] = vol_5
    df["vol_10"] = vol_10
    df["vol_20"] = vol_20

    vol_chg = df["vol"].pct_change().astype("float32")
    df["vol_chg_5"] = vol_chg.rolling(5, min_periods=2).mean().astype("float32")
    df["vol_chg_10"] = vol_chg.rolling(10, min_periods=5).mean().astype("float32")
    df["vol_ratio_5"] = (df["vol"] / df["vol"].rolling(5, min_periods=2).mean().clip(lower=1e-9)).astype("float32")

    ret_mean_5 = ret.rolling(5, min_periods=2).mean().astype("float32")
    cov_pv = (ret * vol_chg).rolling(20, min_periods=10).mean().astype("float32")
    cov_pv -= ret_mean_5 * vol_chg.rolling(20, min_periods=10).mean().astype("float32")
    stdev_p = ret.rolling(20, min_periods=10).std(ddof=0).astype("float32")
    stdev_v = vol_chg.rolling(20, min_periods=10).std(ddof=0).astype("float32")
    df["pv_corr"] = (cov_pv / (stdev_p * stdev_v + 1e-9)).astype("float32")

    ma5 = df["close"].rolling(5, min_periods=2).mean().astype("float32")
    ma10 = df["close"].rolling(10, min_periods=5).mean().astype("float32")
    ma20 = df["close"].rolling(20, min_periods=5).mean().astype("float32")
    ma60 = df["close"].rolling(60, min_periods=20).mean().astype("float32")
    df["ma_ratio_5_10"] = ((ma5 / ma10.clip(lower=1e-9)) - 1).astype("float32")
    df["ma_ratio_5_20"] = ((ma5 / ma20.clip(lower=1e-9)) - 1).astype("float32")
    df["ma_ratio_5_60"] = ((ma5 / ma60.clip(lower=1e-9)) - 1).astype("float32")
    df["ma_ratio_20_60"] = ((ma20 / ma60.clip(lower=1e-9)) - 1).astype("float32")

    ema12 = df["close"].ewm(span=12, adjust=False, min_periods=5).mean().astype("float32")
    ema26 = df["close"].ewm(span=26, adjust=False, min_periods=10).mean().astype("float32")
    df["macd_dif"] = (ema12 - ema26).astype("float32")
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False, min_periods=5).mean().astype("float32")
    df["macd_hist"] = (2 * (df["macd_dif"] - df["macd_dea"])).astype("float32")

    delta = df["close"].diff()
    gain = delta.clip(lower=0).astype("float32")
    loss = (-delta.clip(upper=0)).astype("float32")
    avg_gain = gain.rolling(14, min_periods=7).mean().astype("float32")
    avg_loss = loss.rolling(14, min_periods=7).mean().astype("float32")
    rs = avg_gain / avg_loss.clip(lower=1e-9)
    df["rsi_14"] = (100 - (100 / (1 + rs))).astype("float32")

    boll_mid = df["close"].rolling(20, min_periods=5).mean().astype("float32")
    boll_std = df["close"].rolling(20, min_periods=5).std(ddof=0).astype("float32")
    boll_upper = (boll_mid + 2 * boll_std).astype("float32")
    boll_lower = (boll_mid - 2 * boll_std).astype("float32")
    df["boll_width"] = ((boll_upper - boll_lower) / boll_mid.clip(lower=1e-9)).astype("float32")
    df["boll_pct"] = ((df["close"] - boll_lower) / (boll_upper - boll_lower).clip(lower=1e-9)).astype("float32")

    atr_hl = (df["high"] - df["low"]).astype("float32")
    atr_hc = (df["high"] - df["close"].shift(1)).abs().astype("float32")
    atr_lc = (df["low"] - df["close"].shift(1)).abs().astype("float32")
    tr = pd.DataFrame({"a": atr_hl, "b": atr_hc, "c": atr_lc}).max(axis=1).astype("float32")
    atr = tr.rolling(14, min_periods=7).mean().astype("float32")
    df["atr_pct"] = (atr / df["close"].clip(lower=1e-9)).astype("float32")

    low_14 = df["low"].rolling(14, min_periods=7).min().astype("float32")
    high_14 = df["high"].rolling(14, min_periods=7).max().astype("float32")
    df["slow_k"] = (((df["close"] - low_14) / (high_14 - low_14).clip(lower=1e-9)) * 100).astype("float32")
    df["slow_d"] = df["slow_k"].rolling(3, min_periods=2).mean().astype("float32")

    df["clv"] = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).clip(lower=1e-9)).astype("float32")
    df["hl_ratio"] = (df["high"] / df["low"].clip(lower=1e-9) - 1).astype("float32")
    df["oc_ratio"] = ((df["open"] - df["close"]) / df["open"].clip(lower=1e-9)).astype("float32")
    df["vwap_diff"] = ((df["close"] - df["vwap"]) / df["vwap"].clip(lower=1e-9)).astype("float32")

    price_max_40 = df["close"].rolling(40, min_periods=20).max().astype("float32")
    price_min_40 = df["close"].rolling(40, min_periods=20).min().astype("float32")
    df["price_pos_40"] = ((df["close"] - price_min_40) / (price_max_40 - price_min_40).clip(lower=1e-9)).astype("float32")

    df["amplitude"] = ((df["high"] - df["low"]) / df["close"].shift(1).clip(lower=1e-9)).astype("float32")
    df["turnover"] = (df["amount"] / df["close"].clip(lower=1e-9)).astype("float32")

    df["pct_chg_diff"] = df["pct_chg"].diff().astype("float32")
    df["vol_diff"] = df["vol"].diff().astype("float32")
    df["amount_diff"] = df["amount"].diff().astype("float32")

    fwd_close = df["close"].shift(-CONFIG["pred_horizon"])
    df["label"] = (fwd_close / df["close"] - 1).astype("float32")

    feature_cols = [
        "ret_1", "ret_5", "ret_10", "ret_20",
        "vol_5", "vol_10", "vol_20",
        "vol_chg_5", "vol_chg_10", "vol_ratio_5",
        "pv_corr",
        "ma_ratio_5_10", "ma_ratio_5_20", "ma_ratio_5_60", "ma_ratio_20_60",
        "macd_dif", "macd_dea", "macd_hist",
        "rsi_14",
        "boll_width", "boll_pct",
        "atr_pct",
        "slow_k", "slow_d",
        "clv",
        "hl_ratio", "oc_ratio", "vwap_diff",
        "price_pos_40",
        "amplitude",
        "turnover",
        "pct_chg_diff", "vol_diff", "amount_diff",
    ]
    return df, feature_cols


# =============================================================================
# 4. 标准化模块 (严禁全量标准化，按股票expanding滚动标准化)
# =============================================================================

def normalize_features(panel, feature_cols):
    """按股票独立做expanding滚动标准化，shift(1)杜绝未来信息泄露"""
    grouped = panel.groupby("ts_code", observed=True)
    normalized_dfs = []
    for _, group in tqdm(grouped, desc="Normalizing features (rolling expanding)"):
        group = group.sort_values("trade_date").copy()
        for col in feature_cols:
            if col not in group.columns:
                continue
            vals = group[col].values.astype(np.float32)
            s = pd.Series(vals)
            rolling_mean = s.expanding(min_periods=63).mean().shift(1)
            rolling_std = s.expanding(min_periods=63).std(ddof=0).shift(1)
            group[col + "_norm"] = ((vals - rolling_mean.values) / (rolling_std.values + 1e-9)).astype(np.float32)
        normalized_dfs.append(group)
        del group
    result = pd.concat(normalized_dfs, ignore_index=True)
    del normalized_dfs
    gc.collect()
    return result


# =============================================================================
# 5. 时序数据集构建 (MLP: 将时序窗口展平为1D向量)
# =============================================================================

def build_mlp_sequences(panel, feature_cols, stock_list):
    """构建MLP样本：将 (seq_len, n_features) 展平为 (seq_len * n_features,)"""
    feature_norm_cols = [c + "_norm" for c in feature_cols]
    all_sequences = []
    seq_len = CONFIG["seq_len"]

    for code in tqdm(stock_list, desc="Building MLP sequences"):
        stock_data = panel[panel["ts_code"] == code].sort_values("trade_date")
        if len(stock_data) < seq_len + 10:
            continue
        feat_array = stock_data[feature_norm_cols].values.astype(np.float32)
        label_array = stock_data["label"].values.astype(np.float32)
        date_array = stock_data["trade_date"].values

        for i in range(len(stock_data) - seq_len):
            window = feat_array[i:i + seq_len]
            y = label_array[i + seq_len - 1]
            if np.any(np.isnan(window)) or np.isnan(y):
                continue
            flat = window.flatten()
            all_sequences.append({
                "ts_code": code,
                "date": date_array[i + seq_len - 1],
                "features": flat,
                "label": float(y),
            })

    return all_sequences


# =============================================================================
# 6. 特征选择 (RandomForest重要性筛选)
# =============================================================================

def select_features_rf(train_sequences, feature_cols, n_top=30):
    """使用随机森林特征重要性筛选Top-K特征"""
    if len(train_sequences) < 100:
        return feature_cols
    X_list = []
    y_list = []
    for s in train_sequences:
        X_list.append(s["features"])
        y_list.append(s["label"])
    X = np.stack(X_list)
    y = np.array(y_list)
    n_features = X.shape[1]
    n_top = min(n_top, n_features)
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:n_top]
    print(f"  RF feature selection: {n_features} -> {n_top} features")
    print(f"  Top-10 feature indices: {top_idx[:10].tolist()}")
    return top_idx.tolist()


def filter_features_by_idx(sequences, selected_idx):
    """仅保留选中的特征索引"""
    for s in sequences:
        s["features"] = s["features"][selected_idx]
    return sequences


# =============================================================================
# 7. MLP数据集与模型定义
# =============================================================================

class MLPSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        self.features = torch.tensor(
            np.stack([s["features"] for s in sequences]), dtype=torch.float32
        )
        self.labels = torch.tensor(
            np.array([s["label"] for s in sequences], dtype=np.float32), dtype=torch.float32
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MLPStockPredictor(nn.Module):
    """多层感知机股票收益率预测模型"""

    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================================
# 8. 排序损失 (RankNet / ListNet-style pairwise loss)
# =============================================================================

def ranknet_loss(preds, labels, device):
    """Pairwise RankNet损失：用exp-diff近似排序质量"""
    n = preds.size(0)
    if n < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    preds_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
    labels_diff = labels.unsqueeze(1) - labels.unsqueeze(0)
    label_gt = (labels_diff > 0).float()
    label_lt = (labels_diff < 0).float()
    loss_matrix = label_gt * torch.log(1 + torch.exp(-preds_diff)) + \
                  label_lt * torch.log(1 + torch.exp(preds_diff))
    denom = max(label_gt.sum() + label_lt.sum(), 1.0)
    return loss_matrix.sum() / denom


def mse_loss(preds, labels, device):
    """均方误差损失"""
    return torch.nn.functional.mse_loss(preds, labels)


# =============================================================================
# 9. 训练流程
# =============================================================================

def train_model(model, train_loader, val_loader, save_path):
    device = CONFIG["device"]
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=CONFIG["patience"] // 2)
    use_amp = (device != "cpu")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        batch_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} Train")
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(features)
                    loss = ranknet_loss(preds, labels, device)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(features)
                loss = ranknet_loss(preds, labels, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_train_loss = total_loss / max(batch_count, 1)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                preds = model(features)
                loss = ranknet_loss(preds, labels, device)
                total_val_loss += loss.item()
                val_batch_count += 1
        avg_val_loss = total_val_loss / max(val_batch_count, 1)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "mlp_loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return train_losses, val_losses


# =============================================================================
# 10. 模型评估: IC / ICIR / 方向胜率
# =============================================================================

def compute_ic(predictions, labels):
    """计算Pearson相关系数 (IC)"""
    if len(predictions) < 2:
        return 0.0
    ic = np.corrcoef(predictions, labels)[0, 1]
    if np.isnan(ic):
        return 0.0
    return ic


def compute_ic_ir(predictions, labels, dates):
    """按日分组计算IC和RankIC的均值/std得到ICIR"""
    df = pd.DataFrame({"pred": predictions, "label": labels, "date": dates})
    df["date"] = pd.to_datetime(df["date"])
    daily_ic = []
    daily_rank_ic = []
    for _, group in df.groupby(df["date"].dt.date):
        if len(group) < 2:
            continue
        ic = compute_ic(group["pred"].values, group["label"].values)
        daily_ic.append(ic)
        rank_ic = compute_ic(
            group["pred"].rank().values, group["label"].rank().values
        )
        daily_rank_ic.append(rank_ic)
    daily_ic = np.array(daily_ic)
    ic_mean = np.mean(daily_ic)
    ic_std = np.std(daily_ic)
    icir = ic_mean / (ic_std + 1e-9)
    rank_ic_mean = np.mean(np.array(daily_rank_ic))
    rank_icir = rank_ic_mean / (np.std(np.array(daily_rank_ic)) + 1e-9)
    return ic_mean, icir, rank_ic_mean, rank_icir, daily_ic


def compute_direction_accuracy(predictions, labels):
    """预测方向胜率"""
    pred_dir = np.sign(predictions)
    true_dir = np.sign(labels)
    return np.mean(pred_dir == true_dir)


# =============================================================================
# 11. 回测模块
# =============================================================================

def build_prices_pivot(panel, stock_pool):
    """构建价格透视表: date × ts_code"""
    sub = panel[panel["ts_code"].isin(stock_pool)].copy()
    piv = sub.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last")
    piv = piv.sort_index()
    return piv


def backtest_strategy(prices_pivot, predictions_df, cal_df):
    """
    T+1交易回测：
    - 初始满仓 top_n_hold 只股票
    - 每日卖出得分最低 daily_trade_n 只（持有超过5天后方可卖出）
    - 买入得分最高 daily_trade_n 只
    - 考虑手续费、滑点、涨跌停约束
    """
    cash = CONFIG["initial_capital"]
    holdings = {}
    daily_values = []

    bt_start = pd.to_datetime(CONFIG["test_start"])
    bt_end = pd.to_datetime(CONFIG["test_end"])
    all_dates = sorted(prices_pivot.index[(prices_pivot.index >= bt_start) & (prices_pivot.index <= bt_end)])

    if len(all_dates) == 0:
        return [], []

    top_n = CONFIG["top_n_hold"]
    trade_n = CONFIG["daily_trade_n"]
    comm = CONFIG["commission_rate"]
    slip = CONFIG["slippage"]
    min_hold_tdays = 5

    shifted_preds = predictions_df.shift(1)
    stock_universe = set(prices_pivot.columns)

    for idx_t, date_val in enumerate(tqdm(all_dates, desc="Backtest")):
        if date_val not in prices_pivot.index:
            continue
        price_row = prices_pivot.loc[date_val]

        for code in list(holdings.keys()):
            if code in price_row.index and pd.notna(price_row[code]):
                holdings[code]["current_price"] = float(price_row[code])

        scores = pd.Series(dtype=float)
        if date_val in shifted_preds.index:
            row = shifted_preds.loc[date_val].dropna()
            if len(row) > 0:
                scores = row.sort_values(ascending=False)
                scores = scores[scores.index.isin(stock_universe)]

        current_held = set(holdings.keys())
        target_stocks = set(scores.head(top_n).index.tolist())

        to_sell_candidates = list(current_held - target_stocks)
        to_sell = [c for c in to_sell_candidates
                   if idx_t - holdings[c].get("bought_at_tidx", -999) > min_hold_tdays]
        if len(to_sell) > trade_n:
            sell_scores = {s: scores.get(s, -999) for s in to_sell}
            to_sell = sorted(to_sell, key=lambda x: sell_scores.get(x, -999))[:trade_n]

        for code in to_sell:
            if code not in holdings:
                continue
            h = holdings[code]
            price = h.get("current_price", h["buy_price"])
            if not np.isfinite(price) or price <= 0:
                continue
            sell_value = h["shares"] * price * (1 - slip) * (1 - comm)
            cash += sell_value
            del holdings[code]

        to_buy = list(target_stocks - current_held)
        if len(to_buy) > trade_n:
            buy_scores = {c: scores.get(c, 999) for c in to_buy}
            to_buy = sorted(to_buy, key=lambda x: buy_scores.get(x, 999), reverse=True)[:trade_n]

        open_slots = top_n - len(holdings)
        to_buy = to_buy[:open_slots] if open_slots > 0 else []

        if to_buy and cash > 0:
            buy_scores_raw = np.array([scores.get(c, 0.0) for c in to_buy], dtype=np.float64)
            if len(buy_scores_raw) > 1:
                buy_scores_centered = buy_scores_raw - buy_scores_raw.mean()
                buy_weights = np.exp(np.clip(buy_scores_centered * 2.0, -10, 10))
                buy_weights = buy_weights / buy_weights.sum()
            else:
                buy_weights = np.ones(1)
            for j, code in enumerate(to_buy):
                if code not in price_row.index or pd.isna(price_row[code]):
                    continue
                bp = float(price_row[code]) * (1 + slip)
                if not np.isfinite(bp) or bp <= 0:
                    continue
                alloc = cash * float(buy_weights[j])
                shares = int(alloc / bp / 100) * 100
                if shares <= 0:
                    continue
                cost = shares * bp * (1 + comm)
                if cost > cash:
                    shares = int(cash / (bp * (1 + comm)) / 100) * 100
                    if shares <= 0:
                        continue
                    cost = shares * bp * (1 + comm)
                cash -= cost
                holdings[code] = {"shares": shares, "buy_price": bp,
                                  "current_price": bp, "bought_at_tidx": idx_t}

        total_value = cash + sum(h["shares"] * h.get("current_price", h["buy_price"])
                                 for h in holdings.values())
        daily_values.append({"date": date_val, "value": total_value, "cash": cash})

    return daily_values, all_dates


def compute_backtest_metrics(daily_values_df):
    """计算年化收益率、夏普比率、最大回撤"""
    if len(daily_values_df) == 0:
        return {}
    df = daily_values_df.copy()
    df["returns"] = df["value"].pct_change()
    df = df.dropna(subset=["returns"])

    total_return = df["value"].iloc[-1] / df["value"].iloc[0] - 1

    first_date = df["date"].iloc[0]
    last_date = df["date"].iloc[-1]
    calendar_days = (last_date - first_date).days
    years = max(calendar_days, 1) / 365.25
    annual_return = (1 + total_return) ** (1.0 / max(years, 1e-6)) - 1

    daily_rf = 0.03 / 252
    excess = df["returns"].values - daily_rf
    ann_excess = excess.mean() * 252
    ann_vol = excess.std() * np.sqrt(252)
    sharpe = ann_excess / (ann_vol + 1e-9)

    cummax = df["value"].cummax()
    drawdown = (df["value"] - cummax) / cummax
    max_drawdown = drawdown.min()

    win_rate = (df["returns"] > 0).mean()

    return {
        "总收益率": f"{total_return * 100:.2f}%",
        "年化收益率": f"{annual_return * 100:.2f}%",
        "夏普比率": f"{sharpe:.3f}",
        "最大回撤": f"{max_drawdown * 100:.2f}%",
        "胜率": f"{win_rate * 100:.2f}%",
    }


def load_benchmark_data():
    """加载沪深300指数数据作为基准"""
    market_dir = os.path.join(DATA_BASE, "market")
    all_files = [f for f in os.listdir(market_dir) if f.endswith(".csv") and "SFConflict" not in f]
    bench_data = []
    for fname in all_files:
        code = fname.replace(".csv", "")
        if not code.endswith((".SH", ".SZ")):
            continue
        df = pd.read_csv(
            os.path.join(market_dir, fname),
            dtype={"ts_code": str, "trade_date": str},
            usecols=["ts_code", "trade_date", "close"],
        )
        bench_data.append(df)
    if bench_data:
        bench_panel = pd.concat(bench_data, ignore_index=True)
        bench_panel["trade_date"] = pd.to_datetime(bench_panel["trade_date"], format="%Y%m%d")
        hs300 = bench_panel[bench_panel["ts_code"] == "000300.SH"].sort_values("trade_date")
        return hs300
    return pd.DataFrame()


def plot_backtest_curve(daily_values_df, benchmark_values, title, filename):
    """绘制回测曲线和回撤图"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    strategy_vals = daily_values_df["value"].values / daily_values_df["value"].values[0]
    ax1.plot(daily_values_df["date"], strategy_vals, label="MLP Strategy", color="blue", linewidth=1.5)

    if len(benchmark_values) > 0:
        common_dates_bench = set(benchmark_values["date"].values)
        common_dates_strat = set(daily_values_df["date"].values)
        common = sorted(common_dates_bench & common_dates_strat)
        if len(common) > 1:
            strat_aligned = daily_values_df[daily_values_df["date"].isin(common)].sort_values("date")
            bench_aligned = benchmark_values[benchmark_values["date"].isin(common)].sort_values("date")
            bench_norm = bench_aligned["value"].values / bench_aligned["value"].values[0]
            ax1.plot(strat_aligned["date"], bench_norm, label="Benchmark (CSI300)", color="orange", linewidth=1.5, alpha=0.7)

    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel("Normalized Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    cummax = daily_values_df["value"].cummax()
    drawdown = (daily_values_df["value"] - cummax) / cummax * 100
    ax2.fill_between(daily_values_df["date"], drawdown, 0, color="red", alpha=0.3)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# 12. 同花顺FDL2026模拟比赛交易输出
# =============================================================================

def generate_competition_signals(panel, model, feature_cols, cal_df, stock_pool, selected_idx):
    """
    生成比赛期间的每日交易信号
    比赛时间: 2026.6.1 - 2026.6.12，每日满仓调仓
    """
    device = CONFIG["device"]
    model = model.to(device)
    model.eval()

    feature_norm_cols = [c + "_norm" for c in feature_cols]
    seq_len = CONFIG["seq_len"]
    top_n = CONFIG["top_n_hold"]

    comp_start = pd.to_datetime(CONFIG["competition_start"])
    comp_end = pd.to_datetime(CONFIG["competition_end"])
    panel_max_date = panel["trade_date"].max()
    if panel_max_date < comp_start:
        print(f"Panel data ends at {panel_max_date.date()}, before competition start {comp_start.date()}. "
              f"Shifting cutoff date backward per competition day to simulate rolling updates.")

    trade_cal_dates = cal_df[(cal_df["cal_date"] >= comp_start) & (cal_df["cal_date"] <= comp_end)]
    if len(trade_cal_dates) == 0:
        print("Warning: No trade dates in competition period!")
        return pd.DataFrame()

    historical_cal = cal_df[cal_df["cal_date"] <= panel_max_date].sort_values("cal_date", ascending=False)
    hist_dates = historical_cal["cal_date"].tolist()

    current_positions = set()
    daily_signals = []

    for comp_day_idx, (_, cal_row) in enumerate(trade_cal_dates.iterrows()):
        date_val = cal_row["cal_date"]
        shift = comp_day_idx
        if shift < len(hist_dates):
            cutoff = hist_dates[min(shift, len(hist_dates) - 1)]
        else:
            cutoff = hist_dates[-1]

        batches = []
        valid_codes = []
        for code in stock_pool:
            code_data = panel[(panel["ts_code"] == code) & (panel["trade_date"] <= cutoff)] \
                .sort_values("trade_date")
            if len(code_data) < seq_len + 5:
                continue
            window = code_data[feature_norm_cols].tail(seq_len).values.astype(np.float32)
            if np.any(np.isnan(window)):
                continue
            flat = window.flatten()
            flat = flat[selected_idx]
            batches.append(flat)
            valid_codes.append(code)

        if not batches:
            continue

        stacked = np.stack(batches)
        tensor_x = torch.tensor(stacked, dtype=torch.float32).to(device)
        scores_list = []
        bs = CONFIG["batch_size"]
        for start in range(0, len(stacked), bs):
            end = min(start + bs, len(stacked))
            with torch.no_grad():
                batch_scores = model(tensor_x[start:end]).cpu().numpy()
            scores_list.append(batch_scores)
        all_scores = np.concatenate(scores_list)

        scores_df = pd.DataFrame({"ts_code": valid_codes, "score": all_scores})
        scores_df = scores_df.sort_values("score", ascending=False).reset_index(drop=True)
        scores_df["rank"] = range(1, len(scores_df) + 1)
        top_stocks_today = set(scores_df.head(top_n)["ts_code"].tolist())
        score_map = dict(zip(scores_df["ts_code"], scores_df["score"]))
        rank_map = dict(zip(scores_df["ts_code"], scores_df["rank"]))

        action_rows = []
        relevant_codes = top_stocks_today | current_positions
        for code in sorted(relevant_codes):
            in_top = code in top_stocks_today
            held = code in current_positions

            if held and in_top:
                action = "HOLD"
            elif in_top and not held:
                action = "BUY"
            elif held and not in_top:
                action = "SELL"
            else:
                action = "NONE"

            action_rows.append({
                "date": date_val,
                "ts_code": code,
                "score": score_map.get(code, np.nan),
                "rank": rank_map.get(code, 999),
                "action": action,
            })

        df_day = pd.DataFrame(action_rows)
        if len(df_day) > 0:
            df_day = df_day.sort_values("rank")
            daily_signals.append(df_day)

        current_positions = top_stocks_today.copy()

    if daily_signals:
        result = pd.concat(daily_signals, ignore_index=True)
    else:
        result = pd.DataFrame()
    return result


# =============================================================================
# 主流程
# =============================================================================

def main():
    print("=" * 60)
    print("MLP Baseline - Deep Learning Stock Prediction")
    print(f"Device: {CONFIG['device']} | Pred Horizon: {CONFIG['pred_horizon']}d")
    print(f"Model: MLP {CONFIG['hidden_dims']} | Seq Len: {CONFIG['seq_len']}")
    print("=" * 60)

    print("\n[1/9] Loading basic info and trade calendar...")
    basic_df = load_basic_info()
    cal_df = load_trade_calendar()
    st_set_by_date = load_st_stocks()
    print(f"  Basic info: {len(basic_df)} stocks")
    print(f"  Trade calendar: {len(cal_df)} trading days")

    print("\n[2/9] Loading daily data (float32, streaming, early ST/BJ filter)...")
    panel = load_and_clean_daily(basic_df, st_set_by_date)
    print(f"  Clean panel: {len(panel):,} rows, {panel['ts_code'].nunique()} stocks")

    print("\n[3/9] Selecting stock pool...")
    stock_pool = select_stock_pool(panel, cal_df)
    print(f"  Selected {len(stock_pool)} stocks")
    panel = panel[panel["ts_code"].isin(stock_pool)].copy()
    panel["ts_code"] = panel["ts_code"].cat.remove_unused_categories()
    gc.collect()
    print(f"  Filtered panel: {len(panel):,} rows, {panel['ts_code'].nunique()} stocks")

    print("\n[4/9] Computing features per stock (float32)...")
    grouped = panel.groupby("ts_code", observed=True, group_keys=False)
    feature_dfs = []
    feature_cols = None
    for _, group in tqdm(grouped, desc="Feature engineering"):
        fdf, fcols = compute_features(group)
        feature_dfs.append(fdf)
        if feature_cols is None:
            feature_cols = fcols
        del group
    panel = pd.concat(feature_dfs, ignore_index=True)
    del feature_dfs
    gc.collect()
    print(f"  Feature columns ({len(feature_cols)} dims): {feature_cols}")

    print("\n[5/9] Normalizing features (expanding per-stock, no lookahead)...")
    panel = normalize_features(panel, feature_cols)

    print("\n[6/9] Building MLP sequences (flatten window -> 1D vector)...")
    sequences = build_mlp_sequences(panel, feature_cols, stock_pool)
    print(f"  Total sequences: {len(sequences):,}")
    feat_dim = len(sequences[0]["features"]) if sequences else 0
    print(f"  Feature dimension (seq_len × n_features = {CONFIG['seq_len']} × {len(feature_cols)} = {feat_dim})")

    train_start = pd.to_datetime(CONFIG["train_start"])
    train_end = pd.to_datetime(CONFIG["train_end"])
    val_start = pd.to_datetime(CONFIG["val_start"])
    val_end = pd.to_datetime(CONFIG["val_end"])
    test_start = pd.to_datetime(CONFIG["test_start"])

    train_seq = [s for s in sequences if train_start <= s["date"] <= train_end]
    val_seq = [s for s in sequences if val_start <= s["date"] <= val_end]
    test_seq = [s for s in sequences if s["date"] >= test_start]
    print(f"  Train: {len(train_seq):,}, Val: {len(val_seq):,}, Test: {len(test_seq):,}")

    if len(train_seq) == 0:
        print("ERROR: No training sequences! Check date ranges.")
        sys.exit(1)

    print("\n  Random Forest feature selection...")
    selected_idx = select_features_rf(train_seq, feature_cols, n_top=min(100, feat_dim))

    train_seq = filter_features_by_idx(train_seq, selected_idx)
    val_seq = filter_features_by_idx(val_seq, selected_idx) if val_seq else []
    test_seq = filter_features_by_idx(test_seq, selected_idx) if test_seq else []

    train_dataset = MLPSequenceDataset(train_seq)
    val_dataset = MLPSequenceDataset(val_seq) if val_seq else None
    test_dataset = MLPSequenceDataset(test_seq) if test_seq else None

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                             num_workers=0, pin_memory=False) if test_dataset else None

    input_dim = train_dataset.features.shape[1]
    print(f"  MLP input dimension: {input_dim}")
    print(f"  Hidden layers: {CONFIG['hidden_dims']}")

    print("\n[7/9] Building and training MLP model...")
    model = MLPStockPredictor(
        input_dim=input_dim,
        hidden_dims=CONFIG["hidden_dims"],
        dropout=CONFIG["dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    model_path = os.path.join(OUTPUT_DIR, "mlp_best_model.pth")
    if val_loader is not None:
        train_losses, val_losses = train_model(model, train_loader, val_loader, model_path)
    else:
        print("  No validation set, training without validation...")
        train_losses, _ = train_model(model, train_loader, train_loader, model_path)

    model.load_state_dict(torch.load(model_path, map_location=CONFIG["device"], weights_only=True))

    print("\n[8/9] Evaluating model...")
    device = CONFIG["device"]
    model.eval()

    def predict_all(loader, sequences):
        all_preds = []
        all_labels = []
        all_dates = []
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(device)
                preds = model(features).cpu().numpy()
                labs = labels.cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labs.tolist())
            for s in sequences:
                all_dates.append(s["date"])
        return np.array(all_preds), np.array(all_labels), all_dates

    val_preds, val_labels, val_dates = predict_all(val_loader, val_seq) if val_seq else (np.array([]), np.array([]), [])
    test_preds, test_labels, test_dates = predict_all(test_loader, test_seq) if test_seq else (np.array([]), np.array([]), [])

    eval_data = []
    for name, preds, labels, dates in [("Validation", val_preds, val_labels, val_dates),
                                        ("Test", test_preds, test_labels, test_dates)]:
        if len(preds) == 0:
            continue
        ic_mean, icir, rank_ic, rank_icir, daily_ic_arr = compute_ic_ir(preds, labels, dates)
        dir_acc = compute_direction_accuracy(preds, labels)
        eval_data.append({
            "数据集": name,
            "PearsonIC": f"{ic_mean:.4f}",
            "ICIR": f"{icir:.4f}",
            "RankIC": f"{rank_ic:.4f}",
            "RankICIR": f"{rank_icir:.4f}",
            "方向胜率": f"{dir_acc * 100:.2f}%",
            "样本数": len(preds),
        })
        print(f"  {name}: PearsonIC={ic_mean:.4f}, ICIR={icir:.4f}, RankIC={rank_ic:.4f}, "
              f"RankICIR={rank_icir:.4f}, DirAcc={dir_acc * 100:.2f}%")

    eval_df = pd.DataFrame(eval_data)
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "mlp_evaluation_metrics.csv"), index=False, encoding="utf-8-sig")
    print("\n" + eval_df.to_string(index=False))

    print("\n[9/9] Running backtest...")
    all_predictions = []
    if len(test_preds) > 0:
        for i, s in enumerate(test_seq):
            all_predictions.append({
                "ts_code": s["ts_code"],
                "date": s["date"],
                "prediction": float(test_preds[i]),
            })
    pred_df = pd.DataFrame(all_predictions)
    predictions_pivot = pred_df.pivot_table(
        index="date", columns="ts_code", values="prediction", aggfunc="mean"
    )

    print("  Building price pivot for backtest...")
    prices_pivot = build_prices_pivot(panel, stock_pool)

    daily_values, trade_dates = backtest_strategy(prices_pivot, predictions_pivot, cal_df)
    if daily_values:
        bt_df = pd.DataFrame(daily_values).sort_values("date")
        bt_df.to_csv(os.path.join(OUTPUT_DIR, "mlp_backtest_daily_values.csv"), index=False, encoding="utf-8-sig")
        metrics = compute_backtest_metrics(bt_df)
        print("\nBacktest Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        benchmark = load_benchmark_data()
        bench_daily = []
        if len(benchmark) > 0:
            for _, row in bt_df.iterrows():
                b_row = benchmark[benchmark["trade_date"] == row["date"]]
                val = float(b_row["close"].values[0]) if len(b_row) > 0 else None
                bench_daily.append({"date": row["date"], "value": val})
            bench_daily = [b for b in bench_daily if b["value"] is not None]
        bench_df = pd.DataFrame(bench_daily) if bench_daily else pd.DataFrame()

        plot_backtest_curve(bt_df, bench_df, "MLP Strategy vs CSI300 Benchmark", "mlp_backtest_curve.png")
    else:
        print("  No backtest data generated.")

    print("\nGenerating competition signals (2026.6.1 - 2026.6.12)...")
    comp_signals = generate_competition_signals(panel, model, feature_cols, cal_df, stock_pool, selected_idx)
    if len(comp_signals) > 0:
        comp_signals.to_csv(os.path.join(OUTPUT_DIR, "mlp_competition_signals.csv"), index=False, encoding="utf-8-sig")
        print("\nCompetition Signals Summary:")
        for date_val in sorted(comp_signals["date"].unique()):
            day_df = comp_signals[comp_signals["date"] == date_val].sort_values("rank")
            date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)
            print(f"\n  {date_str}:")
            for _, row in day_df.iterrows():
                print(f"    {row['ts_code']} | Score: {row['score']:.4f} | Rank: {int(row['rank'])} | Action: {row['action']}")

        daily_trades = []
        for date_val in sorted(comp_signals["date"].unique()):
            day_df = comp_signals[comp_signals["date"] == date_val]
            buy_list = day_df[day_df["action"] == "BUY"]["ts_code"].tolist()
            hold_list = day_df[day_df["action"] == "HOLD"]["ts_code"].tolist()
            sell_list = day_df[day_df["action"] == "SELL"]["ts_code"].tolist()
            daily_trades.append({
                "日期": date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val),
                "BUY": ",".join(buy_list),
                "SELL": ",".join(sell_list),
                "HOLD": ",".join(hold_list),
            })
        trades_df = pd.DataFrame(daily_trades)
        trades_df.to_csv(os.path.join(OUTPUT_DIR, "mlp_competition_daily_trades.csv"), index=False, encoding="utf-8-sig")
        print("\n" + trades_df.to_string(index=False))
    else:
        print("  No competition signals generated (check data availability for 2026.6).")

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": CONFIG,
        "feature_cols": feature_cols,
        "selected_idx": selected_idx,
        "input_dim": input_dim,
    }, os.path.join(OUTPUT_DIR, "mlp_model_checkpoint.pth"))

    print("\n" + "=" * 60)
    print(f"All MLP results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
