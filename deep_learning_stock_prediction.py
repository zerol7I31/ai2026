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

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

DATA_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "科大云盘", "A股数据")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    "stock_pool": "hs1000",
    "max_stocks": 1000,
    "seq_len": 30,
    "pred_horizon": 5,
    "batch_size": 128,
    "epochs": 50,
    "learning_rate": 0.0001,
    "weight_decay": 1e-5,
    "dropout": 0.2,
    "hidden_size": 128,
    "num_layers": 2,
    "embed_dim": 64,
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
    "patience": 10,
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


def load_basic_info():
    path = os.path.join(DATA_BASE, "basic.csv")
    df = pd.read_csv(path, dtype={"ts_code": str, "market": "category", "list_date": str})
    df["list_date"] = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
    return df


def load_trade_calendar():
    path = os.path.join(DATA_BASE, "trade_cal.csv")
    df = pd.read_csv(path, dtype={"cal_date": str, "is_open": str, "pretrade_date": str})
    sse = df[df["exchange"] == "SSE"].copy()
    sse["cal_date"] = pd.to_datetime(sse["cal_date"], format="%Y%m%d")
    sse = sse[sse["is_open"] == "1"].sort_values("cal_date")
    return sse


def load_st_stocks():
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
    return set(basic_df[basic_df["market"] == "北交所"]["ts_code"].tolist())


def load_and_clean_daily(basic_df, st_set_by_date):
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


def select_stock_pool_optimized(panel, cal_df):
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


def compute_features(group_df):
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
    ma20 = df["close"].rolling(20, min_periods=5).mean().astype("float32")
    ma60 = df["close"].rolling(60, min_periods=20).mean().astype("float32")
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
    df["label_5d"] = (fwd_close / df["close"] - 1).astype("float32")

    feature_cols = [
        "ret_1", "ret_5", "ret_10", "ret_20",
        "vol_5", "vol_10", "vol_20",
        "vol_chg_5", "vol_chg_10", "vol_ratio_5",
        "pv_corr",
        "ma_ratio_5_20", "ma_ratio_5_60", "ma_ratio_20_60",
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


def normalize_features(panel, feature_cols):
    grouped = panel.groupby("ts_code", observed=True)
    normalized_dfs = []
    for _, group in tqdm(grouped, desc="Normalizing features (rolling)"):
        group = group.sort_values("trade_date").copy()
        for col in feature_cols:
            if col not in group.columns:
                continue
            vals = group[col].values.astype(np.float32)
            s = pd.Series(vals)
            rolling_mean = s.rolling(252, min_periods=63).mean().shift(1)
            rolling_std = s.rolling(252, min_periods=63).std(ddof=0).shift(1)
            group[col + "_norm"] = ((vals - rolling_mean.values) / (rolling_std.values + 1e-9)).astype(np.float32)
        normalized_dfs.append(group)
        del group
    result = pd.concat(normalized_dfs, ignore_index=True)
    del normalized_dfs
    gc.collect()
    return result


def build_sequences(panel, feature_cols, stock_list):
    feature_norm_cols = [c + "_norm" for c in feature_cols]
    all_sequences = []
    seq_len = CONFIG["seq_len"]

    for code in tqdm(stock_list, desc="Building sequences"):
        stock_data = panel[panel["ts_code"] == code].sort_values("trade_date")
        if len(stock_data) < seq_len + 10:
            continue
        feat_array = stock_data[feature_norm_cols].values.astype(np.float32)
        label_array = stock_data["label_5d"].values.astype(np.float32)
        date_array = stock_data["trade_date"].values

        for i in range(len(stock_data) - seq_len):
            x = feat_array[i:i + seq_len]
            y = label_array[i + seq_len - 1]
            if np.any(np.isnan(x)) or np.isnan(y):
                continue
            all_sequences.append({
                "ts_code": code,
                "date": date_array[i + seq_len - 1],
                "features": x,
                "label": float(y),
            })

    return all_sequences


class StockSequenceDataset(Dataset):
    def __init__(self, sequences, code_to_id):
        self.sequences = sequences
        self.features = torch.tensor(
            np.stack([s["features"] for s in sequences]), dtype=torch.float32
        )
        self.labels = torch.tensor(
            np.array([s["label"] for s in sequences], dtype=np.float32), dtype=torch.float32
        )
        self.stock_ids = torch.tensor(
            [code_to_id.get(s["ts_code"], 0) for s in sequences], dtype=torch.long
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.stock_ids[idx], idx


class StockGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout, num_stocks, embed_dim):
        super().__init__()
        self.stock_embedding = nn.Embedding(num_stocks, embed_dim)
        self.gru = nn.GRU(
            input_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.gru_norm = nn.LayerNorm(hidden_size * 2)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2 + embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "embedding" in name:
                nn.init.normal_(param, mean=0, std=0.01)
            elif "gru" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
            elif "norm" in name:
                continue
            elif "weight" in name and param.ndimension() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x, stock_ids):
        gru_out, _ = self.gru(x)
        gru_out = self.gru_norm(gru_out)
        attn_weights = self.attention(gru_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        stock_emb = self.stock_embedding(stock_ids)
        combined = torch.cat([context, stock_emb], dim=1)
        output = self.fc(combined)
        return output.squeeze(-1)


def train_model(model, train_loader, val_loader, save_path):
    device = CONFIG["device"]
    model = model.to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        batch_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} Train")
        for features, labels, stock_ids, _ in pbar:
            features, labels, stock_ids = features.to(device), labels.to(device), stock_ids.to(device)
            optimizer.zero_grad()
            preds = model(features, stock_ids)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / max(batch_count, 1)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for features, labels, stock_ids, _ in val_loader:
                features, labels, stock_ids = features.to(device), labels.to(device), stock_ids.to(device)
                preds = model(features, stock_ids)
                loss = criterion(preds, labels)
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
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return train_losses, val_losses


def compute_ic(predictions, labels):
    if len(predictions) < 2:
        return 0, 0
    ic = np.corrcoef(predictions, labels)[0, 1]
    if np.isnan(ic):
        return 0, 0
    return ic, 0


def compute_ic_ir(predictions, labels, dates):
    df = pd.DataFrame({"pred": predictions, "label": labels, "date": dates})
    df["date"] = pd.to_datetime(df["date"])
    daily_ic = []
    daily_rank_ic = []
    for _, group in df.groupby(df["date"].dt.date):
        if len(group) < 2:
            continue
        ic, _ = compute_ic(group["pred"].values, group["label"].values)
        daily_ic.append(ic)
        rank_ic, _ = compute_ic(
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
    pred_dir = np.sign(predictions)
    true_dir = np.sign(labels)
    return np.mean(pred_dir == true_dir)


def build_prices_pivot(panel, stock_pool):
    sub = panel[panel["ts_code"].isin(stock_pool)].copy()
    piv = sub.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last")
    piv = piv.sort_index()
    return piv


def backtest_strategy(prices_pivot, predictions_df, cal_df):
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

        total_value = cash + sum(h["shares"] * h.get("current_price", h["buy_price"]) for h in holdings.values())
        daily_values.append({"date": date_val, "value": total_value, "cash": cash})

    return daily_values, all_dates


def compute_backtest_metrics(daily_values_df):
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


def plot_backtest_curve(daily_values_df, benchmark_values, title, filename):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    strategy_vals = daily_values_df["value"].values / daily_values_df["value"].values[0]
    ax1.plot(daily_values_df["date"], strategy_vals, label="Strategy", color="blue", linewidth=1.5)

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


def generate_competition_signals(panel, ensemble_models, feature_cols, cal_df, stock_pool, code_to_id):
    device = CONFIG["device"]
    for m in ensemble_models:
        m.to(device)
        m.eval()

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
        valid_ids = []
        for code in stock_pool:
            code_data = panel[(panel["ts_code"] == code) & (panel["trade_date"] <= cutoff)] \
                .sort_values("trade_date")
            if len(code_data) < seq_len + 5:
                continue
            window = code_data[feature_norm_cols].tail(seq_len).values.astype(np.float32)
            if np.any(np.isnan(window)):
                continue
            batches.append(window)
            valid_codes.append(code)
            valid_ids.append(code_to_id.get(code, 0))

        if not batches:
            continue

        stacked = np.stack(batches)
        tensor_x = torch.tensor(stacked, dtype=torch.float32).to(device)
        tensor_ids = torch.tensor(valid_ids, dtype=torch.long).to(device)
        scores_list = []
        bs = CONFIG["batch_size"]
        for start in range(0, len(stacked), bs):
            end = min(start + bs, len(stacked))
            with torch.no_grad():
                batch_preds = np.zeros(end - start, dtype=np.float64)
                for m in ensemble_models:
                    batch_preds += m(tensor_x[start:end], tensor_ids[start:end]).cpu().numpy().astype(np.float64)
                batch_preds /= len(ensemble_models)
            scores_list.append(batch_preds)
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


def load_benchmark_data():
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


def main():
    print("=" * 60)
    print("Deep Learning Stock Trend Prediction - GRU with Attention")
    print(f"Memory Optimization: float32 + category dtypes + vectorized ST filter")
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
    panel.info(memory_usage="deep")

    print("\n[3/9] Selecting stock pool (groupby-rank)...")
    stock_pool = select_stock_pool_optimized(panel, cal_df)
    print(f"  Selected {len(stock_pool)} stocks")
    panel = panel[panel["ts_code"].isin(stock_pool)].copy()
    panel["ts_code"] = panel["ts_code"].cat.remove_unused_categories()
    gc.collect()
    print(f"  Filtered panel: {len(panel):,} rows, {panel['ts_code'].nunique()} stocks")
    panel.info(memory_usage="deep")

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

    print("\n[5/9] Normalizing features (rolling per-stock, no lookahead)...")
    panel = normalize_features(panel, feature_cols)

    print("\n[6/9] Building sequences (label: 5-day average forward return)...")
    sequences = build_sequences(panel, feature_cols, stock_pool)
    print(f"  Total sequences: {len(sequences):,}")

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

    code_to_id = {code: i for i, code in enumerate(stock_pool)}
    num_stocks = len(stock_pool)
    print(f"  Stock ID mapping: {num_stocks} unique stocks")

    train_dataset = StockSequenceDataset(train_seq, code_to_id)
    val_dataset = StockSequenceDataset(val_seq, code_to_id) if val_seq else None
    test_dataset = StockSequenceDataset(test_seq, code_to_id) if test_seq else None

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=False) if test_dataset else None

    input_dim = train_dataset.features.shape[2]

    print("\n[7/9] Training ensemble of 3 GRU models (seeds: 42, 123, 456)...")
    input_dim = train_dataset.features.shape[2]
    ensemble_seeds = [42, 123, 456]
    ensemble_models = []

    for si, seed in enumerate(ensemble_seeds):
        set_seed(seed)
        print(f"\n  --- Model {si + 1}/3 (seed={seed}) ---")

        model = StockGRUModel(
            input_dim=input_dim,
            hidden_size=CONFIG["hidden_size"],
            num_layers=CONFIG["num_layers"],
            dropout=CONFIG["dropout"],
            num_stocks=num_stocks,
            embed_dim=CONFIG["embed_dim"],
        )
        print(f"  Model: GRU (input={input_dim}, hidden={CONFIG['hidden_size']}, layers={CONFIG['num_layers']})")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        model_path = os.path.join(OUTPUT_DIR, f"best_model_seed{seed}.pth")
        if val_loader is not None:
            train_losses, val_losses = train_model(model, train_loader, val_loader, model_path)
        else:
            print("  No validation set, training without validation...")
            train_losses, _ = train_model(model, train_loader, train_loader, model_path)

        model.load_state_dict(torch.load(model_path, map_location=CONFIG["device"], weights_only=True))
        model.eval()
        ensemble_models.append(model)

    print(f"\n  Ensemble of {len(ensemble_models)} models trained.")

    print("\n[8/9] Evaluating ensemble...")
    device = CONFIG["device"]

    def ensemble_predict(loader, sequences):
        all_preds = []
        all_labels = []
        all_dates = []
        with torch.no_grad():
            for features, labels, stock_ids, indices in loader:
                features, stock_ids = features.to(device), stock_ids.to(device)
                preds_ensemble = np.zeros(len(features), dtype=np.float64)
                for m in ensemble_models:
                    m.eval()
                    preds_ensemble += m(features, stock_ids).cpu().numpy().astype(np.float64)
                preds_ensemble /= len(ensemble_models)
                labs = labels.cpu().numpy()
                all_preds.extend(preds_ensemble.tolist())
                all_labels.extend(labs.tolist())
                for idx in indices.tolist():
                    all_dates.append(sequences[idx]["date"])
        return np.array(all_preds), np.array(all_labels), all_dates

    val_preds, val_labels, val_dates = ensemble_predict(val_loader, val_seq) if val_seq else (np.array([]), np.array([]), [])
    test_preds, test_labels, test_dates = ensemble_predict(test_loader, test_seq) if test_seq else (np.array([]), np.array([]), [])

    eval_data = []
    for name, preds, labels, dates in [("Validation", val_preds, val_labels, val_dates), ("Test", test_preds, test_labels, test_dates)]:
        if len(preds) == 0:
            continue
        ic_mean, icir, rank_ic, rank_icir, _ = compute_ic_ir(preds, labels, dates)
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
        print(f"  {name}: PearsonIC={ic_mean:.4f}, ICIR={icir:.4f}, RankIC={rank_ic:.4f}, RankICIR={rank_icir:.4f}, DirAcc={dir_acc * 100:.2f}%")

    eval_df = pd.DataFrame(eval_data)
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"), index=False, encoding="utf-8-sig")
    print(eval_df.to_string(index=False))

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

    print("  Building price pivot for vectorized backtest...")
    prices_pivot = build_prices_pivot(panel, stock_pool)

    daily_values, trade_dates = backtest_strategy(prices_pivot, predictions_pivot, cal_df)
    if daily_values:
        bt_df = pd.DataFrame(daily_values).sort_values("date")
        bt_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_daily_values.csv"), index=False, encoding="utf-8-sig")
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

        plot_backtest_curve(bt_df, bench_df, "Strategy vs Benchmark", "backtest_curve.png")
    else:
        print("  No backtest data generated.")

    print("\nGenerating competition signals (2026.6.1 - 2026.6.12)...")
    comp_signals = generate_competition_signals(panel, ensemble_models, feature_cols, cal_df, stock_pool, code_to_id)
    if len(comp_signals) > 0:
        comp_signals.to_csv(os.path.join(OUTPUT_DIR, "competition_signals.csv"), index=False, encoding="utf-8-sig")
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
        trades_df.to_csv(os.path.join(OUTPUT_DIR, "competition_daily_trades.csv"), index=False, encoding="utf-8-sig")
        print("\n" + trades_df.to_string(index=False))
    else:
        print("  No competition signals generated (check data availability for 2026.6).")

    torch.save({
        "model_state_dicts": [m.state_dict() for m in ensemble_models],
        "config": CONFIG,
        "feature_cols": feature_cols,
        "input_dim": input_dim,
        "code_to_id": code_to_id,
        "ensemble_seeds": ensemble_seeds,
    }, os.path.join(OUTPUT_DIR, "ensemble_checkpoint.pth"))

    print("\n" + "=" * 60)
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
