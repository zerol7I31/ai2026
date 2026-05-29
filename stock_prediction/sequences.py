import gc

import numpy as np
import pandas as pd
from tqdm import tqdm


def build_mlp_sequences(panel, feature_cols, stock_list, config):
    feature_norm_cols = [c + "_norm" for c in feature_cols]
    all_sequences = []
    seq_len = config["seq_len"]

    for code in tqdm(stock_list, desc="Building MLP sequences"):
        stock_data = panel[panel["ts_code"] == code].sort_values("trade_date")
        if len(stock_data) < seq_len + 10:
            continue
        feat_array = stock_data[feature_norm_cols].values.astype(np.float32)
        label_array = stock_data["label"].values.astype(np.float32)
        date_array = stock_data["trade_date"].values

        for i in range(len(stock_data) - seq_len):
            window = feat_array[i : i + seq_len]
            y = label_array[i + seq_len - 1]
            if np.any(np.isnan(window)) or np.isnan(y):
                continue
            all_sequences.append({
                "ts_code": code,
                "date": date_array[i + seq_len - 1],
                "features": window.flatten(),
                "label": float(y),
            })

    return all_sequences


def build_gru_sequences(panel, feature_cols, stock_list, config):
    feature_norm_cols = [c + "_norm" for c in feature_cols]
    all_sequences = []
    seq_len = config["seq_len"]

    for code in tqdm(stock_list, desc="Building sequences"):
        stock_data = panel[panel["ts_code"] == code].sort_values("trade_date")
        if len(stock_data) < seq_len + 10:
            continue
        feat_array = stock_data[feature_norm_cols].values.astype(np.float32)
        label_array = stock_data["label_5d"].values.astype(np.float32)
        date_array = stock_data["trade_date"].values

        for i in range(len(stock_data) - seq_len):
            x = feat_array[i : i + seq_len]
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


def select_features_rf(train_sequences, n_top=30):
    if not train_sequences:
        return []
    n_features = len(train_sequences[0]["features"])
    if len(train_sequences) < 100:
        return list(range(n_features))

    from sklearn.ensemble import RandomForestRegressor

    x = np.stack([s["features"] for s in train_sequences])
    y = np.array([s["label"] for s in train_sequences])
    n_top = min(n_top, n_features)
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(x, y)
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:n_top]
    print(f"  RF feature selection: {n_features} -> {n_top} features")
    print(f"  Top-10 feature indices: {top_idx[:10].tolist()}")
    return top_idx.tolist()


def filter_features_by_idx(sequences, selected_idx):
    for s in sequences:
        s["features"] = s["features"][selected_idx]
    return sequences


def apply_cross_sectional_rank(sequences, enabled=True):
    if not enabled:
        return sequences

    df = pd.DataFrame([{"idx": i, "date": s["date"], "label": s["label"]} for i, s in enumerate(sequences)])
    df["date"] = pd.to_datetime(df["date"])
    df["cs_rank"] = np.nan

    for _, group in tqdm(df.groupby("date"), desc="Cross-sectional ranking"):
        if len(group) < 5:
            continue
        ranks = group["label"].rank(pct=True)
        df.loc[group.index, "cs_rank"] = ranks.values * 2 - 1

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Applying CS rank labels"):
        if not np.isnan(row["cs_rank"]):
            sequences[int(row["idx"])]["label"] = float(row["cs_rank"])

    gc.collect()
    return sequences

