import gc

import numpy as np
import pandas as pd
from tqdm import tqdm


def build_mlp_sequences(panel, feature_cols, stock_list, config):
    feature_norm_cols = [c + "_norm" for c in feature_cols]
    all_sequences = []
    seq_len = config["seq_len"]
    next_trade_dates = _build_future_trade_date_map(panel, 1)
    label_end_dates_by_signal = _build_future_trade_date_map(panel, config["pred_horizon"] + 1)

    for code in tqdm(stock_list, desc="Building MLP sequences"):
        stock_data = panel[panel["ts_code"] == code].sort_values("trade_date")
        if len(stock_data) < seq_len + 10:
            continue
        feat_array = stock_data[feature_norm_cols].values.astype(np.float32)
        label_array = stock_data["label"].values.astype(np.float32)
        date_array = stock_data["trade_date"].values
        label_start_dates = stock_data["label_start_date"].values
        label_end_dates = stock_data["label_end_date"].values

        for i in range(len(stock_data) - seq_len):
            label_idx = i + seq_len - 1
            window = feat_array[i : i + seq_len]
            y = label_array[label_idx]
            signal_date = _as_timestamp(date_array[label_idx])
            trade_date = _as_timestamp(label_start_dates[label_idx])
            label_end_date = _as_timestamp(label_end_dates[label_idx])
            if np.any(np.isnan(window)) or np.isnan(y):
                continue
            if pd.isna(trade_date) or pd.isna(label_end_date):
                continue
            if next_trade_dates.get(signal_date) != trade_date:
                continue
            if label_end_dates_by_signal.get(signal_date) != label_end_date:
                continue
            all_sequences.append({
                "ts_code": code,
                "date": signal_date,
                "trade_date": trade_date,
                "label_end_date": label_end_date,
                "features": window.flatten(),
                "label": float(y),
                "raw_label": float(y),
            })

    return all_sequences


def build_gru_sequences(panel, feature_cols, stock_list, config):
    feature_norm_cols = [c + "_norm" for c in feature_cols]
    all_sequences = []
    seq_len = config["seq_len"]
    next_trade_dates = _build_future_trade_date_map(panel, 1)
    label_end_dates_by_signal = _build_future_trade_date_map(panel, config["pred_horizon"] + 1)

    for code in tqdm(stock_list, desc="Building sequences"):
        stock_data = panel[panel["ts_code"] == code].sort_values("trade_date")
        if len(stock_data) < seq_len + 10:
            continue
        feat_array = stock_data[feature_norm_cols].values.astype(np.float32)
        label_array = stock_data["label_5d"].values.astype(np.float32)
        date_array = stock_data["trade_date"].values
        label_start_dates = stock_data["label_start_date"].values
        label_end_dates = stock_data["label_end_date"].values

        for i in range(len(stock_data) - seq_len):
            label_idx = i + seq_len - 1
            x = feat_array[i : i + seq_len]
            y = label_array[label_idx]
            signal_date = _as_timestamp(date_array[label_idx])
            trade_date = _as_timestamp(label_start_dates[label_idx])
            label_end_date = _as_timestamp(label_end_dates[label_idx])
            if np.any(np.isnan(x)) or np.isnan(y):
                continue
            if pd.isna(trade_date) or pd.isna(label_end_date):
                continue
            if next_trade_dates.get(signal_date) != trade_date:
                continue
            if label_end_dates_by_signal.get(signal_date) != label_end_date:
                continue
            all_sequences.append({
                "ts_code": code,
                "date": signal_date,
                "trade_date": trade_date,
                "label_end_date": label_end_date,
                "features": x,
                "label": float(y),
                "raw_label": float(y),
            })

    return all_sequences


def _build_future_trade_date_map(panel, offset):
    dates = pd.DatetimeIndex(pd.to_datetime(panel["trade_date"].dropna().unique())).sort_values()
    if len(dates) <= offset:
        return {}
    return dict(zip(dates[:-offset], dates[offset:]))


def _as_timestamp(value):
    if pd.isna(value):
        return pd.NaT
    return pd.Timestamp(value)


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


def select_features_corr(train_sequences, n_top=30):
    if not train_sequences:
        return []
    n_features = len(train_sequences[0]["features"])
    n_top = min(n_top, n_features)
    if len(train_sequences) < 100:
        return list(range(n_features))

    x = np.stack([s["features"] for s in train_sequences]).astype(np.float32, copy=False)
    y = np.array([s["label"] for s in train_sequences], dtype=np.float32)

    n = len(y)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    y_mean = y.mean()
    y_std = y.std()

    if y_std < 1e-12:
        print("  Correlation feature selection skipped: labels have near-zero variance")
        return list(range(n_top))

    cov = (x.T @ y) / max(n, 1) - x_mean * y_mean
    scores = np.abs(cov / (x_std * y_std + 1e-12))
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    top_idx = np.argsort(scores)[::-1][:n_top]
    print(f"  Correlation feature selection: {n_features} -> {n_top} features")
    print(f"  Top-10 feature indices: {top_idx[:10].tolist()}")
    return top_idx.tolist()


def select_mlp_features(train_sequences, n_top=30, method="corr"):
    if method == "none":
        if not train_sequences:
            return []
        return list(range(len(train_sequences[0]["features"])))
    if method == "rf":
        return select_features_rf(train_sequences, n_top=n_top)
    if method == "corr":
        return select_features_corr(train_sequences, n_top=n_top)
    raise ValueError(f"Unsupported MLP feature selection method: {method}")


def filter_features_by_idx(sequences, selected_idx):
    for s in sequences:
        s["features"] = s["features"][selected_idx]
    return sequences


def filter_sequences_by_trade_period(sequences, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered = []

    for sequence in sequences:
        trade_date = pd.to_datetime(sequence.get("trade_date", sequence["date"]))
        label_end_date = pd.to_datetime(sequence.get("label_end_date", trade_date))
        if pd.isna(trade_date) or pd.isna(label_end_date):
            continue
        if start_date <= trade_date <= end_date and label_end_date <= end_date:
            filtered.append(sequence)

    return filtered


def apply_cross_sectional_rank(sequences, enabled=True):
    if not enabled:
        return sequences

    for sequence in sequences:
        sequence.setdefault("raw_label", float(sequence["label"]))

    df = pd.DataFrame([
        {
            "idx": i,
            "rank_date": s.get("trade_date", s["date"]),
            "label": s["raw_label"],
        }
        for i, s in enumerate(sequences)
    ])
    df["rank_date"] = pd.to_datetime(df["rank_date"])
    df["cs_rank"] = np.nan

    for _, group in tqdm(df.groupby("rank_date"), desc="Cross-sectional ranking"):
        if len(group) < 5:
            continue
        ranks = group["label"].rank(pct=True)
        df.loc[group.index, "cs_rank"] = ranks.values * 2 - 1

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Applying CS rank labels"):
        if not np.isnan(row["cs_rank"]):
            sequences[int(row["idx"])]["label"] = float(row["cs_rank"])

    gc.collect()
    return sequences
