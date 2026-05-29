import gc

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_features(
    group_df,
    pred_horizon,
    label_col,
    include_ma_5_10=False,
    include_enhanced=False,
):
    df = group_df.sort_values("trade_date").copy()
    for col in ["open", "close", "high", "low", "vol", "amount", "pct_chg", "vwap"]:
        df[col] = df[col].astype("float32")

    ret = df["close"].pct_change().astype("float32")

    df["ret_1"] = ret
    df["ret_5"] = (df["close"] / df["close"].shift(5) - 1).astype("float32")
    df["ret_10"] = (df["close"] / df["close"].shift(10) - 1).astype("float32")
    df["ret_20"] = (df["close"] / df["close"].shift(20) - 1).astype("float32")

    df["vol_5"] = ret.rolling(5, min_periods=3).std(ddof=0).astype("float32")
    df["vol_10"] = ret.rolling(10, min_periods=5).std(ddof=0).astype("float32")
    df["vol_20"] = ret.rolling(20, min_periods=10).std(ddof=0).astype("float32")

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
    if include_ma_5_10:
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

    feature_cols = [
        "ret_1",
        "ret_5",
        "ret_10",
        "ret_20",
        "vol_5",
        "vol_10",
        "vol_20",
        "vol_chg_5",
        "vol_chg_10",
        "vol_ratio_5",
        "pv_corr",
    ]
    if include_ma_5_10:
        feature_cols.append("ma_ratio_5_10")
    feature_cols.extend([
        "ma_ratio_5_20",
        "ma_ratio_5_60",
        "ma_ratio_20_60",
        "macd_dif",
        "macd_dea",
        "macd_hist",
        "rsi_14",
        "boll_width",
        "boll_pct",
        "atr_pct",
        "slow_k",
        "slow_d",
        "clv",
        "hl_ratio",
        "oc_ratio",
        "vwap_diff",
        "price_pos_40",
        "amplitude",
        "turnover",
        "pct_chg_diff",
        "vol_diff",
        "amount_diff",
    ])

    if include_enhanced:
        _add_enhanced_features(df, ret)
        feature_cols.extend([
            "overnight_ret",
            "intraday_ret",
            "downside_vol_5",
            "upside_vol_5",
            "mom_rev_5_20",
            "vol_ma_ratio",
            "price_vol_div_10",
            "max_ret_5",
            "min_ret_5",
            "ret_skew_20",
            "ret_kurt_20",
        ])

    fwd_close = df["close"].shift(-pred_horizon)
    df[label_col] = (fwd_close / df["close"] - 1).astype("float32")
    return df, feature_cols


def compute_mlp_features(group_df, config):
    return compute_features(
        group_df,
        pred_horizon=config["pred_horizon"],
        label_col="label",
        include_ma_5_10=True,
        include_enhanced=False,
    )


def compute_gru_features(group_df, config):
    return compute_features(
        group_df,
        pred_horizon=config["pred_horizon"],
        label_col="label_5d",
        include_ma_5_10=False,
        include_enhanced=True,
    )


def _add_enhanced_features(df, ret):
    df["overnight_ret"] = ((df["open"] / df["close"].shift(1).clip(lower=1e-9)) - 1).astype("float32")
    df["intraday_ret"] = ((df["close"] / df["open"].clip(lower=1e-9)) - 1).astype("float32")

    ret_5_neg = ret.rolling(5, min_periods=3).apply(
        lambda x: -np.sum(x[x < 0]) if np.any(x < 0) else 0,
        raw=True,
    ).astype("float32")
    ret_5_pos = ret.rolling(5, min_periods=3).apply(
        lambda x: np.sum(x[x > 0]) if np.any(x > 0) else 0,
        raw=True,
    ).astype("float32")
    df["downside_vol_5"] = ret_5_neg
    df["upside_vol_5"] = ret_5_pos

    ret_20_sum = ret.rolling(20, min_periods=10).sum().astype("float32")
    ret_5_sum = ret.rolling(5, min_periods=3).sum().astype("float32")
    df["mom_rev_5_20"] = (ret_5_sum - ret_20_sum).astype("float32")

    vol_ma5 = df["vol"].rolling(5, min_periods=3).mean().astype("float32")
    vol_ma20 = df["vol"].rolling(20, min_periods=10).mean().astype("float32")
    df["vol_ma_ratio"] = ((vol_ma5 / vol_ma20.clip(lower=1e-9)) - 1).astype("float32")

    ret_10_sum = ret.rolling(10, min_periods=5).sum().astype("float32")
    vol_10_sum = df["vol"].rolling(10, min_periods=5).sum().astype("float32")
    vol_10_mean = df["vol"].rolling(10, min_periods=5).mean().values
    df["price_vol_div_10"] = (
        np.sign(ret_10_sum.values) * np.sign(vol_10_sum.values - vol_10_mean) * (-1)
    ).astype("float32")

    df["max_ret_5"] = ret.rolling(5, min_periods=3).max().astype("float32")
    df["min_ret_5"] = ret.rolling(5, min_periods=3).min().astype("float32")
    df["ret_skew_20"] = ret.rolling(20, min_periods=10).skew().astype("float32")
    df["ret_kurt_20"] = ret.rolling(20, min_periods=10).kurt().astype("float32")


def build_feature_panel(panel, feature_fn):
    grouped = panel.groupby("ts_code", observed=True, group_keys=False)
    feature_dfs = []
    feature_cols = None
    for _, group in tqdm(grouped, desc="Feature engineering"):
        fdf, fcols = feature_fn(group)
        feature_dfs.append(fdf)
        if feature_cols is None:
            feature_cols = fcols
        del group
    result = pd.concat(feature_dfs, ignore_index=True)
    del feature_dfs
    gc.collect()
    return result, feature_cols


def normalize_features(panel, feature_cols, mode="rolling_252"):
    grouped = panel.groupby("ts_code", observed=True)
    normalized_dfs = []
    desc = "Normalizing features (rolling expanding)" if mode == "expanding" else "Normalizing features (rolling)"
    for _, group in tqdm(grouped, desc=desc):
        group = group.sort_values("trade_date").copy()
        for col in feature_cols:
            if col not in group.columns:
                continue
            vals = group[col].values.astype(np.float32)
            s = pd.Series(vals)
            if mode == "expanding":
                rolling_mean = s.expanding(min_periods=63).mean().shift(1)
                rolling_std = s.expanding(min_periods=63).std(ddof=0).shift(1)
            else:
                rolling_mean = s.rolling(252, min_periods=63).mean().shift(1)
                rolling_std = s.rolling(252, min_periods=63).std(ddof=0).shift(1)
            group[col + "_norm"] = ((vals - rolling_mean.values) / (rolling_std.values + 1e-9)).astype(np.float32)
        normalized_dfs.append(group)
        del group
    result = pd.concat(normalized_dfs, ignore_index=True)
    del normalized_dfs
    gc.collect()
    return result
