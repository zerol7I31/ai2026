import gc
import os
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from .settings import DATA_BASE, DAILY_DTYPE, DAILY_USECOLS


def load_basic_info(data_base=DATA_BASE):
    path = os.path.join(data_base, "basic.csv")
    df = pd.read_csv(path, dtype={"ts_code": str, "market": "category", "list_date": str})
    df["list_date"] = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
    return df


def load_trade_calendar(data_base=DATA_BASE):
    path = os.path.join(data_base, "trade_cal.csv")
    df = pd.read_csv(path, dtype={"cal_date": str, "is_open": str, "pretrade_date": str})
    sse = df[df["exchange"] == "SSE"].copy()
    sse["cal_date"] = pd.to_datetime(sse["cal_date"], format="%Y%m%d")
    sse = sse[sse["is_open"] == "1"].sort_values("cal_date")
    return sse


def load_st_stocks(data_base=DATA_BASE):
    st_dir = os.path.join(data_base, "stock_st")
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


def load_and_clean_daily(basic_df, st_set_by_date, data_base=DATA_BASE):
    daily_dir = os.path.join(data_base, "daily")
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
    for _, grp in panel.groupby("ts_code", observed=True, sort=False):
        for col in outlier_cols:
            vals = grp[col].values.astype("float64")
            mu = vals.mean()
            sigma = vals.std()
            if sigma < 1e-9:
                continue
            z = abs(vals - mu) / sigma
            idx = grp.index[z > 3]
            panel.loc[idx, "_keep"] = False
    n_before = len(panel)
    panel = panel[panel["_keep"]].copy()
    n_after = len(panel)
    print(f"  3-sigma outlier filter: {n_before:,} -> {n_after:,} rows ({n_before - n_after:,} removed)")
    panel.drop(columns=["_keep"], inplace=True)

    gc.collect()
    return panel


def select_stock_pool(panel, config):
    if config["stock_pool"] == "all":
        stock_counts = panel.groupby("ts_code", observed=True).size()
        valid_stocks = stock_counts[stock_counts >= config["seq_len"] + 50].index.tolist()
        return valid_stocks[: config["max_stocks"]]

    pool_start_dt = pd.to_datetime(config["train_start"]) - timedelta(days=90)
    pool_end_dt = pd.to_datetime(config["test_end"])

    sub = panel[(panel["trade_date"] >= pool_start_dt) & (panel["trade_date"] <= pool_end_dt)].copy()
    if len(sub) == 0:
        return panel["ts_code"].unique().tolist()[: config["max_stocks"]]

    sub["amount_rank"] = sub.groupby("trade_date", observed=True)["amount"].rank(ascending=False)
    stock_avg_rank = sub.groupby("ts_code", observed=True)["amount_rank"].mean()
    sub.drop(columns=["amount_rank"], inplace=True)

    sorted_stocks = stock_avg_rank.sort_values().index.tolist()
    del sub
    gc.collect()
    return sorted_stocks[: config["max_stocks"]]


def load_benchmark_data(data_base=DATA_BASE):
    market_dir = os.path.join(data_base, "market")
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

