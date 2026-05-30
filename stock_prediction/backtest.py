import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm


def build_close_prices_pivot(panel, stock_pool):
    sub = panel[panel["ts_code"].isin(stock_pool)].copy()
    close_piv = sub.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last")
    return close_piv.sort_index()


def build_price_pivots(panel, stock_pool):
    sub = panel[panel["ts_code"].isin(stock_pool)].copy()
    close_piv = sub.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last")
    open_piv = sub.pivot_table(index="trade_date", columns="ts_code", values="open", aggfunc="first")
    return close_piv.sort_index(), open_piv.sort_index()


def backtest_strategy(prices_pivot, predictions_df, config, open_pivot=None):
    cash = config["initial_capital"]
    holdings = {}
    daily_values = []

    bt_start = pd.to_datetime(config["test_start"])
    bt_end = pd.to_datetime(config["test_end"])
    all_dates = sorted(prices_pivot.index[(prices_pivot.index >= bt_start) & (prices_pivot.index <= bt_end)])

    if len(all_dates) == 0:
        return [], []

    top_n = config["top_n_hold"]
    trade_n = config["daily_trade_n"]
    comm = config["commission_rate"]
    slip = config["slippage"]
    min_hold_tdays = 5

    stock_universe = set(prices_pivot.columns)

    for idx_t, date_val in enumerate(tqdm(all_dates, desc="Backtest")):
        if date_val not in prices_pivot.index:
            continue
        close_row = prices_pivot.loc[date_val]
        use_open_price = open_pivot is not None and date_val in open_pivot.index
        trade_row = close_row
        if use_open_price:
            trade_row = open_pivot.loc[date_val]

        scores = pd.Series(dtype=float)
        if date_val in predictions_df.index:
            row = predictions_df.loc[date_val].dropna()
            if len(row) > 0:
                scores = row.sort_values(ascending=False)
                scores = scores[scores.index.isin(stock_universe)]

        if scores.empty:
            for code in list(holdings.keys()):
                if code in close_row.index and pd.notna(close_row[code]):
                    holdings[code]["current_price"] = float(close_row[code])
            total_value = cash + sum(
                holding["shares"] * holding.get("current_price", holding["buy_price"])
                for holding in holdings.values()
            )
            daily_values.append({"date": date_val, "value": total_value, "cash": cash})
            continue

        current_held = set(holdings.keys())
        target_stocks = set(scores.head(top_n).index.tolist())

        to_sell_candidates = list(current_held - target_stocks)
        to_sell = [
            code
            for code in to_sell_candidates
            if idx_t - holdings[code].get("bought_at_tidx", -999) > min_hold_tdays
        ]
        if len(to_sell) > trade_n:
            sell_scores = {stock: scores.get(stock, -999) for stock in to_sell}
            to_sell = sorted(to_sell, key=lambda x: sell_scores.get(x, -999))[:trade_n]

        for code in to_sell:
            if code not in holdings:
                continue
            holding = holdings[code]
            if code in trade_row.index and pd.notna(trade_row[code]):
                price = float(trade_row[code])
            else:
                price = holding.get("current_price", holding["buy_price"])
            if not np.isfinite(price) or price <= 0:
                continue
            sell_value = holding["shares"] * price * (1 - slip) * (1 - comm)
            cash += sell_value
            del holdings[code]

        to_buy = list(target_stocks - current_held)
        if len(to_buy) > trade_n:
            buy_scores = {code: scores.get(code, 999) for code in to_buy}
            to_buy = sorted(to_buy, key=lambda x: buy_scores.get(x, 999), reverse=True)[:trade_n]

        open_slots = top_n - len(holdings)
        to_buy = to_buy[:open_slots] if open_slots > 0 else []

        if to_buy and cash > 0 and trade_row is not None:
            buy_scores_raw = np.array([scores.get(code, 0.0) for code in to_buy], dtype=np.float64)
            if len(buy_scores_raw) > 1:
                buy_scores_centered = buy_scores_raw - buy_scores_raw.mean()
                buy_weights = np.exp(np.clip(buy_scores_centered * 2.0, -10, 10))
                buy_weights = buy_weights / buy_weights.sum()
            else:
                buy_weights = np.ones(1)

            available_cash = cash
            for j, code in enumerate(to_buy):
                if code not in trade_row.index or pd.isna(trade_row[code]):
                    continue
                bp = float(trade_row[code]) * (1 + slip)
                if not np.isfinite(bp) or bp <= 0:
                    continue
                alloc = available_cash * float(buy_weights[j])
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
                current_price = bp
                if code in close_row.index and pd.notna(close_row[code]):
                    current_price = float(close_row[code])
                holdings[code] = {
                    "shares": shares,
                    "buy_price": bp,
                    "current_price": current_price,
                    "bought_at_tidx": idx_t,
                }

        for code in list(holdings.keys()):
            if code in close_row.index and pd.notna(close_row[code]):
                holdings[code]["current_price"] = float(close_row[code])

        total_value = cash + sum(
            holding["shares"] * holding.get("current_price", holding["buy_price"])
            for holding in holdings.values()
        )
        daily_values.append({"date": date_val, "value": total_value, "cash": cash})

    return daily_values, all_dates


def compute_backtest_metrics(daily_values_df, initial_capital=None):
    if len(daily_values_df) == 0:
        return {}
    df = daily_values_df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["returns"] = df["value"].pct_change()

    base_value = float(initial_capital) if initial_capital is not None else df["value"].iloc[0]
    total_return = df["value"].iloc[-1] / base_value - 1

    first_date = df["date"].iloc[0]
    last_date = df["date"].iloc[-1]
    calendar_days = (last_date - first_date).days
    years = max(calendar_days, 1) / 365.25
    annual_return = (1 + total_return) ** (1.0 / max(years, 1e-6)) - 1

    daily_rf = 0.03 / 252
    returns = df["returns"].dropna()
    excess = returns.values - daily_rf
    ann_excess = excess.mean() * 252
    ann_vol = excess.std() * np.sqrt(252)
    sharpe = ann_excess / (ann_vol + 1e-9)

    cummax = df["value"].cummax()
    drawdown = (df["value"] - cummax) / cummax
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).mean()

    return {
        "总收益率": f"{total_return * 100:.2f}%",
        "年化收益率": f"{annual_return * 100:.2f}%",
        "夏普比率": f"{sharpe:.3f}",
        "最大回撤": f"{max_drawdown * 100:.2f}%",
        "胜率": f"{win_rate * 100:.2f}%",
    }


def plot_backtest_curve(
    daily_values_df,
    benchmark_values,
    title,
    filename,
    output_dir,
    strategy_label="Strategy",
    initial_capital=None,
):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    base_value = float(initial_capital) if initial_capital is not None else daily_values_df["value"].values[0]
    strategy_vals = daily_values_df["value"].values / base_value
    ax1.plot(daily_values_df["date"], strategy_vals, label=strategy_label, color="blue", linewidth=1.5)

    if len(benchmark_values) > 0:
        common_dates_bench = set(benchmark_values["date"].values)
        common_dates_strat = set(daily_values_df["date"].values)
        common = sorted(common_dates_bench & common_dates_strat)
        if len(common) > 1:
            strat_aligned = daily_values_df[daily_values_df["date"].isin(common)].sort_values("date")
            bench_aligned = benchmark_values[benchmark_values["date"].isin(common)].sort_values("date")
            bench_norm = bench_aligned["value"].values / bench_aligned["value"].values[0]
            ax1.plot(
                strat_aligned["date"],
                bench_norm,
                label="Benchmark (CSI300)",
                color="orange",
                linewidth=1.5,
                alpha=0.7,
            )

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
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()


def align_benchmark_to_backtest(benchmark, bt_df):
    bench_daily = []
    if len(benchmark) > 0:
        for _, row in bt_df.iterrows():
            b_row = benchmark[benchmark["trade_date"] == row["date"]]
            val = float(b_row["close"].values[0]) if len(b_row) > 0 else None
            bench_daily.append({"date": row["date"], "value": val})
        bench_daily = [item for item in bench_daily if item["value"] is not None]
    return pd.DataFrame(bench_daily) if bench_daily else pd.DataFrame()
