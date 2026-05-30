import numpy as np
import pandas as pd
import torch


def generate_mlp_competition_signals(panel, model, feature_cols, cal_df, stock_pool, selected_idx, config):
    device = config["device"]
    model = model.to(device)
    model.eval()

    feature_norm_cols = [c + "_norm" for c in feature_cols]
    seq_len = config["seq_len"]
    top_n = config["top_n_hold"]
    trade_n = config["daily_trade_n"]

    trade_schedule = _competition_dates(panel, cal_df, config)
    if len(trade_schedule) == 0:
        print("Warning: No trade dates in competition period!")
        return pd.DataFrame()

    current_positions = set()
    daily_signals = []

    for _, cal_row in trade_schedule.iterrows():
        date_val = cal_row["cal_date"]
        cutoff = cal_row["cutoff_date"]

        batches = []
        valid_codes = []
        for code in stock_pool:
            code_data = panel[(panel["ts_code"] == code) & (panel["trade_date"] <= cutoff)].sort_values("trade_date")
            if len(code_data) < seq_len + 5:
                continue
            if code_data["trade_date"].iloc[-1] != cutoff:
                continue
            window = code_data[feature_norm_cols].tail(seq_len).values.astype(np.float32)
            if np.any(np.isnan(window)):
                continue
            flat = window.flatten()[selected_idx]
            batches.append(flat)
            valid_codes.append(code)

        if not batches:
            continue

        stacked = np.stack(batches)
        tensor_x = torch.tensor(stacked, dtype=torch.float32).to(device)
        scores_list = []
        batch_size = config["batch_size"]
        for start in range(0, len(stacked), batch_size):
            end = min(start + batch_size, len(stacked))
            with torch.no_grad():
                batch_scores = model(tensor_x[start:end]).cpu().numpy()
            scores_list.append(batch_scores)
        all_scores = np.concatenate(scores_list)

        df_day, current_positions = _build_signal_day(
            date_val,
            valid_codes,
            all_scores,
            current_positions,
            top_n,
            trade_n,
            cutoff,
            cal_row.get("stale_cutoff", False),
        )
        if len(df_day) > 0:
            daily_signals.append(df_day)

    return pd.concat(daily_signals, ignore_index=True) if daily_signals else pd.DataFrame()


def generate_gru_competition_signals(panel, ensemble_models, feature_cols, cal_df, stock_pool, code_to_id, config):
    device = config["device"]
    for model in ensemble_models:
        model.to(device)
        model.eval()

    feature_norm_cols = [c + "_norm" for c in feature_cols]
    seq_len = config["seq_len"]
    top_n = config["top_n_hold"]
    trade_n = config["daily_trade_n"]

    trade_schedule = _competition_dates(panel, cal_df, config)
    if len(trade_schedule) == 0:
        print("Warning: No trade dates in competition period!")
        return pd.DataFrame()

    current_positions = set()
    daily_signals = []

    for _, cal_row in trade_schedule.iterrows():
        date_val = cal_row["cal_date"]
        cutoff = cal_row["cutoff_date"]

        batches = []
        valid_codes = []
        valid_ids = []
        for code in stock_pool:
            code_data = panel[(panel["ts_code"] == code) & (panel["trade_date"] <= cutoff)].sort_values("trade_date")
            if len(code_data) < seq_len + 5:
                continue
            if code_data["trade_date"].iloc[-1] != cutoff:
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
        batch_size = config["batch_size"]
        for start in range(0, len(stacked), batch_size):
            end = min(start + batch_size, len(stacked))
            with torch.no_grad():
                batch_preds = np.zeros(end - start, dtype=np.float64)
                for model in ensemble_models:
                    batch_preds += model(tensor_x[start:end], tensor_ids[start:end]).cpu().numpy().astype(np.float64)
                batch_preds /= len(ensemble_models)
            scores_list.append(batch_preds)
        all_scores = np.concatenate(scores_list)

        df_day, current_positions = _build_signal_day(
            date_val,
            valid_codes,
            all_scores,
            current_positions,
            top_n,
            trade_n,
            cutoff,
            cal_row.get("stale_cutoff", False),
        )
        if len(df_day) > 0:
            daily_signals.append(df_day)

    return pd.concat(daily_signals, ignore_index=True) if daily_signals else pd.DataFrame()


def summarize_daily_trades(comp_signals):
    daily_trades = []
    for date_val in sorted(comp_signals["date"].unique()):
        day_df = comp_signals[comp_signals["date"] == date_val]
        buy_list = day_df[day_df["action"] == "BUY"]["ts_code"].tolist()
        hold_list = day_df[day_df["action"] == "HOLD"]["ts_code"].tolist()
        sell_list = day_df[day_df["action"] == "SELL"]["ts_code"].tolist()
        daily_trades.append({
            "日期": date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val),
            "信号数据截止日": _format_date(day_df["cutoff_date"].iloc[0]) if "cutoff_date" in day_df else "",
            "BUY": ",".join(buy_list),
            "SELL": ",".join(sell_list),
            "HOLD": ",".join(hold_list),
        })
    return pd.DataFrame(daily_trades)


def print_competition_summary(comp_signals):
    print("\nCompetition Signals Summary:")
    for date_val in sorted(comp_signals["date"].unique()):
        day_df = comp_signals[comp_signals["date"] == date_val].sort_values("rank")
        date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)
        cutoff_str = ""
        if "cutoff_date" in day_df:
            cutoff_str = f" | cutoff={_format_date(day_df['cutoff_date'].iloc[0])}"
        print(f"\n  {date_str}{cutoff_str}:")
        for _, row in day_df.iterrows():
            print(
                f"    {row['ts_code']} | Score: {row['score']:.4f} | "
                f"Rank: {int(row['rank'])} | Action: {row['action']}"
            )


def _competition_dates(panel, cal_df, config):
    comp_start = pd.to_datetime(config["competition_start"])
    comp_end = pd.to_datetime(config["competition_end"])
    trade_schedule = cal_df[
        (cal_df["cal_date"] >= comp_start) & (cal_df["cal_date"] <= comp_end)
    ].sort_values("cal_date").copy()
    if len(trade_schedule) == 0:
        return trade_schedule

    available_dates = pd.DatetimeIndex(pd.to_datetime(panel["trade_date"].dropna().unique())).sort_values()
    if len(available_dates) == 0:
        trade_schedule["cutoff_date"] = pd.NaT
        return trade_schedule.iloc[0:0]

    cutoff_dates = []
    stale_dates = []
    cal_open_dates = pd.DatetimeIndex(pd.to_datetime(cal_df["cal_date"].dropna().unique())).sort_values()

    for trade_date in trade_schedule["cal_date"]:
        pos = available_dates.searchsorted(trade_date, side="left") - 1
        cutoff = available_dates[pos] if pos >= 0 else pd.NaT
        cutoff_dates.append(cutoff)

        cal_pos = cal_open_dates.searchsorted(trade_date, side="left") - 1
        prev_trade_date = cal_open_dates[cal_pos] if cal_pos >= 0 else pd.NaT
        stale_dates.append(pd.notna(cutoff) and pd.notna(prev_trade_date) and cutoff < prev_trade_date)

    trade_schedule["cutoff_date"] = cutoff_dates
    trade_schedule["stale_cutoff"] = stale_dates
    trade_schedule = trade_schedule.dropna(subset=["cutoff_date"])

    if trade_schedule["stale_cutoff"].any():
        stale_rows = trade_schedule[trade_schedule["stale_cutoff"]]
        first = stale_rows.iloc[0]
        print(
            "Warning: some competition signals use stale data because newer daily files are not available. "
            f"First stale signal: {first['cal_date'].date()} uses cutoff {first['cutoff_date'].date()}."
        )

    return trade_schedule


def _build_signal_day(
    date_val,
    valid_codes,
    all_scores,
    current_positions,
    top_n,
    trade_n,
    cutoff_date=None,
    stale_cutoff=False,
):
    scores_df = pd.DataFrame({"ts_code": valid_codes, "score": all_scores})
    scores_df = scores_df.sort_values("score", ascending=False).reset_index(drop=True)
    scores_df["rank"] = range(1, len(scores_df) + 1)
    score_map = dict(zip(scores_df["ts_code"], scores_df["score"]))
    rank_map = dict(zip(scores_df["ts_code"], scores_df["rank"]))

    if not current_positions:
        sell_set = set()
        buy_set = set(scores_df.head(top_n)["ts_code"].tolist())
        next_positions = buy_set.copy()
    else:
        target_top = set(scores_df.head(top_n)["ts_code"].tolist())
        sell_candidates = list(current_positions - target_top)
        sell_candidates = sorted(sell_candidates, key=lambda code: score_map.get(code, -np.inf))
        sell_set = set(sell_candidates[:trade_n])

        positions_after_sell = current_positions - sell_set
        open_slots = max(top_n - len(positions_after_sell), 0)
        buy_limit = min(trade_n, open_slots)
        buy_set = set()
        if buy_limit > 0:
            for code in scores_df["ts_code"]:
                if code not in positions_after_sell:
                    buy_set.add(code)
                    if len(buy_set) >= buy_limit:
                        break
        next_positions = positions_after_sell | buy_set

    action_rows = []
    relevant_codes = next_positions | sell_set
    for code in sorted(relevant_codes):
        if code in buy_set:
            action = "BUY"
        elif code in sell_set:
            action = "SELL"
        elif code in next_positions:
            action = "HOLD"
        else:
            action = "NONE"

        action_rows.append({
            "date": date_val,
            "cutoff_date": cutoff_date,
            "stale_cutoff": bool(stale_cutoff),
            "ts_code": code,
            "score": score_map.get(code, np.nan),
            "rank": rank_map.get(code, 999),
            "action": action,
        })

    df_day = pd.DataFrame(action_rows)
    if len(df_day) > 0:
        df_day = df_day.sort_values("rank")
    return df_day, next_positions.copy()


def _format_date(date_val):
    if pd.isna(date_val):
        return ""
    if hasattr(date_val, "strftime"):
        return date_val.strftime("%Y-%m-%d")
    return str(date_val)
