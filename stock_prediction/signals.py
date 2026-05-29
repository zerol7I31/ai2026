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

    trade_cal_dates, hist_dates = _competition_dates(panel, cal_df, config)
    if len(trade_cal_dates) == 0:
        print("Warning: No trade dates in competition period!")
        return pd.DataFrame()

    current_positions = set()
    daily_signals = []

    for comp_day_idx, (_, cal_row) in enumerate(trade_cal_dates.iterrows()):
        date_val = cal_row["cal_date"]
        cutoff = hist_dates[min(comp_day_idx, len(hist_dates) - 1)]

        batches = []
        valid_codes = []
        for code in stock_pool:
            code_data = panel[(panel["ts_code"] == code) & (panel["trade_date"] <= cutoff)].sort_values("trade_date")
            if len(code_data) < seq_len + 5:
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

    trade_cal_dates, hist_dates = _competition_dates(panel, cal_df, config)
    if len(trade_cal_dates) == 0:
        print("Warning: No trade dates in competition period!")
        return pd.DataFrame()

    current_positions = set()
    daily_signals = []

    for comp_day_idx, (_, cal_row) in enumerate(trade_cal_dates.iterrows()):
        date_val = cal_row["cal_date"]
        cutoff = hist_dates[min(comp_day_idx, len(hist_dates) - 1)]

        batches = []
        valid_codes = []
        valid_ids = []
        for code in stock_pool:
            code_data = panel[(panel["ts_code"] == code) & (panel["trade_date"] <= cutoff)].sort_values("trade_date")
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
        print(f"\n  {date_str}:")
        for _, row in day_df.iterrows():
            print(
                f"    {row['ts_code']} | Score: {row['score']:.4f} | "
                f"Rank: {int(row['rank'])} | Action: {row['action']}"
            )


def _competition_dates(panel, cal_df, config):
    comp_start = pd.to_datetime(config["competition_start"])
    comp_end = pd.to_datetime(config["competition_end"])
    panel_max_date = panel["trade_date"].max()
    if panel_max_date < comp_start:
        print(
            f"Panel data ends at {panel_max_date.date()}, before competition start {comp_start.date()}. "
            f"Shifting cutoff date backward per competition day to simulate rolling updates."
        )

    trade_cal_dates = cal_df[(cal_df["cal_date"] >= comp_start) & (cal_df["cal_date"] <= comp_end)]
    historical_cal = cal_df[cal_df["cal_date"] <= panel_max_date].sort_values("cal_date", ascending=False)
    hist_dates = historical_cal["cal_date"].tolist()
    return trade_cal_dates, hist_dates


def _build_signal_day(date_val, valid_codes, all_scores, current_positions, top_n):
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
    return df_day, top_stocks_today.copy()

