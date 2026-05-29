import gc
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from stock_prediction.backtest import (
    align_benchmark_to_backtest,
    backtest_strategy,
    build_close_prices_pivot,
    compute_backtest_metrics,
    plot_backtest_curve,
)
from stock_prediction.data import (
    load_and_clean_daily,
    load_basic_info,
    load_benchmark_data,
    load_st_stocks,
    load_trade_calendar,
    select_stock_pool,
)
from stock_prediction.evaluation import (
    compute_direction_accuracy,
    compute_ic_ir,
    predict_mlp,
)
from stock_prediction.features import (
    build_feature_panel,
    compute_mlp_features,
    normalize_features,
)
from stock_prediction.models import MLPSequenceDataset, MLPStockPredictor
from stock_prediction.sequences import (
    build_mlp_sequences,
    filter_features_by_idx,
    select_features_rf,
)
from stock_prediction.settings import MLP_CONFIG, OUTPUT_DIR
from stock_prediction.signals import (
    generate_mlp_competition_signals,
    print_competition_summary,
    summarize_daily_trades,
)
from stock_prediction.training import train_mlp_model
from stock_prediction.utils import set_seed


def main(config=None):
    config = config or MLP_CONFIG
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")
    set_seed(42)

    print("=" * 60)
    print("MLP Baseline - Deep Learning Stock Prediction")
    print(f"Device: {config['device']} | Pred Horizon: {config['pred_horizon']}d")
    print(f"Model: MLP {config['hidden_dims']} | Seq Len: {config['seq_len']}")
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
    stock_pool = select_stock_pool(panel, config)
    print(f"  Selected {len(stock_pool)} stocks")
    panel = panel[panel["ts_code"].isin(stock_pool)].copy()
    panel["ts_code"] = panel["ts_code"].cat.remove_unused_categories()
    gc.collect()
    print(f"  Filtered panel: {len(panel):,} rows, {panel['ts_code'].nunique()} stocks")

    print("\n[4/9] Computing features per stock (float32)...")
    panel, feature_cols = build_feature_panel(
        panel,
        lambda group: compute_mlp_features(group, config),
    )
    print(f"  Feature columns ({len(feature_cols)} dims): {feature_cols}")

    print("\n[5/9] Normalizing features (expanding per-stock, no lookahead)...")
    panel = normalize_features(panel, feature_cols, mode="expanding")

    print("\n[6/9] Building MLP sequences (flatten window -> 1D vector)...")
    sequences = build_mlp_sequences(panel, feature_cols, stock_pool, config)
    print(f"  Total sequences: {len(sequences):,}")
    feat_dim = len(sequences[0]["features"]) if sequences else 0
    print(
        f"  Feature dimension (seq_len x n_features = "
        f"{config['seq_len']} x {len(feature_cols)} = {feat_dim})"
    )

    train_start = pd.to_datetime(config["train_start"])
    train_end = pd.to_datetime(config["train_end"])
    val_start = pd.to_datetime(config["val_start"])
    val_end = pd.to_datetime(config["val_end"])
    test_start = pd.to_datetime(config["test_start"])

    train_seq = [s for s in sequences if train_start <= s["date"] <= train_end]
    val_seq = [s for s in sequences if val_start <= s["date"] <= val_end]
    test_seq = [s for s in sequences if s["date"] >= test_start]
    print(f"  Train: {len(train_seq):,}, Val: {len(val_seq):,}, Test: {len(test_seq):,}")

    if len(train_seq) == 0:
        print("ERROR: No training sequences! Check date ranges.")
        sys.exit(1)

    print("\n  Random Forest feature selection...")
    selected_idx = select_features_rf(train_seq, n_top=min(100, feat_dim))

    train_seq = filter_features_by_idx(train_seq, selected_idx)
    val_seq = filter_features_by_idx(val_seq, selected_idx) if val_seq else []
    test_seq = filter_features_by_idx(test_seq, selected_idx) if test_seq else []

    train_dataset = MLPSequenceDataset(train_seq)
    val_dataset = MLPSequenceDataset(val_seq) if val_seq else None
    test_dataset = MLPSequenceDataset(test_seq) if test_seq else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
        if val_dataset
        else None
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
        if test_dataset
        else None
    )

    input_dim = train_dataset.features.shape[1]
    print(f"  MLP input dimension: {input_dim}")
    print(f"  Hidden layers: {config['hidden_dims']}")

    print("\n[7/9] Building and training MLP model...")
    model = MLPStockPredictor(
        input_dim=input_dim,
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    model_path = os.path.join(OUTPUT_DIR, "mlp_best_model.pth")
    if val_loader is not None:
        train_mlp_model(model, train_loader, val_loader, model_path, config, OUTPUT_DIR)
    else:
        print("  No validation set, training without validation...")
        train_mlp_model(model, train_loader, train_loader, model_path, config, OUTPUT_DIR)

    model.load_state_dict(torch.load(model_path, map_location=config["device"], weights_only=True))

    print("\n[8/9] Evaluating model...")
    device = config["device"]
    model.eval()

    val_preds, val_labels, val_dates = (
        predict_mlp(model, val_loader, val_seq, device)
        if val_seq
        else (np.array([]), np.array([]), [])
    )
    test_preds, test_labels, test_dates = (
        predict_mlp(model, test_loader, test_seq, device)
        if test_seq
        else (np.array([]), np.array([]), [])
    )

    eval_data = []
    for name, preds, labels, dates in [
        ("Validation", val_preds, val_labels, val_dates),
        ("Test", test_preds, test_labels, test_dates),
    ]:
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
        print(
            f"  {name}: PearsonIC={ic_mean:.4f}, ICIR={icir:.4f}, "
            f"RankIC={rank_ic:.4f}, RankICIR={rank_icir:.4f}, "
            f"DirAcc={dir_acc * 100:.2f}%"
        )

    eval_df = pd.DataFrame(eval_data)
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "mlp_evaluation_metrics.csv"), index=False, encoding="utf-8-sig")
    print("\n" + eval_df.to_string(index=False))

    print("\n[9/9] Running backtest...")
    all_predictions = [
        {
            "ts_code": sequence["ts_code"],
            "date": sequence["date"],
            "prediction": float(test_preds[i]),
        }
        for i, sequence in enumerate(test_seq)
        if len(test_preds) > 0
    ]
    pred_df = pd.DataFrame(all_predictions)
    predictions_pivot = (
        pred_df.pivot_table(index="date", columns="ts_code", values="prediction", aggfunc="mean")
        if len(pred_df) > 0
        else pd.DataFrame()
    )

    print("  Building price pivot for backtest...")
    prices_pivot = build_close_prices_pivot(panel, stock_pool)

    daily_values, _ = backtest_strategy(prices_pivot, predictions_pivot, config)
    if daily_values:
        bt_df = pd.DataFrame(daily_values).sort_values("date")
        bt_df.to_csv(os.path.join(OUTPUT_DIR, "mlp_backtest_daily_values.csv"), index=False, encoding="utf-8-sig")
        metrics = compute_backtest_metrics(bt_df)
        print("\nBacktest Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        benchmark = load_benchmark_data()
        bench_df = align_benchmark_to_backtest(benchmark, bt_df)
        plot_backtest_curve(
            bt_df,
            bench_df,
            "MLP Strategy vs CSI300 Benchmark",
            "mlp_backtest_curve.png",
            OUTPUT_DIR,
            strategy_label="MLP Strategy",
        )
    else:
        print("  No backtest data generated.")

    print("\nGenerating competition signals (2026.6.1 - 2026.6.12)...")
    comp_signals = generate_mlp_competition_signals(
        panel,
        model,
        feature_cols,
        cal_df,
        stock_pool,
        selected_idx,
        config,
    )
    if len(comp_signals) > 0:
        comp_signals.to_csv(os.path.join(OUTPUT_DIR, "mlp_competition_signals.csv"), index=False, encoding="utf-8-sig")
        print_competition_summary(comp_signals)
        trades_df = summarize_daily_trades(comp_signals)
        trades_df.to_csv(os.path.join(OUTPUT_DIR, "mlp_competition_daily_trades.csv"), index=False, encoding="utf-8-sig")
        print("\n" + trades_df.to_string(index=False))
    else:
        print("  No competition signals generated (check data availability for 2026.6).")

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_cols": feature_cols,
        "selected_idx": selected_idx,
        "input_dim": input_dim,
    }, os.path.join(OUTPUT_DIR, "mlp_model_checkpoint.pth"))

    print("\n" + "=" * 60)
    print(f"All MLP results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
