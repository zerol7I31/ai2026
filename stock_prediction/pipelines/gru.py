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
    build_price_pivots,
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
    cross_sectional_normalize,
    ensemble_predict_gru,
)
from stock_prediction.features import (
    build_feature_panel,
    compute_gru_features,
    normalize_features,
)
from stock_prediction.models import StockGRUModel, StockSequenceDataset
from stock_prediction.sequences import (
    apply_cross_sectional_rank,
    build_gru_sequences,
    filter_sequences_by_trade_period,
)
from stock_prediction.settings import GRU_CONFIG, GRU_OUTPUT_DIR
from stock_prediction.signals import (
    generate_gru_competition_signals,
    print_competition_summary,
    summarize_daily_trades,
)
from stock_prediction.training import train_gru_model
from stock_prediction.utils import set_seed


def main(config=None):
    config = config or GRU_CONFIG
    output_dir = config.get("output_dir", GRU_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("Deep Learning Stock Trend Prediction - GRU with Attention")
    print("Memory Optimization: float32 + category dtypes + vectorized ST filter")
    print(f"Using device: {config['device']}")
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
    stock_pool = select_stock_pool(panel, config)
    print(f"  Selected {len(stock_pool)} stocks")
    panel = panel[panel["ts_code"].isin(stock_pool)].copy()
    panel["ts_code"] = panel["ts_code"].cat.remove_unused_categories()
    gc.collect()
    print(f"  Filtered panel: {len(panel):,} rows, {panel['ts_code'].nunique()} stocks")
    panel.info(memory_usage="deep")

    print("\n[4/9] Computing features per stock (float32)...")
    panel, feature_cols = build_feature_panel(
        panel,
        lambda group: compute_gru_features(group, config),
    )
    print(f"  Feature columns ({len(feature_cols)} dims): {feature_cols}")

    print("\n[5/9] Normalizing features (rolling per-stock, no lookahead)...")
    panel = normalize_features(panel, feature_cols, mode="rolling_252")

    print("\n[6/9] Building sequences (label: forward return over prediction horizon)...")
    sequences = build_gru_sequences(panel, feature_cols, stock_pool, config)
    print(f"  Total sequences: {len(sequences):,}")

    print("  Applying cross-sectional rank normalization to labels...")
    sequences = apply_cross_sectional_rank(
        sequences,
        enabled=config.get("label_type") == "cs_rank",
    )

    train_seq = filter_sequences_by_trade_period(sequences, config["train_start"], config["train_end"])
    val_seq = filter_sequences_by_trade_period(sequences, config["val_start"], config["val_end"])
    test_seq = filter_sequences_by_trade_period(sequences, config["test_start"], config["test_end"])
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    train_eval_loader = DataLoader(
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

    input_dim = train_dataset.features.shape[2]
    ensemble_seeds = config.get("ensemble_seeds", [42])
    print(f"\n[7/9] Training ensemble of {len(ensemble_seeds)} GRU models (seeds: {ensemble_seeds})...")
    ensemble_models = []

    for seed_index, seed in enumerate(ensemble_seeds):
        set_seed(seed)
        print(f"\n  --- Model {seed_index + 1}/{len(ensemble_seeds)} (seed={seed}) ---")

        model = StockGRUModel(
            input_dim=input_dim,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            num_stocks=num_stocks,
            embed_dim=config["embed_dim"],
        )
        print(f"  Model: GRU (input={input_dim}, hidden={config['hidden_size']}, layers={config['num_layers']})")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        model_path = os.path.join(output_dir, f"best_model_seed{seed}.pth")
        if val_loader is not None:
            train_gru_model(model, train_loader, val_loader, model_path, config, output_dir)
        else:
            print("  No validation set, training without validation...")
            train_gru_model(model, train_loader, train_loader, model_path, config, output_dir)

        model.load_state_dict(torch.load(model_path, map_location=config["device"], weights_only=True))
        model.eval()
        ensemble_models.append(model)

    print(f"\n  Ensemble of {len(ensemble_models)} models trained.")

    print("\n[8/9] Evaluating ensemble...")
    device = config["device"]

    train_preds, train_labels, train_dates = ensemble_predict_gru(
        ensemble_models,
        train_eval_loader,
        train_seq,
        device,
    )
    val_preds, val_labels, val_dates = (
        ensemble_predict_gru(ensemble_models, val_loader, val_seq, device)
        if val_seq
        else (np.array([]), np.array([]), [])
    )
    test_preds, test_labels, test_dates = (
        ensemble_predict_gru(ensemble_models, test_loader, test_seq, device)
        if test_seq
        else (np.array([]), np.array([]), [])
    )

    if len(train_preds) > 0:
        train_preds = cross_sectional_normalize(train_preds, train_dates)
    if len(val_preds) > 0:
        val_preds = cross_sectional_normalize(val_preds, val_dates)
    if len(test_preds) > 0:
        test_preds = cross_sectional_normalize(test_preds, test_dates)

    eval_data = []
    for name, preds, labels, dates in [
        ("Train", train_preds, train_labels, train_dates),
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
    eval_df.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False, encoding="utf-8-sig")
    print(eval_df.to_string(index=False))

    print("\n[9/9] Running backtest...")
    all_predictions = [
        {
            "ts_code": sequence["ts_code"],
            "date": sequence.get("trade_date", sequence["date"]),
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

    print("  Building price pivot for vectorized backtest...")
    prices_pivot, open_pivot = build_price_pivots(panel, stock_pool)

    daily_values, _ = backtest_strategy(prices_pivot, predictions_pivot, config, open_pivot=open_pivot)
    if daily_values:
        bt_df = pd.DataFrame(daily_values).sort_values("date")
        bt_df.to_csv(os.path.join(output_dir, "backtest_daily_values.csv"), index=False, encoding="utf-8-sig")
        metrics = compute_backtest_metrics(bt_df, initial_capital=config["initial_capital"])
        pd.DataFrame([metrics]).to_csv(
            os.path.join(output_dir, "backtest_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        print("\nBacktest Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        benchmark = load_benchmark_data()
        bench_df = align_benchmark_to_backtest(benchmark, bt_df)
        plot_backtest_curve(
            bt_df,
            bench_df,
            "Strategy vs Benchmark",
            "backtest_curve.png",
            output_dir,
            strategy_label="Strategy",
            initial_capital=config["initial_capital"],
        )
    else:
        print("  No backtest data generated.")

    print("\nGenerating competition signals (2026.6.1 - 2026.6.12)...")
    comp_signals = generate_gru_competition_signals(
        panel,
        ensemble_models,
        feature_cols,
        cal_df,
        stock_pool,
        code_to_id,
        config,
    )
    if len(comp_signals) > 0:
        comp_signals.to_csv(os.path.join(output_dir, "competition_signals.csv"), index=False, encoding="utf-8-sig")
        print_competition_summary(comp_signals)
        trades_df = summarize_daily_trades(comp_signals)
        trades_df.to_csv(os.path.join(output_dir, "competition_daily_trades.csv"), index=False, encoding="utf-8-sig")
        print("\n" + trades_df.to_string(index=False))
    else:
        print("  No competition signals generated (check data availability for 2026.6).")

    torch.save({
        "model_state_dicts": [model.state_dict() for model in ensemble_models],
        "config": config,
        "feature_cols": feature_cols,
        "input_dim": input_dim,
        "code_to_id": code_to_id,
        "ensemble_seeds": ensemble_seeds,
    }, os.path.join(output_dir, "ensemble_checkpoint.pth"))

    print("\n" + "=" * 60)
    print(f"All results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
