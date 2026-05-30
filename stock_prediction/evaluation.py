import numpy as np
import pandas as pd
import torch


def compute_ic(predictions, labels):
    if len(predictions) < 2:
        return 0.0
    ic = np.corrcoef(predictions, labels)[0, 1]
    if np.isnan(ic):
        return 0.0
    return ic


def compute_ic_ir(predictions, labels, dates):
    df = pd.DataFrame({"pred": predictions, "label": labels, "date": dates})
    df["date"] = pd.to_datetime(df["date"])
    daily_ic = []
    daily_rank_ic = []
    for _, group in df.groupby(df["date"].dt.date):
        if len(group) < 2:
            continue
        ic = compute_ic(group["pred"].values, group["label"].values)
        daily_ic.append(ic)
        rank_ic = compute_ic(group["pred"].rank().values, group["label"].rank().values)
        daily_rank_ic.append(rank_ic)

    if not daily_ic:
        return 0.0, 0.0, 0.0, 0.0, np.array([])

    daily_ic = np.array(daily_ic)
    daily_rank_ic = np.array(daily_rank_ic)
    ic_mean = np.mean(daily_ic)
    ic_std = np.std(daily_ic)
    icir = ic_mean / (ic_std + 1e-9)
    rank_ic_mean = np.mean(daily_rank_ic)
    rank_icir = rank_ic_mean / (np.std(daily_rank_ic) + 1e-9)
    return ic_mean, icir, rank_ic_mean, rank_icir, daily_ic


def compute_direction_accuracy(predictions, labels):
    pred_dir = np.sign(predictions)
    true_dir = np.sign(labels)
    return np.mean(pred_dir == true_dir)


def predict_mlp(model, loader, sequences, device):
    all_preds = []
    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device)
            preds = model(features).cpu().numpy()
            all_preds.extend(preds.tolist())
    all_labels = [s.get("raw_label", s["label"]) for s in sequences]
    all_dates = [s.get("trade_date", s["date"]) for s in sequences]
    return np.array(all_preds), np.array(all_labels), all_dates


def ensemble_predict_gru(ensemble_models, loader, sequences, device):
    all_preds = []
    all_labels = []
    all_dates = []
    with torch.no_grad():
        for features, _, stock_ids, indices in loader:
            features, stock_ids = features.to(device), stock_ids.to(device)
            preds_ensemble = np.zeros(len(features), dtype=np.float64)
            for model in ensemble_models:
                model.eval()
                preds_ensemble += model(features, stock_ids).cpu().numpy().astype(np.float64)
            preds_ensemble /= len(ensemble_models)
            all_preds.extend(preds_ensemble.tolist())
            for idx in indices.tolist():
                sequence = sequences[idx]
                all_labels.append(sequence.get("raw_label", sequence["label"]))
                all_dates.append(sequence.get("trade_date", sequence["date"]))
    return np.array(all_preds), np.array(all_labels), all_dates


def cross_sectional_normalize(preds, dates):
    df = pd.DataFrame({"pred": preds, "date": dates})
    df["date"] = pd.to_datetime(df["date"])
    df["pred_cs_norm"] = np.nan
    for _, group in df.groupby("date"):
        if len(group) < 5:
            df.loc[group.index, "pred_cs_norm"] = group["pred"]
            continue
        vals = group["pred"].values
        mu = np.mean(vals)
        sigma = np.std(vals)
        if sigma < 1e-9:
            df.loc[group.index, "pred_cs_norm"] = 0.0
        else:
            df.loc[group.index, "pred_cs_norm"] = (vals - mu) / sigma
    return df["pred_cs_norm"].values
