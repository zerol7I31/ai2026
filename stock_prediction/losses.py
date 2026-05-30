import torch
import torch.nn as nn
import torch.nn.functional as F


def ranknet_loss(preds, labels, device):
    n = preds.size(0)
    if n < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    preds_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
    labels_diff = labels.unsqueeze(1) - labels.unsqueeze(0)
    label_gt = (labels_diff > 0).float()
    label_lt = (labels_diff < 0).float()
    loss_matrix = label_gt * F.softplus(-preds_diff) + label_lt * F.softplus(preds_diff)
    denom = (label_gt.sum() + label_lt.sum()).clamp_min(1.0)
    return loss_matrix.sum() / denom


def mse_loss(preds, labels, device):
    return torch.nn.functional.mse_loss(preds, labels)


def list_mle_loss(preds, labels):
    _, indices = torch.sort(labels, descending=True)
    sorted_preds = preds[indices]
    log_cumsum = torch.logcumsumexp(sorted_preds.flip(0), dim=0).flip(0)
    return (log_cumsum - sorted_preds).mean()


def pearson_ic_loss(preds, labels):
    pred_mean = preds.mean()
    label_mean = labels.mean()
    pred_centered = preds - pred_mean
    label_centered = labels - label_mean
    cov = (pred_centered * label_centered).mean()
    pred_std = torch.sqrt((pred_centered ** 2).mean() + 1e-8)
    label_std = torch.sqrt((label_centered ** 2).mean() + 1e-8)
    pearson_r = cov / (pred_std * label_std + 1e-8)
    return 1.0 - pearson_r


def combined_loss(preds, labels, ic_loss_weight=0.5):
    huber = nn.functional.huber_loss(preds, labels, delta=0.2)
    ic_loss = pearson_ic_loss(preds, labels)
    return (1.0 - ic_loss_weight) * huber + ic_loss_weight * ic_loss
