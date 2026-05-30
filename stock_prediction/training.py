import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from .losses import combined_loss, ranknet_loss


def train_mlp_model(model, train_loader, val_loader, save_path, config, output_dir):
    device = config["device"]
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config["patience"] // 2,
    )
    use_amp = device != "cpu"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        batch_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} Train")
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(features)
                    loss = ranknet_loss(preds, labels, device)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(features)
                loss = ranknet_loss(preds, labels, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_train_loss = total_loss / max(batch_count, 1)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                preds = model(features)
                loss = ranknet_loss(preds, labels, device)
                total_val_loss += loss.item()
                val_batch_count += 1
        avg_val_loss = total_val_loss / max(val_batch_count, 1)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    _plot_loss_curve(
        train_losses,
        val_losses,
        title="MLP Training and Validation Loss",
        filename="loss_curve.png",
        output_dir=output_dir,
    )
    return train_losses, val_losses


def train_gru_model(model, train_loader, val_loader, save_path, config, output_dir):
    device = config["device"]
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    warmup_epochs = config.get("warmup_epochs", 5)
    total_epochs = config["epochs"]
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=1e-6,
            ),
        ],
        milestones=[warmup_epochs],
    )

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    mixup_alpha = config.get("mixup_alpha", 0.0)
    ic_loss_weight = config.get("ic_loss_weight", 0.5)

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        batch_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} Train")
        for features, labels, stock_ids, _ in pbar:
            features, labels, stock_ids = features.to(device), labels.to(device), stock_ids.to(device)

            if mixup_alpha > 0 and epoch >= warmup_epochs:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                if lam < 0.5:
                    lam = 1.0 - lam
                perm = torch.randperm(features.size(0), device=features.device)
                features = lam * features + (1 - lam) * features[perm]
                labels = lam * labels + (1 - lam) * labels[perm]

            optimizer.zero_grad()
            preds = model(features, stock_ids)
            loss = combined_loss(preds, labels, ic_loss_weight=ic_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        avg_train_loss = total_loss / max(batch_count, 1)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for features, labels, stock_ids, _ in val_loader:
                features, labels, stock_ids = features.to(device), labels.to(device), stock_ids.to(device)
                preds = model(features, stock_ids)
                loss = combined_loss(preds, labels, ic_loss_weight=ic_loss_weight)
                total_val_loss += loss.item()
                val_batch_count += 1
        avg_val_loss = total_val_loss / max(val_batch_count, 1)
        val_losses.append(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f}, "
            f"Val Loss={avg_val_loss:.6f}, LR={current_lr:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    _plot_loss_curve(
        train_losses,
        val_losses,
        title="Training and Validation Loss",
        filename="loss_curve.png",
        output_dir=output_dir,
    )
    return train_losses, val_losses


def _plot_loss_curve(train_losses, val_losses, title, filename, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()
