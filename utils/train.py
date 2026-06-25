import torch
import torch.nn as nn
import copy
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
)
import mlflow

"""
def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
"""


def init_weights(m):
    # Only initialize Conv and Linear, skip if it's part of a Mamba module
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        # Check if it's a standard layer and not a specialized Mamba internal
        if "Mamba" not in str(type(m)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def train_model(
    train_loader,
    val_loader,
    num_epochs=100,
    patience=12,
    lr=5e-5,  # this is the *peak* LR after warmup
    weight_decay=0.1,
    pos_weight_value=2.12,
    model=None,
    device_nr=1,
    min_delta=1e-4,
    warmup_epochs=4,
    plateau_patience=8,  # epochs of no improvement before reducing LR
    loss_fn="bce",       # options: "bce", "focal", "focal_smooth"
    focal_gamma=2.0,     # focal loss gamma (only used when loss_fn contains "focal")
    label_smoothing=0.1, # label smoothing factor (only used with "focal_smooth")
    optimizer_type="adamw",  # options: "adamw", "adam", "sgd"
    scheduler_type="plateau",  # options: "plateau", "cosine", "cyclic", "none"
    use_amp=True,        # automatic mixed precision (fp16 forward/backward)
):
    device = torch.device(f"cuda:{device_nr}" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and device.type == "cuda"  # AMP only meaningful on GPU
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Using device: {device}  |  AMP: {use_amp}")

    model = model.to(device)
    model.apply(init_weights)

    # Optimizer
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    # Scheduler
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=plateau_patience,
            verbose=True,
            threshold=min_delta,
            min_lr=lr * 0.01,
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )
    elif scheduler_type == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr * 0.01,
            max_lr=lr,
            step_size_up=len(train_loader) * 4,
            mode="triangular2",
            cycle_momentum=False,
        )
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    # Loss
    import torch.nn.functional as F

    pos_weight_tensor = (
        torch.tensor([pos_weight_value], device=device)
        if pos_weight_value is not None else None
    )

    if loss_fn == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"BCEWithLogitsLoss with pos_weight={pos_weight_value:.2f}")

    elif loss_fn == "focal":
        # Focal loss: down-weights easy examples
        def criterion(logits, targets):
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight_tensor, reduction="none"
            )
            pt   = torch.exp(-bce)
            loss = ((1 - pt) ** focal_gamma * bce).mean()
            return loss
        print(f"Focal loss (gamma={focal_gamma}, pos_weight={pos_weight_value:.2f})")

    elif loss_fn == "focal_smooth":
        # Focal loss + label smoothing combined
        def criterion(logits, targets):
            targets_s = targets * (1 - label_smoothing) + 0.5 * label_smoothing
            bce = F.binary_cross_entropy_with_logits(
                logits, targets_s, pos_weight=pos_weight_tensor, reduction="none"
            )
            pt   = torch.exp(-F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            ))
            loss = ((1 - pt) ** focal_gamma * bce).mean()
            return loss
        print(f"Focal+LabelSmooth loss (gamma={focal_gamma}, smoothing={label_smoothing}, pos_weight={pos_weight_value:.2f})")

    else:
        raise ValueError(f"Unknown loss_fn: {loss_fn}. Choose from: bce, focal, focal_smooth")

    # Tracking
    best_val_pr_auc = 0.0
    best_state = None
    epochs_no_improve = 0
    lr_reduced = False  # flag to enforce reduction only once

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "train_auc": [],
        "train_pr_auc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_auc": [],
        "val_pr_auc": [],
    }

    for epoch in range(num_epochs):
        current_epoch = epoch + 1  # 1-based for readability

        # TRAIN
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        train_labels, train_preds, train_probs = [], [], []

        for signals, labels, _ in train_loader:
            signals = signals.to(device)
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(signals)
            logits_f = logits.float()
            # Replace any NaN/Inf in logits before they reach the loss
            if not torch.isfinite(logits_f).all():
                print(f"!!! NaN/Inf logits @ epoch {current_epoch} — skipping batch !!!")
                optimizer.zero_grad()
                continue
            logits_f = logits_f.clamp(-20, 20)
            loss = criterion(logits_f, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"!!! NaN/Inf loss @ epoch {current_epoch} — skipping batch !!!")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # tighter clip
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.numel()

            train_probs.extend(probs.detach().cpu().numpy().flatten().tolist())
            train_preds.extend(
                preds.detach().cpu().numpy().flatten().astype(int).tolist()
            )
            train_labels.extend(
                labels.detach().cpu().numpy().flatten().astype(int).tolist()
            )

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0.0

        # Filter out any NaN probs produced by skipped batches before sklearn
        train_probs_arr = np.array(train_probs)
        train_labels_arr = np.array(train_labels)
        train_preds_arr = np.array(train_preds)
        valid = np.isfinite(train_probs_arr)
        if not valid.all():
            print(f"  Filtering {(~valid).sum()} NaN probs from train metrics")
            train_probs_arr  = train_probs_arr[valid]
            train_labels_arr = train_labels_arr[valid]
            train_preds_arr  = train_preds_arr[valid]

        train_f1 = f1_score(train_labels_arr, train_preds_arr, zero_division=0)
        train_auc = (
            roc_auc_score(train_labels_arr, train_probs_arr)
            if len(set(train_labels_arr.tolist())) > 1
            else 0.5
        )
        train_pr_auc = (
            average_precision_score(train_labels_arr, train_probs_arr)
            if len(set(train_labels_arr.tolist())) > 1
            else float(train_labels_arr.mean())
        )

        # --- MLFLOW LOGGING FOR TRAIN ---
        mlflow.log_metric("train_loss", train_loss, step=current_epoch)
        mlflow.log_metric("train_acc", train_acc, step=current_epoch)
        mlflow.log_metric("train_f1", train_f1, step=current_epoch)
        mlflow.log_metric("train_auc", train_auc, step=current_epoch)
        mlflow.log_metric("train_pr_auc", train_pr_auc, step=current_epoch)

        # VALIDATION
        model.eval()
        val_running_loss = 0.0
        val_correct, val_total = 0, 0
        val_labels, val_preds, val_probs = [], [], []

        with torch.no_grad():
            for signals, labels, _ in val_loader:
                signals = signals.to(device)
                labels = labels.to(device).float().view(-1, 1)

                logits = model(signals)
                loss = criterion(logits.float().clamp(-20, 20), labels)

                val_running_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()

                val_probs.extend(probs.detach().cpu().numpy().flatten().tolist())
                val_preds.extend(
                    preds.detach().cpu().numpy().flatten().astype(int).tolist()
                )
                val_labels.extend(
                    labels.detach().cpu().numpy().flatten().astype(int).tolist()
                )

        val_loss = (
            val_running_loss / len(val_loader) if len(val_loader) > 0 else float("inf")
        )
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        # Filter NaN probs (can occur if model weights got corrupted by a NaN batch)
        val_probs_arr = np.array(val_probs)
        val_labels_arr = np.array(val_labels)
        val_preds_arr = np.array(val_preds)
        valid_val = np.isfinite(val_probs_arr)
        if not valid_val.all():
            print(f"  Filtering {(~valid_val).sum()} NaN probs from val metrics")
            val_probs_arr  = val_probs_arr[valid_val]
            val_labels_arr = val_labels_arr[valid_val]
            val_preds_arr  = val_preds_arr[valid_val]

        val_f1 = f1_score(val_labels_arr, val_preds_arr, zero_division=0)
        val_auc = (
            roc_auc_score(val_labels_arr, val_probs_arr) if len(set(val_labels_arr.tolist())) > 1 else 0.5
        )
        val_pr_auc = (
            average_precision_score(val_labels_arr, val_probs_arr)
            if len(set(val_labels_arr.tolist())) > 1
            else float(val_labels_arr.mean())
        )

        # --- MLFLOW LOGGING FOR VAL ---
        current_lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metric("val_loss", val_loss, step=current_epoch)
        mlflow.log_metric("val_acc", val_acc, step=current_epoch)
        mlflow.log_metric("val_f1", val_f1, step=current_epoch)
        mlflow.log_metric("val_auc", val_auc, step=current_epoch)
        mlflow.log_metric("val_pr_auc", val_pr_auc, step=current_epoch)
        mlflow.log_metric("lr", current_lr, step=current_epoch)

        # Learning Rate Warmup & Plateau
        if current_epoch <= warmup_epochs:
            # Linear warmup: from ~1% to 100% of target lr
            progress = current_epoch / warmup_epochs
            warmup_lr = lr * max(0.01, progress)  # avoid zero
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
        else:
            if scheduler is None:
                pass
            elif scheduler_type == "plateau":
                # Only reduce once
                if not lr_reduced:
                    scheduler.step(val_pr_auc)
                    current_lr = optimizer.param_groups[0]["lr"]
                    if current_lr < lr * 0.99:
                        lr_reduced = True
                        print(
                            f"-> LR reduced once to {current_lr:.2e} (after {plateau_patience} epochs no improvement)"
                        )
                        epochs_no_improve = 0
            elif scheduler_type == "cosine":
                scheduler.step()
            elif scheduler_type == "cyclic":
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        print(
            f"Epoch {current_epoch:03d}/{num_epochs} | "
            f"TRAIN [Loss: {train_loss:.4f} Acc: {train_acc:.3f} AUC: {train_auc:.3f} PR: {train_pr_auc:.3f} F1: {train_f1:.3f}] | "
            f"VAL   [Loss: {val_loss:.4f} Acc: {val_acc:.3f} AUC: {val_auc:.3f} PR: {val_pr_auc:.3f} F1: {val_f1:.3f}] | "
            f"LR: {current_lr:.2e}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["train_auc"].append(train_auc)
        history["train_pr_auc"].append(train_pr_auc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)
        history["val_pr_auc"].append(val_pr_auc)

        # Early stopping & best model
        if val_pr_auc > best_val_pr_auc + min_delta:
            print(f"   -> New best val PR-AUC: {val_pr_auc:.4f} (saving)")
            best_val_pr_auc = val_pr_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping at epoch {current_epoch} (no PR-AUC improve for {patience} epochs)"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best weights (highest val PR-AUC)")

    return model, history
