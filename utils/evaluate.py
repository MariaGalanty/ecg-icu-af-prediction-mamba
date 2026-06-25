import torch
import torch.nn as nn
import copy
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
    classification_report,
)


def evaluate_on_test(model, test_loader, threshold=0.5, device_nr=1):
    """
    Evaluates model on test set.
    Prints window-level and patient-level results.
    Returns a dictionary with all key metrics for cross-fold summarization.
    """
    device = torch.device(f"cuda:{device_nr}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []
    all_record_idx = []

    with torch.no_grad():
        for signals, labels, record_idx in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            logits = model(signals)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_record_idx.extend(record_idx.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels).astype(int)
    all_record_idx = np.array(all_record_idx).astype(int)
    preds = (all_probs >= threshold).astype(int)

    # ───────────────────────────────────────────────
    # Window-level metrics
    # ───────────────────────────────────────────────
    window_metrics = {}
    window_metrics["accuracy"] = accuracy_score(all_labels, preds)
    window_metrics["roc_auc"] = (
        roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else np.nan
    )
    window_metrics["pr_auc"] = average_precision_score(all_labels, all_probs)

    print("\n" + "=" * 50)
    print("          TEST EVALUATION - WINDOW LEVEL")
    print("=" * 50)
    print(f"Threshold:          {threshold:.3f}")
    print(
        f"Accuracy:           {window_metrics['accuracy']:.4f} ({window_metrics['accuracy'] * 100:.2f}%)"
    )
    print(f"ROC-AUC:            {window_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:             {window_metrics['pr_auc']:.4f}")
    print("\nConfusion Matrix (true \\ pred):")
    print(confusion_matrix(all_labels, preds))

    # ───────────────────────────────────────────────
    # Patient-level metrics (aggregation by mean probability)
    # ───────────────────────────────────────────────
    unique_records = np.unique(all_record_idx)

    patient_true = []
    patient_pred = []
    patient_prob_mean = []

    for rid in unique_records:
        mask = all_record_idx == rid
        if not np.any(mask):
            continue

        labels_r = all_labels[mask]
        preds_r = preds[mask]
        probs_r = all_probs[mask]

        true_label = int(round(labels_r.mean())) if len(labels_r) > 0 else 0
        mean_prob = probs_r.mean() if len(probs_r) > 0 else 0.0
        pred_label = 1 if mean_prob >= threshold else 0  # using same threshold

        patient_true.append(true_label)
        patient_pred.append(pred_label)
        patient_prob_mean.append(mean_prob)

    patient_true = np.array(patient_true)
    patient_pred = np.array(patient_pred)
    patient_prob_mean = np.array(patient_prob_mean)

    patient_metrics = {}
    patient_metrics["accuracy"] = accuracy_score(patient_true, patient_pred)
    patient_metrics["roc_auc"] = (
        roc_auc_score(patient_true, patient_prob_mean)
        if len(set(patient_true)) > 1
        else np.nan
    )
    patient_metrics["pr_auc"] = average_precision_score(patient_true, patient_prob_mean)

    print("\n" + "=" * 50)
    print("        TEST EVALUATION - SOURCE SAMPLE LEVEL")
    print("=" * 50)
    print(f"Number of source samples: {len(patient_true)}")
    print(
        f"Accuracy:           {patient_metrics['accuracy']:.4f} ({patient_metrics['accuracy'] * 100:.2f}%)"
    )
    print(f"ROC-AUC (mean prob):{patient_metrics['roc_auc']:.4f}")
    print(f"PR-AUC (mean prob): {patient_metrics['pr_auc']:.4f}")
    print("Confusion Matrix (true \\ pred):")
    print(confusion_matrix(patient_true, patient_pred))

    # ───────────────────────────────────────────────
    # Return dictionary for cross-fold summary
    # ───────────────────────────────────────────────
    return {
        "window_accuracy": window_metrics["accuracy"],
        "window_roc_auc": window_metrics["roc_auc"],
        "window_pr_auc": window_metrics["pr_auc"],
        "patient_accuracy": patient_metrics["accuracy"],
        "patient_roc_auc": patient_metrics["roc_auc"],
        "patient_pr_auc": patient_metrics["pr_auc"],
        # You can add more if needed, e.g. number of patients/test samples
        "n_patients_test": len(patient_true),
        "n_windows_test": len(all_labels),
    }
