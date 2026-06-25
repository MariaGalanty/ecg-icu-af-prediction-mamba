import warnings

# This must come before any other imports
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import random

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from models.cnn_ribero import RibeiroECGNet
from models.mamba_ribeiro import RibeiroMambaECGNet
from utils.data import SignalWindowDatasetEqual, get_record_paths_labels_patients
from utils.evaluate import evaluate_on_test
from utils.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="ECG AF Prediction Training Script")

    # General Experiment Params
    parser.add_argument(
        "--exp_name",
        type=str,
        default="5fold_experiment",
        help="Experiment name for saving results",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="mamba",
        choices=["mamba", "resnet"],
        help="Choose model architecture",
    )

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument(
        "--loss_fn", type=str, default="bce",
        choices=["bce", "focal", "focal_smooth"],
        help="Loss function: bce | focal | focal_smooth (focal + label smoothing)",
    )
    parser.add_argument(
        "--focal_gamma", type=float, default=2.0,
        help="Gamma for focal loss (ignored for bce)",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1,
        help="Label smoothing factor for focal_smooth (ignored for bce/focal)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimizer",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "cyclic", "none"],
        help="LR scheduler",
    )
    parser.add_argument(
        "--plateau_patience",
        type=int,
        default=4,
        help="Epochs before ReduceLROnPlateau halves the LR",
    )
    parser.add_argument(
        "--n_folds", type=int, default=3, help="Number of folds for cross-validation"
    )

    # Data Params
    parser.add_argument("--window_minutes", type=int, default=30)
    parser.add_argument(
        "--max_windows", type=int, default=4, help="Max windows per record"
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.20,
        help="Fraction of training patients used for validation",
    )

    # Model parameters
    parser.add_argument(
        "--n_mamba_blocks",
        type=int,
        default=2,
        help="Number of mamba blocks in the model",
    )

    # Dataset selection (both on by default, use --no_snow or --no_mimic to exclude)
    parser.add_argument(
        "--no_snow",
        dest="use_snow",
        action="store_false",
        default=True,
        help="Exclude Snowflake dataset",
    )
    parser.add_argument(
        "--no_mimic",
        dest="use_mimic",
        action="store_false",
        default=True,
        help="Exclude MIMIC dataset",
    )

    # Notes (visible as a tag in MLflow UI)
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-text note attached to the MLflow run as a tag",
    )

    # Hardware
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")

    return parser.parse_args()


def get_model(args):
    """Factory function to initialize the chosen model."""
    if args.model_type == "mamba":
        return RibeiroMambaECGNet(
            n_classes=1,
            duration_minutes=float(args.window_minutes),
            fs=500,
            kernel_size=16,
            n_mamba_blocks=args.n_mamba_blocks,
            pre_mamba_stride=4,
            conv_dropout=0.3,
            mamba_dropout=0.2,
            final_dropout=0.5,
        )
    elif args.model_type == "resnet":
        return RibeiroECGNet(
            n_classes=1, duration_minutes=float(args.window_minutes), fs=500
        )


def _safe_metrics(d):
    """Replace NaN/inf values with 0.0 so MLflow can always log them."""
    return {
        k: float(v) if isinstance(v, (int, float)) and np.isfinite(v) else 0.0
        for k, v in d.items()
    }


def run_project(args):
    # ───────────────────────────────────────────────
    # 0. Print Configuration
    # ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("      RUNNING EXPERIMENT CONFIGURATION")
    print("=" * 50)
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"  {key:20}: {value}")
    print("=" * 50 + "\n")

    # ───────────────────────────────────────────────
    # 1. Setup MLflow & Parent Run
    # ───────────────────────────────────────────────
    # Set the experiment name (creates a category in the UI)
    mlflow.set_experiment(args.exp_name)

    client = mlflow.MlflowClient()

    # Start the Parent Run (The "Container" for all folds)
    with mlflow.start_run(
        run_name=f"Execution_{pd.Timestamp.now().strftime('%m%d_%H%M')}"
    ) as parent_run:
        parent_run_id = parent_run.info.run_id
        if args.notes:
            mlflow.set_tag("notes", args.notes)
        # Log all Argparse settings as parameters
        mlflow.log_params(args_dict)

        # ───────────────────────────────────────────────
        # 2. Paths & Data Loading
        # ───────────────────────────────────────────────
        AF_DIR_SNOW = "/home/maria/data/AF_prediction_data/AF_prediction_project_lead_II_v2/snowflake/Pre_AF_SR/"
        SR_DIR_SNOW = "/home/maria/data/AF_prediction_data/AF_prediction_project_lead_II_v2/snowflake/Pure_SR_Controls/"
        MIMIC_BASE = "/home/maria/data/AF_prediction_data/AF_prediction_project_lead_II_v2/mimic/"
        MIMIC_SUBDIRS = [
            ("30/Pre_AF_SR/", "30/Pure_SR_Controls/"),
            ("31/Pre_AF_SR/", "31/Pure_SR_Controls/"),
        ]

        if not args.use_snow and not args.use_mimic:
            raise ValueError(
                "At least one of --use_snow or --use_mimic must be enabled."
            )

        all_paths, all_labels, all_pids = [], [], []

        if args.use_snow:
            snow_paths, snow_labels, snow_pids = get_record_paths_labels_patients(
                AF_DIR_SNOW, SR_DIR_SNOW, dataset_prefix="SF_"
            )
            all_paths.append(snow_paths)
            all_labels.append(snow_labels)
            all_pids.append(snow_pids)
            print(f"Snowflake: {len(snow_pids)} records loaded")

        if args.use_mimic:
            total_mimic = 0
            for af_sub, sr_sub in MIMIC_SUBDIRS:
                m_paths, m_labels, m_pids = get_record_paths_labels_patients(
                    MIMIC_BASE + af_sub, MIMIC_BASE + sr_sub, dataset_prefix="M_"
                )
                all_paths.append(m_paths)
                all_labels.append(m_labels)
                all_pids.append(m_pids)
                total_mimic += len(m_pids)
                print(f"MIMIC ({af_sub.split('/')[0]}): {len(m_pids)} records loaded")
            print(f"MIMIC total: {total_mimic} records")

        file_paths = np.concatenate(all_paths)
        labels = np.concatenate(all_labels)
        patient_ids = np.concatenate(all_pids)

        patient_to_label = {pid: lab for pid, lab in zip(patient_ids, labels)}
        unique_patients = np.array(list(patient_to_label.keys()))
        unique_patient_labels = np.array([patient_to_label[p] for p in unique_patients])

        # ───────────────────────────────────────────────
        # 3. Per-dataset stratified fold assignment
        # ───────────────────────────────────────────────
        # Split each dataset independently so that positive/negative ratios
        # are balanced within Snowflake AND within MIMIC across every fold.
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

        # Assign each patient a fold index, stratified within their dataset
        patient_fold = np.full(len(unique_patients), -1, dtype=int)

        dataset_prefixes = []
        if args.use_snow:
            dataset_prefixes.append("SF_")
        if args.use_mimic:
            dataset_prefixes.append("M_")

        for prefix in dataset_prefixes:
            mask = np.array([str(p).startswith(prefix) for p in unique_patients])
            ds_patients = unique_patients[mask]
            ds_labels = unique_patient_labels[mask]
            for fold_idx, (_, test_idx) in enumerate(skf.split(ds_patients, ds_labels)):
                patient_fold[np.where(mask)[0][test_idx]] = fold_idx

        # Build fold splits from the per-dataset assignments
        fold_splits = [
            (
                np.where(patient_fold != fold_idx)[0],  # train_val
                np.where(patient_fold == fold_idx)[0],  # test
            )
            for fold_idx in range(args.n_folds)
        ]

        all_fold_test_metrics = []
        all_fold_ds_metrics = {"snow": [], "mimic": []}  # per-dataset accumulators

        for fold, (train_val_idx, test_idx) in enumerate(fold_splits):
            fold_num = fold + 1
            print(f"\n{'=' * 30} FOLD {fold_num} / {args.n_folds} {'=' * 30}")

            # START NESTED MLFLOW RUN (One per fold)
            with mlflow.start_run(run_name=f"Fold_{fold_num}", nested=True):
                test_patients = unique_patients[test_idx]
                train_val_patients = unique_patients[train_val_idx]
                train_val_labels = unique_patient_labels[train_val_idx]

                train_patients, val_patients = train_test_split(
                    train_val_patients,
                    test_size=args.val_fraction,
                    stratify=train_val_labels,
                    random_state=42,
                )

                train_mask = np.isin(patient_ids, train_patients)
                val_mask = np.isin(patient_ids, val_patients)
                test_mask = np.isin(patient_ids, test_patients)

                ds_params = {
                    "window_minutes": args.window_minutes,
                    "max_windows_per_record": args.max_windows,
                }
                train_ds = SignalWindowDatasetEqual(
                    file_paths[train_mask],
                    labels[train_mask],
                    augment=True,
                    **ds_params,
                )
                val_ds = SignalWindowDatasetEqual(
                    file_paths[val_mask], labels[val_mask], augment=False, **ds_params
                )
                test_ds = SignalWindowDatasetEqual(
                    file_paths[test_mask], labels[test_mask], augment=False, **ds_params
                )

                n_pos, n_neg = (
                    (train_ds.labels == 1).sum(),
                    (train_ds.labels == 0).sum(),
                )
                pos_weight = min(n_neg / max(n_pos, 1), 4.0)

                train_loader = DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    val_ds, batch_size=args.batch_size, shuffle=False
                )
                test_loader = DataLoader(
                    test_ds, batch_size=args.batch_size, shuffle=False
                )

                model = get_model(args)

                # Train - Ensure your train_model function has mlflow.log_metric calls inside
                model, history = train_model(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=args.epochs,
                    patience=args.patience,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    pos_weight_value=pos_weight,
                    model=model,
                    device_nr=args.device,
                    warmup_epochs=args.warmup_epochs,
                    loss_fn=args.loss_fn,
                    focal_gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                    optimizer_type=args.optimizer,
                    scheduler_type=args.scheduler,
                    plateau_patience=args.plateau_patience,
                )

                # Log per-epoch metrics to the parent run with fold-prefixed names
                history_df = pd.DataFrame(history)
                history_df.index = history_df.index + 1  # 1-based epoch
                for epoch, row in history_df.iterrows():
                    for metric, value in row.items():
                        if np.isfinite(value):
                            client.log_metric(
                                parent_run_id,
                                f"fold{fold_num}_{metric}",
                                value,
                                step=epoch,
                            )

                # ── Evaluate: combined test set ───────────────────────────────
                metrics = evaluate_on_test(
                    model=model, test_loader=test_loader, device_nr=args.device
                )
                metrics["fold"] = fold_num
                all_fold_test_metrics.append(metrics)

                fold_test_metrics = {
                    "test_window_accuracy": metrics["window_accuracy"],
                    "test_window_roc_auc": metrics["window_roc_auc"],
                    "test_window_pr_auc": metrics["window_pr_auc"],
                    "test_patient_accuracy": metrics["patient_accuracy"],
                    "test_patient_roc_auc": metrics["patient_roc_auc"],
                    "test_patient_pr_auc": metrics["patient_pr_auc"],
                    "test_n_patients": metrics["n_patients_test"],
                    "test_n_windows": metrics["n_windows_test"],
                }
                mlflow.log_metrics(_safe_metrics(fold_test_metrics))

                # ── Evaluate: per-dataset breakdown ───────────────────────────
                dataset_splits = {}
                if args.use_snow:
                    dataset_splits["snow"] = ("SF_", {})
                if args.use_mimic:
                    dataset_splits["mimic"] = ("M_", {})

                for ds_name, (prefix, _) in dataset_splits.items():
                    ds_test_mask = np.isin(
                        patient_ids[test_mask],
                        [p for p in test_patients if str(p).startswith(prefix)],
                    )
                    if ds_test_mask.sum() == 0:
                        continue
                    ds_test_ds = SignalWindowDatasetEqual(
                        file_paths[test_mask][ds_test_mask],
                        labels[test_mask][ds_test_mask],
                        augment=False,
                        **ds_params,
                    )
                    ds_test_loader = DataLoader(
                        ds_test_ds, batch_size=args.batch_size, shuffle=False
                    )
                    ds_metrics = evaluate_on_test(
                        model=model, test_loader=ds_test_loader, device_nr=args.device
                    )
                    ds_test_metrics = {
                        f"{ds_name}_test_window_accuracy": ds_metrics[
                            "window_accuracy"
                        ],
                        f"{ds_name}_test_window_roc_auc": ds_metrics["window_roc_auc"],
                        f"{ds_name}_test_window_pr_auc": ds_metrics["window_pr_auc"],
                        f"{ds_name}_test_patient_accuracy": ds_metrics[
                            "patient_accuracy"
                        ],
                        f"{ds_name}_test_patient_roc_auc": ds_metrics[
                            "patient_roc_auc"
                        ],
                        f"{ds_name}_test_patient_pr_auc": ds_metrics["patient_pr_auc"],
                        f"{ds_name}_test_n_patients": ds_metrics["n_patients_test"],
                    }
                    mlflow.log_metrics(_safe_metrics(ds_test_metrics))
                    dataset_splits[ds_name] = (prefix, ds_metrics)
                    if ds_name in all_fold_ds_metrics:
                        all_fold_ds_metrics[ds_name].append(ds_metrics)

                # Save the model for this specific fold
                # mlflow.pytorch.log_model(model, f"model_fold_{fold_num}")
                model_path = f"best_model_fold_{fold_num}.pt"
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path)
                os.remove(model_path)  # Clean up local file after uploading to MLflow

            # Also log combined + per-dataset test metrics to the parent run (step=fold_num)
            # so they are visible without navigating into nested runs
            mlflow.log_metrics(_safe_metrics(fold_test_metrics), step=fold_num)
            for ds_name, (_, ds_metrics) in dataset_splits.items():
                if not ds_metrics:
                    continue
                mlflow.log_metrics(
                    _safe_metrics(
                        {
                            f"{ds_name}_test_window_accuracy": ds_metrics[
                                "window_accuracy"
                            ],
                            f"{ds_name}_test_window_roc_auc": ds_metrics[
                                "window_roc_auc"
                            ],
                            f"{ds_name}_test_window_pr_auc": ds_metrics[
                                "window_pr_auc"
                            ],
                            f"{ds_name}_test_patient_accuracy": ds_metrics[
                                "patient_accuracy"
                            ],
                            f"{ds_name}_test_patient_roc_auc": ds_metrics[
                                "patient_roc_auc"
                            ],
                            f"{ds_name}_test_patient_pr_auc": ds_metrics[
                                "patient_pr_auc"
                            ],
                            f"{ds_name}_test_n_patients": ds_metrics["n_patients_test"],
                        }
                    ),
                    step=fold_num,
                )

        # ───────────────────────────────────────────────
        # 4. Final Summary & Saving
        # ───────────────────────────────────────────────
        df = pd.DataFrame(all_fold_test_metrics).set_index("fold")
        print("\nFinal Results:\n", df.round(4))

        # Log average combined metrics across all folds
        numeric_cols = df.select_dtypes(include="number").columns
        avg_metrics = {f"avg_{col}": df[col].mean() for col in numeric_cols}
        mlflow.log_metrics(_safe_metrics(avg_metrics))

        # Log average per-dataset metrics across all folds
        for ds_name, fold_results in all_fold_ds_metrics.items():
            if not fold_results:
                continue
            ds_df = pd.DataFrame(fold_results)
            for col in ds_df.select_dtypes(include="number").columns:
                mlflow.log_metric(f"avg_{ds_name}_{col}", ds_df[col].mean())

        os.makedirs("results", exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        csv_path = f"results/{args.exp_name}_{timestamp}.csv"
        df.to_csv(csv_path)

        # Log the final CSV as an artifact
        mlflow.log_artifact(csv_path)

    print("Experiment complete. All data logged to MLflow.")


if __name__ == "__main__":
    args = parse_args()
    run_project(args)
