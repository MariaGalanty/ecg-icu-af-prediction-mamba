# Prediction of Atrial Fibrillation from Continuous ICU ECG

This repository contains the code for training and evaluating deep learning models for predicting atrial fibrillation (AF) onset from continuous single-lead ECG recordings in the ICU setting.

## Overview

We train and compare two model architectures on a combined dataset of Amsterdam UMC (Snowflake) and MIMIC-IV ICU ECG recordings:

- **Mamba-ResNet** (`RibeiroMambaECGNet`): A hybrid architecture combining ResNet feature extraction with Mamba selective state-space blocks for long-range temporal modelling
- **ResNet** (`RibeiroECGNet`): A 1D ResNet baseline inspired by Ribeiro et al.

Both models are trained using 3-fold patient-stratified cross-validation with temperature scaling calibration.

## Repository Structure

```
├── main.py                          # Training entry point
├── config.py                        # Configuration
├── models/
│   ├── mamba_ribeiro.py             # Mamba-ResNet model (RibeiroMambaECGNet)
│   └── cnn_ribero.py                # ResNet baseline (RibeiroECGNet)
├── utils/
│   ├── data.py                      # Data loading and windowing
│   ├── train.py                     # Training loop
│   └── evaluate.py                  # Evaluation metrics
└── evaluation/
    └── gradcam_mamba_d16_30min_v2.ipynb   # Saliency visualization (Grad-CAM, Integrated Gradients, Mamba Δ-proxy)
```

## Data Format

The code expects ECG recordings in binary `.dat` / `.hea` (WFDB-compatible) format, organised as:

```
AF_DIR/
  <patient_id>_<date>_<time>.dat
  <patient_id>_<date>_<time>.hea
SR_DIR/
  ...
```

Data paths are configured in `main.py` via command-line arguments.

## Training

```bash
python main.py \
  --model_type mamba \
  --window_minutes 30 \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --exp_name mamba_d16_30min
```

Key arguments:
| Argument | Default | Description |
|---|---|---|
| `--model_type` | `mamba` | `mamba` or `resnet` |
| `--window_minutes` | `30` | ECG window length in minutes |
| `--batch_size` | `8` | Batch size |
| `--lr` | `1e-5` | Learning rate |
| `--epochs` | `100` | Max epochs |
| `--patience` | `6` | Early stopping patience |

## Explainability

The notebook `evaluation/gradcam_mamba_d16_30min_v2.ipynb` provides an interactive saliency viewer with four methods:

- **Grad-CAM**: Gradient-weighted class activation maps via the last ResNet stage
- **Integrated Gradients**: Axiomatically grounded full-model attribution
- **Mamba Δ-proxy**: ‖output − input‖₂ per MambaResidualBlock time step, a proxy for Δ-driven activity
- **Grad-CAM × Δ**: Combined method highlighting regions that are both discriminative and actively updated

## Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

Note: `mamba-ssm` requires a CUDA-capable GPU and compatible CUDA/PyTorch versions. See the [mamba-ssm repository](https://github.com/state-spaces/mamba) for installation instructions.

## Citation

If you use this code, please cite our work (citation to be added upon publication).

## License

This repository is made available for research purposes. The data used in this study is not publicly available due to patient privacy restrictions.
