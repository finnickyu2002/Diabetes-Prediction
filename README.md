# Clinical Neural Network Playground

A compact, production-minded machine learning workspace for **clinical outcome prediction** using neural networks.

This repository currently includes two end-to-end pipelines:
- **Diabetes outcome classification** (`Outcome`) from tabular clinical features
- **WHO performance status prediction** (`who_performance_status`) from GBM clinical records

Both workflows are designed to be practical, reproducible, and easy to extend.

## Why This Project Looks Serious

- Clean, script-based training pipelines (no notebook lock-in)
- Robust preprocessing for real-world clinical tables (missing values, scaling, categorical handling)
- Neural-network baselines with reproducible train/test split
- Artifact-first workflow: trained models + exported evaluation figures
- Lightweight structure you can adapt for research demos, class projects, or prototyping

## Repository Structure

```text
.
├── cancer/
│   ├── CFB-GBM_clinical_data_v02_20260129.tsv
│   ├── train_who_nn.py
│   └── who_performance_status_nn.joblib
├── diabetes/
│   ├── diabetes.csv
│   ├── train_outcome_nn.py
│   ├── diabetes_outcome_nn.joblib
│   ├── nn_loss_curve.png
│   ├── nn_confusion_matrix.png
│   └── nn_roc_curve.png
└── README.md
```

## Quick Start

### 1. Create and activate environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install pandas scikit-learn matplotlib joblib
```

## Training Workflows

### A) Diabetes Neural Network

Train a minimal NN classifier for `Outcome` and save model + plots in `diabetes/`:

```bash
python diabetes/train_outcome_nn.py --data diabetes/diabetes.csv
```

Generated outputs:
- `diabetes/diabetes_outcome_nn.joblib`
- `diabetes/nn_loss_curve.png`
- `diabetes/nn_confusion_matrix.png`
- `diabetes/nn_roc_curve.png`

### B) GBM WHO Performance Status Neural Network

Train classifier for `who_performance_status`:

```bash
python cancer/train_who_nn.py --data cancer/CFB-GBM_clinical_data_v02_20260129.tsv
```

Generated output:
- `cancer/who_performance_status_nn.joblib`

## Modeling Notes

- **Diabetes pipeline** includes handling for clinically invalid zeros in key measurements by treating them as missing values.
- Uses a scaled-input MLP with regularization and early stopping strategy for better convergence behavior.
- Evaluation includes classification metrics and visual diagnostics (loss, confusion matrix, ROC).

## Example Use Cases

- Baseline clinical risk stratification
- Rapid benchmarking of NN performance on tabular medical data
- Teaching/demo repository for preprocessing + MLP workflows

## Next-Level Extensions

- Cross-validation and confidence intervals
- Hyperparameter search (Optuna / GridSearchCV)
- Calibration curves and threshold optimization
- Explainability layer (SHAP / permutation importance)
- CI checks for training script health

## Disclaimer

This repository is for **research and educational use**. It is **not** a medical device and should not be used for clinical decision-making.
