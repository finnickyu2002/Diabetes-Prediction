# Clinical NN Baselines

Small repo for training simple neural-network baselines on two clinical tabular datasets.

## What's here

- `diabetes/`
  - `diabetes.csv`
  - `train_outcome_nn.py` (predicts `Outcome`)
  - outputs: model + loss/confusion/ROC plots
- `cancer/`
  - `CFB-GBM_clinical_data_v02_20260129.tsv`
  - `train_who_nn.py` (predicts `who_performance_status`)
  - output: trained model

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas scikit-learn matplotlib joblib
```

## Train

### Diabetes

```bash
python diabetes/train_outcome_nn.py --data diabetes/diabetes.csv
```

Expected files in `diabetes/`:
- `diabetes_outcome_nn.joblib`
- `nn_loss_curve.png`
- `nn_confusion_matrix.png`
- `nn_roc_curve.png`

### Cancer (WHO status)

```bash
python cancer/train_who_nn.py --data cancer/CFB-GBM_clinical_data_v02_20260129.tsv
```

Expected file in `cancer/`:
- `who_performance_status_nn.joblib`

## Notes

- These are baseline models, not optimized SOTA pipelines.
- Diabetes script treats some zero values as missing (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`).
- For plotting in restricted environments, the script uses a non-interactive Matplotlib backend.

## Disclaimer

Research/learning use only. Not for clinical decision-making.
