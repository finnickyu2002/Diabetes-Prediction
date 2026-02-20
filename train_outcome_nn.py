import argparse
import os
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache")))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = "Outcome"
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def make_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "nn",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    learning_rate="adaptive",
                    learning_rate_init=5e-4,
                    alpha=1e-3,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    max_iter=1500,
                    random_state=seed,
                ),
            ),
        ]
    )


def load_xy(csv_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    cols = [c for c in ZERO_IS_MISSING if c in df.columns]
    if cols:
        df[cols] = df[cols].replace(0, np.nan)

    X = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")

    keep = y.notna()
    return X.loc[keep], y.loc[keep].astype(int)


def save_plots(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, out_dir: Path) -> None:
    nn = model.named_steps["nn"]

    plt.figure(figsize=(7, 4))
    plt.plot(nn.loss_curve_, lw=2)
    plt.title("NN Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(out_dir / "nn_loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "nn_confusion_matrix.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_dir / "nn_roc_curve.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("diabetes/diabetes.csv"))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=Path, default=None)
    args = parser.parse_args()

    X, y = load_xy(args.data, args.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    model = make_model(args.seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    out_dir = args.data.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_out or (out_dir / "diabetes_outcome_nn.joblib")
    joblib.dump(model, model_path)
    save_plots(model, X_test, y_test, out_dir)

    print(f"accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"epochs: {len(model.named_steps['nn'].loss_curve_)}")
    print(classification_report(y_test, pred))
    print(f"model: {model_path}")
    print(f"plots: {out_dir / 'nn_loss_curve.png'}, {out_dir / 'nn_confusion_matrix.png'}, {out_dir / 'nn_roc_curve.png'}")


if __name__ == "__main__":
    main()
