from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache")))
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_model(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "nn",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate="adaptive",
                    learning_rate_init=5e-4,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    max_iter=1500,
                    random_state=random_state,
                ),
            ),
        ]
    )


def save_loss_curve(model: Pipeline, output_dir: Path) -> None:
    nn = model.named_steps["nn"]
    plt.figure(figsize=(7, 4))
    plt.plot(nn.loss_curve_, linewidth=2)
    plt.title("NN Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "nn_loss_curve.png", dpi=160)
    plt.close()


def save_confusion_matrix(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    plt.figure(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "nn_confusion_matrix.png", dpi=160)
    plt.close()


def save_roc_curve(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "nn_roc_curve.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural network model for diabetes Outcome prediction.")
    parser.add_argument("--data", type=Path, default=Path("diabetes/diabetes.csv"), help="Path to diabetes CSV file")
    parser.add_argument("--target", type=str, default="Outcome", help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows for test split")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Path for saved model file (.joblib). Defaults to same folder as data.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.data}")

    # In Pima diabetes data, 0 is invalid for these measurements and indicates missing.
    zero_as_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    existing_cols = [c for c in zero_as_missing_cols if c in df.columns]
    if existing_cols:
        df[existing_cols] = df[existing_cols].replace(0, np.nan)

    X = df.drop(columns=[args.target])
    # Ensure numeric features stay numeric after missing-value replacement.
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[args.target], errors="coerce")
    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_model(random_state=args.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    output_dir = args.data.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model_out = args.model_out if args.model_out is not None else output_dir / "diabetes_outcome_nn.joblib"
    joblib.dump(model, model_out)

    save_loss_curve(model, output_dir)
    save_confusion_matrix(model, X_test, y_test, output_dir)
    save_roc_curve(model, X_test, y_test, output_dir)

    print(f"Accuracy: {acc:.4f}")
    print(f"NN epochs run: {len(model.named_steps['nn'].loss_curve_)}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print(f"\nModel saved to: {model_out}")
    print("Saved plots:")
    print(f"- {output_dir / 'nn_loss_curve.png'}")
    print(f"- {output_dir / 'nn_confusion_matrix.png'}")
    print(f"- {output_dir / 'nn_roc_curve.png'}")


if __name__ == "__main__":
    main()
