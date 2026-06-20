#!/usr/bin/env python3
"""Optional classical baseline runner for downloaded CSV datasets.

This is intentionally lightweight. Install scikit-learn before running:

    python -m pip install scikit-learn

The benchmark package does not vendor external datasets or ML libraries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to local CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--drop", nargs="*", default=[], help="Columns to drop before training")
    parser.add_argument("--sample", type=int, default=0, help="Optional row sample for quick tests")
    return parser.parse_args()


def require_sklearn():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError:
        print("scikit-learn is not installed. Install it before running this baseline script.", file=sys.stderr)
        raise
    return {
        "ColumnTransformer": ColumnTransformer,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "average_precision_score": average_precision_score,
        "roc_auc_score": roc_auc_score,
        "f1_score": f1_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


def main() -> None:
    sk = require_sklearn()
    args = parse_args()
    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=42)

    if args.target not in df.columns:
        raise SystemExit(f"Target column not found: {args.target}")

    y = df[args.target].astype(int)
    X = df.drop(columns=[args.target] + args.drop, errors="ignore")

    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipeline = sk["Pipeline"](
        [
            ("imputer", sk["SimpleImputer"](strategy="median")),
            ("scaler", sk["StandardScaler"]()),
        ]
    )
    categorical_pipeline = sk["Pipeline"](
        [
            ("imputer", sk["SimpleImputer"](strategy="most_frequent")),
            ("onehot", sk["OneHotEncoder"](handle_unknown="ignore")),
        ]
    )
    preprocessor = sk["ColumnTransformer"](
        [
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    X_train, X_test, y_train, y_test = sk["train_test_split"](
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": sk["LogisticRegression"](max_iter=1000, class_weight="balanced"),
        "random_forest": sk["RandomForestClassifier"](
            n_estimators=200, random_state=42, class_weight="balanced_subsample", n_jobs=-1
        ),
    }

    for name, model in models.items():
        pipe = sk["Pipeline"]([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        score = pipe.predict_proba(X_test)[:, 1]
        pred = (score >= 0.5).astype(int)
        print(name)
        print(f"  AUPRC: {sk['average_precision_score'](y_test, score):.6f}")
        print(f"  AUROC: {sk['roc_auc_score'](y_test, score):.6f}")
        print(f"  F1: {sk['f1_score'](y_test, pred):.6f}")
        print(f"  Balanced accuracy: {sk['balanced_accuracy_score'](y_test, pred):.6f}")


if __name__ == "__main__":
    main()

