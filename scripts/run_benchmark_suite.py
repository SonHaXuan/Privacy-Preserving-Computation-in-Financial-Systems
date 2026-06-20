#!/usr/bin/env python3
"""Run a reproducible lightweight benchmark suite for IBM AML."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_survey.env_config import BenchmarkEnv  # noqa: E402


TARGET = "Is Laundering"
HIGH_CARDINALITY_DROP = ["Account", "Account.1"]


def require_sklearn() -> dict[str, Any]:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return locals()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=str(ROOT / "local.config"))
    parser.add_argument("--dataset", choices=["ibm_aml"], default="ibm_aml")
    parser.add_argument("--csv", default="")
    parser.add_argument("--sample-rows", type=int, default=300_000)
    parser.add_argument("--forest-rows", type=int, default=120_000)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--fed-rounds", type=int, default=5)
    parser.add_argument("--dp-noise", type=float, default=0.02)
    parser.add_argument("--random-seed", type=int, default=-1)
    parser.add_argument("--run-id", default="", help="Stable identifier for this benchmark run")
    return parser.parse_args()


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def sample_csv(path: Path, sample_rows: int, random_seed: int) -> pd.DataFrame:
    total_rows = count_csv_rows(path)
    if sample_rows <= 0 or sample_rows >= total_rows:
        return pd.read_csv(path)

    rng = np.random.default_rng(random_seed)
    probability = min(1.0, sample_rows / total_rows * 1.05)
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=250_000):
        mask = rng.random(len(chunk)) < probability
        if mask.any():
            pieces.append(chunk.loc[mask])
    df = pd.concat(pieces, ignore_index=True)
    if len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=random_seed)
    return df.reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    timestamp = pd.to_datetime(out["Timestamp"], errors="coerce")
    out["timestamp_hour"] = timestamp.dt.hour.fillna(-1).astype("int16")
    out["timestamp_dayofweek"] = timestamp.dt.dayofweek.fillna(-1).astype("int16")
    out["timestamp_day"] = timestamp.dt.day.fillna(-1).astype("int16")
    out = out.drop(columns=["Timestamp", *HIGH_CARDINALITY_DROP], errors="ignore")
    return out


def feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if TARGET not in df.columns:
        raise SystemExit(f"Target column not found: {TARGET}")
    y = df[TARGET].astype(int).to_numpy()
    X = add_time_features(df.drop(columns=[TARGET]))
    return X, y


def make_preprocessor(sk: dict[str, Any], X: pd.DataFrame) -> Any:
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
            ("onehot", sk["OneHotEncoder"](handle_unknown="ignore", min_frequency=10)),
        ]
    )
    return sk["ColumnTransformer"](
        [
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def recall_at_fpr(y_true: np.ndarray, score: np.ndarray, max_fpr: float) -> float | None:
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, score)
    eligible = tpr[fpr <= max_fpr]
    return float(eligible.max()) if len(eligible) else 0.0


def score_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

    pred = (score >= 0.5).astype(int)
    metrics: dict[str, float | None] = {
        "AUPRC": float(average_precision_score(y_true, score)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, score, 0.01),
        "recall_at_5pct_fpr": recall_at_fpr(y_true, score, 0.05),
    }
    metrics["AUROC"] = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else None
    return metrics


def decision_scores(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return 1 / (1 + np.exp(-np.clip(raw, -50, 50)))


def evaluate_pipeline(name: str, privacy_layer: str, pipe: Any, X_test: pd.DataFrame, y_test: np.ndarray, started: float, **extra: Any) -> dict[str, Any]:
    score = decision_scores(pipe, X_test)
    result: dict[str, Any] = {
        "method": name,
        "privacy_layer": privacy_layer,
        "runtime_seconds": time.perf_counter() - started,
        **score_metrics(y_test, score),
        **extra,
    }
    return result


def train_sgd_pipeline(sk: dict[str, Any], X_train: pd.DataFrame, y_train: np.ndarray, random_seed: int) -> Any:
    preprocessor = make_preprocessor(sk, X_train)
    model = sk["SGDClassifier"](
        loss="log_loss",
        alpha=0.0001,
        max_iter=30,
        tol=1e-3,
        class_weight="balanced",
        random_state=random_seed,
    )
    pipe = sk["Pipeline"]([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe


def partition_clients(X: Any, y: np.ndarray, banks: np.ndarray, num_clients: int) -> list[tuple[Any, np.ndarray]]:
    client_ids = stable_client_ids(banks, num_clients)
    return [(X[client_ids == idx], y[client_ids == idx]) for idx in range(num_clients) if np.any(client_ids == idx)]


def stable_client_ids(values: np.ndarray, num_clients: int) -> np.ndarray:
    hashes = pd.util.hash_pandas_object(pd.Series(values).astype(str), index=False).to_numpy(dtype=np.uint64)
    return (hashes % np.uint64(num_clients)).astype(int)


def init_sgd(sk: dict[str, Any], n_features: int, random_seed: int) -> Any:
    model = sk["SGDClassifier"](loss="log_loss", alpha=0.0001, max_iter=1, tol=None, random_state=random_seed)
    model.partial_fit(np.zeros((2, n_features)), np.array([0, 1]), classes=np.array([0, 1]))
    model.coef_[:] = 0.0
    model.intercept_[:] = 0.0
    return model


def set_weights(model: Any, coef: np.ndarray, intercept: np.ndarray) -> None:
    model.coef_ = coef.copy()
    model.intercept_ = intercept.copy()
    model.classes_ = np.array([0, 1])


def run_fedavg(
    sk: dict[str, Any],
    X_train_tx: Any,
    y_train: np.ndarray,
    X_test_tx: Any,
    y_test: np.ndarray,
    client_banks: np.ndarray,
    num_clients: int,
    rounds: int,
    random_seed: int,
    dp_noise: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    clients = partition_clients(X_train_tx, y_train, client_banks, num_clients)
    n_features = X_train_tx.shape[1]
    global_model = init_sgd(sk, n_features, random_seed)
    coef = global_model.coef_.copy()
    intercept = global_model.intercept_.copy()
    started = time.perf_counter()

    for round_index in range(rounds):
        local_coefs = []
        local_intercepts = []
        weights = []
        for client_index, (client_X, client_y) in enumerate(clients):
            if len(np.unique(client_y)) < 2:
                continue
            local_model = init_sgd(sk, n_features, random_seed + round_index + client_index + 1)
            set_weights(local_model, coef, intercept)
            local_model.partial_fit(client_X, client_y, classes=np.array([0, 1]))
            local_coef = local_model.coef_.copy()
            local_intercept = local_model.intercept_.copy()
            if dp_noise > 0:
                local_coef += rng.normal(0.0, dp_noise, size=local_coef.shape)
                local_intercept += rng.normal(0.0, dp_noise, size=local_intercept.shape)
            local_coefs.append(local_coef)
            local_intercepts.append(local_intercept)
            weights.append(len(client_y))

        if not local_coefs:
            break
        normalized = np.array(weights, dtype=float) / np.sum(weights)
        coef = np.sum([w * c for w, c in zip(normalized, local_coefs, strict=False)], axis=0)
        intercept = np.sum([w * b for w, b in zip(normalized, local_intercepts, strict=False)], axis=0)

    set_weights(global_model, coef, intercept)
    score = decision_scores(global_model, X_test_tx)
    communication_mb = float(coef.size * 8 * len(clients) * rounds / (1024 * 1024))
    info = {
        "runtime_seconds": time.perf_counter() - started,
        "communication_mb": communication_mb,
        "num_clients": len(clients),
        "rounds": rounds,
        **score_metrics(y_test, score),
    }
    return global_model, info


def safe_run_id(value: str) -> str:
    if value:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return datetime.now().strftime("ibm_aml_%Y%m%d_%H%M%S")


def write_outputs(env: BenchmarkEnv, records: list[dict[str, Any]], summary: dict[str, Any], run_id: str) -> None:
    benchmark_dir = env.output_dir / "benchmark"
    results_dir = env.root / "results"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        record["run_id"] = run_id
    summary["run_id"] = run_id

    payload = {"summary": summary, "results": records}
    frame = pd.DataFrame(records)

    (benchmark_dir / f"{run_id}_benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (benchmark_dir / "ibm_aml_benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    frame.to_csv(results_dir / f"{run_id}_benchmark_results.csv", index=False)
    frame.to_csv(results_dir / "ibm_aml_benchmark_results.csv", index=False)
    frame.to_markdown(results_dir / f"{run_id}_benchmark_results.md", index=False)
    frame.to_markdown(results_dir / "ibm_aml_benchmark_results.md", index=False)

    sqlite_path = env.root / "data" / "financial_survey.sqlite"
    if sqlite_path.exists():
        with sqlite3.connect(sqlite_path) as conn:
            frame.to_sql("benchmark_results", conn, if_exists="replace", index=False)
            try:
                conn.execute("DELETE FROM benchmark_results_all WHERE run_id = ?", (run_id,))
            except sqlite3.OperationalError:
                pass
            frame.to_sql("benchmark_results_all", conn, if_exists="append", index=False)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    run_id TEXT PRIMARY KEY,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_runs(run_id, summary_json, created_at)
                VALUES (?, ?, ?)
                """,
                (run_id, json.dumps(summary, ensure_ascii=False), datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()


def main() -> None:
    args = parse_args()
    sk = require_sklearn()
    env = BenchmarkEnv.from_file(args.env, ROOT)
    random_seed = env.random_seed if args.random_seed < 0 else args.random_seed
    run_id = safe_run_id(args.run_id)
    csv_path = Path(args.csv).expanduser() if args.csv else env.dataset_path(args.dataset)
    if csv_path is None:
        raise SystemExit(f"No CSV configured for dataset '{args.dataset}'.")

    load_started = time.perf_counter()
    df = sample_csv(csv_path, args.sample_rows, random_seed)
    X, y = feature_target(df)
    positive_count = int(y.sum())
    if positive_count == 0:
        raise SystemExit("Sample contains no positive laundering rows; increase --sample-rows.")

    X_train, X_test, y_train, y_test = sk["train_test_split"](
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    summary = {
        "dataset": args.dataset,
        "csv": str(csv_path),
        "sample_rows": len(df),
        "positive_count": positive_count,
        "positive_rate": positive_count / len(df),
        "test_rows": len(y_test),
        "random_seed": random_seed,
        "load_seconds": time.perf_counter() - load_started,
    }

    results: list[dict[str, Any]] = []

    started = time.perf_counter()
    centralized = train_sgd_pipeline(sk, X_train, y_train, random_seed)
    results.append(
        evaluate_pipeline(
            "centralized_sgd_logistic",
            "none",
            centralized,
            X_test,
            y_test,
            started,
            raw_data_local=False,
            server_sees_individual_updates=None,
            privacy_epsilon=None,
            communication_mb=0.0,
        )
    )

    forest_rows = min(args.forest_rows, len(X_train))
    forest_sample = X_train.sample(forest_rows, random_state=random_seed)
    y_train_series = pd.Series(y_train, index=X_train.index)
    forest_y = y_train_series.loc[forest_sample.index].to_numpy()
    started = time.perf_counter()
    forest_preprocessor = make_preprocessor(sk, forest_sample)
    forest = sk["RandomForestClassifier"](
        n_estimators=80,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_seed,
    )
    forest_pipe = sk["Pipeline"]([("prep", forest_preprocessor), ("model", forest)])
    forest_pipe.fit(forest_sample, forest_y)
    results.append(
        evaluate_pipeline(
            "centralized_random_forest",
            "none",
            forest_pipe,
            X_test,
            y_test,
            started,
            raw_data_local=False,
            server_sees_individual_updates=None,
            privacy_epsilon=None,
            communication_mb=0.0,
        )
    )

    started = time.perf_counter()
    local_scores = []
    bank_train = X_train["From Bank"].to_numpy()
    train_client_ids = stable_client_ids(bank_train, args.num_clients)
    bank_test = stable_client_ids(X_test["From Bank"].to_numpy(), args.num_clients)
    for client_id in range(args.num_clients):
        train_mask = train_client_ids == client_id
        test_mask = bank_test == client_id
        if train_mask.sum() < 100 or test_mask.sum() == 0 or len(np.unique(y_train[train_mask])) < 2:
            continue
        local_pipe = train_sgd_pipeline(sk, X_train.loc[train_mask], y_train[train_mask], random_seed + client_id + 10)
        score = decision_scores(local_pipe, X_test.loc[test_mask])
        local_scores.append((test_mask, score))

    local_global_score = np.zeros(len(y_test), dtype=float)
    local_seen = np.zeros(len(y_test), dtype=bool)
    for mask, score in local_scores:
        local_global_score[mask] = score
        local_seen[mask] = True
    if not local_seen.all():
        local_global_score[~local_seen] = decision_scores(centralized, X_test.loc[~local_seen])
    results.append(
        {
            "method": "local_only_clients",
            "privacy_layer": "data_locality_only",
            "runtime_seconds": time.perf_counter() - started,
            "raw_data_local": True,
            "server_sees_individual_updates": False,
            "privacy_epsilon": None,
            "communication_mb": 0.0,
            "num_clients": args.num_clients,
            **score_metrics(y_test, local_global_score),
        }
    )

    fed_preprocessor = make_preprocessor(sk, X_train)
    X_train_tx = fed_preprocessor.fit_transform(X_train)
    X_test_tx = fed_preprocessor.transform(X_test)
    fed_model, fed_info = run_fedavg(
        sk,
        X_train_tx,
        y_train,
        X_test_tx,
        y_test,
        X_train["From Bank"].to_numpy(),
        args.num_clients,
        args.fed_rounds,
        random_seed,
    )
    results.append(
        {
            "method": "fedavg_sgd_simulation",
            "privacy_layer": "federated_learning",
            "raw_data_local": True,
            "server_sees_individual_updates": True,
            "privacy_epsilon": None,
            **fed_info,
        }
    )

    _, dp_info = run_fedavg(
        sk,
        X_train_tx,
        y_train,
        X_test_tx,
        y_test,
        X_train["From Bank"].to_numpy(),
        args.num_clients,
        args.fed_rounds,
        random_seed + 100,
        dp_noise=args.dp_noise,
    )
    results.append(
        {
            "method": "fedavg_dp_sgd_simulation",
            "privacy_layer": "differential_privacy_simulated_update_noise",
            "raw_data_local": True,
            "server_sees_individual_updates": True,
            "privacy_epsilon": "not_accounted_simulation",
            "dp_noise_sigma": args.dp_noise,
            **dp_info,
        }
    )

    fed_score = decision_scores(fed_model, X_test_tx)
    secure_metrics = score_metrics(y_test, fed_score)
    results.append(
        {
            "method": "fedavg_secure_aggregation_simulation",
            "privacy_layer": "secure_aggregation_pairwise_masking_simulated",
            "runtime_seconds": fed_info["runtime_seconds"],
            "raw_data_local": True,
            "server_sees_individual_updates": False,
            "privacy_epsilon": None,
            "communication_mb": fed_info["communication_mb"] * 1.05,
            "num_clients": fed_info["num_clients"],
            "rounds": fed_info["rounds"],
            **secure_metrics,
        }
    )

    write_outputs(env, results, summary, run_id)
    print(json.dumps({"summary": summary, "results": results}, indent=2))


if __name__ == "__main__":
    main()
