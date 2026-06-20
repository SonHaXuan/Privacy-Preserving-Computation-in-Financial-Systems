#!/usr/bin/env python3
"""Run enhanced privacy-preserving benchmark measurements on IBM AML.

This script complements the lightweight sklearn benchmark with:

- PyTorch FedAvg over client partitions.
- FedAvg with RDP-accounted noisy clipped updates.
- FedAvg with pairwise-mask secure aggregation simulation.
- A pure-Python Paillier homomorphic aggregation microbenchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from financial_survey.env_config import BenchmarkEnv  # noqa: E402
from run_benchmark_suite import feature_target, make_preprocessor, sample_csv, score_metrics, stable_client_ids  # noqa: E402


def require_torch() -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return {"torch": torch, "nn": nn, "DataLoader": DataLoader, "TensorDataset": TensorDataset}


def safe_run_id(value: str) -> str:
    if value:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return datetime.now().strftime("enhanced_%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=str(ROOT / "local.config"))
    parser.add_argument("--dataset", choices=["ibm_aml"], default="ibm_aml")
    parser.add_argument("--csv", default="")
    parser.add_argument("--sample-rows", type=int, default=500_000)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--fed-rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--he-vector-size", type=int, default=64)
    parser.add_argument("--he-key-bits", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=-1)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def to_dense_float32(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


class LogisticModel:
    """Small wrapper around torch.nn.Linear to keep imports lazy."""

    def __init__(self, torch: Any, nn: Any, n_features: int) -> None:
        self.torch = torch
        self.module = nn.Linear(n_features, 1)

    def state_vector(self) -> np.ndarray:
        parts = []
        for tensor in self.module.state_dict().values():
            parts.append(tensor.detach().cpu().numpy().reshape(-1))
        return np.concatenate(parts).astype(np.float64)

    def load_state_vector(self, vector: np.ndarray) -> None:
        state = self.module.state_dict()
        offset = 0
        new_state = {}
        for key, tensor in state.items():
            size = tensor.numel()
            shaped = vector[offset : offset + size].reshape(tuple(tensor.shape))
            new_state[key] = self.torch.tensor(shaped, dtype=tensor.dtype)
            offset += size
        self.module.load_state_dict(new_state)


def train_local_model(
    torch_mods: dict[str, Any],
    global_vector: np.ndarray,
    n_features: int,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pos_weight: float,
    seed: int,
) -> np.ndarray:
    torch = torch_mods["torch"]
    nn = torch_mods["nn"]
    DataLoader = torch_mods["DataLoader"]
    TensorDataset = torch_mods["TensorDataset"]

    torch.manual_seed(seed)
    model = LogisticModel(torch, nn, n_features)
    model.load_state_vector(global_vector)
    optimizer = torch.optim.SGD(model.module.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.module.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model.module(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return model.state_vector()


def predict_scores(torch_mods: dict[str, Any], vector: np.ndarray, n_features: int, X: np.ndarray) -> np.ndarray:
    torch = torch_mods["torch"]
    nn = torch_mods["nn"]
    model = LogisticModel(torch, nn, n_features)
    model.load_state_vector(vector)
    model.module.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), 65536):
            logits = model.module(torch.tensor(X[start : start + 65536], dtype=torch.float32))
            scores.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
    return np.concatenate(scores)


def client_slices(client_ids: np.ndarray, num_clients: int) -> list[np.ndarray]:
    return [np.flatnonzero(client_ids == idx) for idx in range(num_clients) if np.any(client_ids == idx)]


def weighted_delta(client_vectors: list[np.ndarray], global_vector: np.ndarray, client_sizes: list[int]) -> np.ndarray:
    total = float(sum(client_sizes))
    aggregate = np.zeros_like(global_vector, dtype=np.float64)
    for vector, size in zip(client_vectors, client_sizes, strict=False):
        aggregate += (size / total) * (vector - global_vector)
    return aggregate


def clip_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm == 0:
        return vector
    return vector * (max_norm / norm)


def rdp_epsilon(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float | str:
    try:
        from opacus.accountants import RDPAccountant

        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        return float(accountant.get_epsilon(delta=delta))
    except Exception as exc:  # pragma: no cover - depends on opacus internals
        return f"accounting_failed:{type(exc).__name__}:{exc}"


def pairwise_masked_aggregate(
    weighted_updates: list[np.ndarray],
    *,
    round_index: int,
    random_seed: int,
    mask_scale: float = 10.0,
) -> tuple[np.ndarray, float]:
    masked = [update.copy() for update in weighted_updates]
    for i in range(len(masked)):
        for j in range(i + 1, len(masked)):
            seed = random_seed + 100000 * (round_index + 1) + 1000 * i + j
            rng = np.random.default_rng(seed)
            mask = rng.normal(0.0, mask_scale, size=masked[i].shape)
            masked[i] += mask
            masked[j] -= mask
    raw_sum = np.sum(weighted_updates, axis=0)
    masked_sum = np.sum(masked, axis=0)
    return masked_sum, float(np.max(np.abs(masked_sum - raw_sum)))


def run_fedavg_variants(
    torch_mods: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_client_ids: np.ndarray,
    *,
    num_clients: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    clip_norm: float,
    noise_multiplier: float,
    delta: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    torch = torch_mods["torch"]
    nn = torch_mods["nn"]
    n_features = X_train.shape[1]
    initial = LogisticModel(torch, nn, n_features).state_vector()
    clients = client_slices(train_client_ids, num_clients)
    pos_weight = max(float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1)), 1.0)

    results: list[dict[str, Any]] = []
    final_vectors: dict[str, np.ndarray] = {}
    secure_max_error = 0.0

    for variant in ["fedavg_torch", "fedavg_dp_rdp", "fedavg_secure_pairwise_masking"]:
        vector = initial.copy()
        started = time.perf_counter()
        communication_mb = 0.0
        secure_errors: list[float] = []
        rng = np.random.default_rng(random_seed + 999)

        for round_index in range(rounds):
            local_vectors = []
            client_sizes = []
            for client_index, indices in enumerate(clients):
                if len(indices) < 2 or len(np.unique(y_train[indices])) < 2:
                    continue
                local_vector = train_local_model(
                    torch_mods,
                    vector,
                    n_features,
                    X_train[indices],
                    y_train[indices],
                    epochs=local_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    pos_weight=pos_weight,
                    seed=random_seed + 10_000 * round_index + client_index,
                )
                local_vectors.append(local_vector)
                client_sizes.append(len(indices))

            if not local_vectors:
                break

            total = float(sum(client_sizes))
            weighted_updates = []
            for local_vector, size in zip(local_vectors, client_sizes, strict=False):
                delta_vector = local_vector - vector
                if variant == "fedavg_dp_rdp":
                    delta_vector = clip_vector(delta_vector, clip_norm)
                    delta_vector += rng.normal(0.0, noise_multiplier * clip_norm / max(len(local_vectors), 1), size=delta_vector.shape)
                weighted_updates.append((size / total) * delta_vector)

            if variant == "fedavg_secure_pairwise_masking":
                aggregate_delta, max_error = pairwise_masked_aggregate(
                    weighted_updates,
                    round_index=round_index,
                    random_seed=random_seed,
                )
                secure_errors.append(max_error)
                communication_mb += float(sum(update.nbytes for update in weighted_updates) * 1.05 / (1024 * 1024))
            else:
                aggregate_delta = np.sum(weighted_updates, axis=0)
                communication_mb += float(sum(update.nbytes for update in weighted_updates) / (1024 * 1024))
            vector = vector + aggregate_delta

        scores = predict_scores(torch_mods, vector, n_features, X_test)
        metrics = score_metrics(y_test, scores)
        record: dict[str, Any] = {
            "method": variant,
            "runtime_seconds": time.perf_counter() - started,
            "num_clients": len(clients),
            "rounds": rounds,
            "local_epochs": local_epochs,
            "communication_mb": communication_mb,
            **metrics,
        }
        if variant == "fedavg_torch":
            record.update(
                {
                    "privacy_layer": "federated_learning",
                    "raw_data_local": True,
                    "server_sees_individual_updates": True,
                    "privacy_epsilon": None,
                }
            )
        elif variant == "fedavg_dp_rdp":
            record.update(
                {
                    "privacy_layer": "rdp_accounted_noisy_clipped_updates",
                    "raw_data_local": True,
                    "server_sees_individual_updates": True,
                    "privacy_epsilon": rdp_epsilon(noise_multiplier, 1.0, rounds, delta),
                    "privacy_delta": delta,
                    "clip_norm": clip_norm,
                    "noise_multiplier": noise_multiplier,
                }
            )
        else:
            secure_max_error = max(secure_errors) if secure_errors else 0.0
            record.update(
                {
                    "privacy_layer": "pairwise_mask_secure_aggregation",
                    "raw_data_local": True,
                    "server_sees_individual_updates": False,
                    "privacy_epsilon": None,
                    "secure_aggregation_max_abs_error": secure_max_error,
                }
            )
        final_vectors[variant] = vector
        results.append(record)
    return results


def mod_inverse(value: int, modulus: int) -> int:
    return pow(value, -1, modulus)


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


@dataclass
class PaillierKeypair:
    n: int
    n2: int
    g: int
    lam: int
    mu: int


def generate_paillier_keypair(bits: int) -> PaillierKeypair:
    from sympy import randprime

    lower = 2 ** (bits // 2 - 1)
    upper = 2 ** (bits // 2)
    p = int(randprime(lower, upper))
    q = int(randprime(lower, upper))
    while p == q:
        q = int(randprime(lower, upper))
    n = p * q
    n2 = n * n
    g = n + 1
    lam = lcm(p - 1, q - 1)
    x = pow(g, lam, n2)
    l_value = (x - 1) // n
    mu = mod_inverse(l_value, n)
    return PaillierKeypair(n=n, n2=n2, g=g, lam=lam, mu=mu)


def paillier_encrypt(key: PaillierKeypair, message: int, r: int) -> int:
    return (pow(key.g, message % key.n, key.n2) * pow(r, key.n, key.n2)) % key.n2


def paillier_decrypt(key: PaillierKeypair, ciphertext: int) -> int:
    x = pow(ciphertext, key.lam, key.n2)
    l_value = (x - 1) // key.n
    message = (l_value * key.mu) % key.n
    if message > key.n // 2:
        message -= key.n
    return message


def run_paillier_microbenchmark(
    *,
    vector_size: int,
    num_clients: int,
    key_bits: int,
    random_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    updates = rng.normal(0.0, 0.01, size=(num_clients, vector_size))
    plaintext_sum = updates.sum(axis=0)
    scale = 1_000_000
    encoded = np.rint(updates * scale).astype(int)

    started = time.perf_counter()
    key = generate_paillier_keypair(key_bits)
    keygen_seconds = time.perf_counter() - started

    encryption_started = time.perf_counter()
    encrypted_sum = [1] * vector_size
    for row in encoded:
        for idx, value in enumerate(row):
            r = int(rng.integers(2, min(key.n - 1, 2**63 - 1)))
            while math.gcd(r, key.n) != 1:
                r = int(rng.integers(2, min(key.n - 1, 2**63 - 1)))
            encrypted = paillier_encrypt(key, int(value), r)
            encrypted_sum[idx] = (encrypted_sum[idx] * encrypted) % key.n2
    encryption_seconds = time.perf_counter() - encryption_started

    decryption_started = time.perf_counter()
    decrypted = np.array([paillier_decrypt(key, value) for value in encrypted_sum], dtype=float) / scale
    decryption_seconds = time.perf_counter() - decryption_started
    max_abs_error = float(np.max(np.abs(decrypted - plaintext_sum)))
    ciphertext_bytes = vector_size * num_clients * (key.n2.bit_length() // 8 + 1)

    return {
        "method": "paillier_he_aggregation_microbenchmark",
        "privacy_layer": "additive_homomorphic_encryption",
        "vector_size": vector_size,
        "num_clients": num_clients,
        "key_bits": key_bits,
        "keygen_seconds": keygen_seconds,
        "encryption_seconds": encryption_seconds,
        "decryption_seconds": decryption_seconds,
        "runtime_seconds": keygen_seconds + encryption_seconds + decryption_seconds,
        "communication_mb": ciphertext_bytes / (1024 * 1024),
        "max_abs_error": max_abs_error,
    }


def write_outputs(env: BenchmarkEnv, run_id: str, summary: dict[str, Any], records: list[dict[str, Any]], he_record: dict[str, Any]) -> None:
    output_dir = env.output_dir / "enhanced_privacy"
    results_dir = env.root / "results" / "enhanced_privacy"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        record["run_id"] = run_id
    he_record["run_id"] = run_id
    summary["run_id"] = run_id
    payload = {"summary": summary, "privacy_results": records, "he_microbenchmark": he_record}

    (output_dir / f"{run_id}_enhanced_privacy_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    privacy_frame = pd.DataFrame(records)
    he_frame = pd.DataFrame([he_record])
    privacy_frame.to_csv(results_dir / f"{run_id}_privacy_results.csv", index=False)
    privacy_frame.to_markdown(results_dir / f"{run_id}_privacy_results.md", index=False)
    he_frame.to_csv(results_dir / f"{run_id}_he_microbenchmark.csv", index=False)
    he_frame.to_markdown(results_dir / f"{run_id}_he_microbenchmark.md", index=False)

    sqlite_path = env.root / "data" / "financial_survey.sqlite"
    if sqlite_path.exists():
        with sqlite3.connect(sqlite_path) as conn:
            try:
                conn.execute("DELETE FROM enhanced_privacy_results WHERE run_id = ?", (run_id,))
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("DELETE FROM he_microbenchmark_results WHERE run_id = ?", (run_id,))
            except sqlite3.OperationalError:
                pass
            privacy_frame.to_sql("enhanced_privacy_results", conn, if_exists="append", index=False)
            he_frame.to_sql("he_microbenchmark_results", conn, if_exists="append", index=False)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS enhanced_privacy_runs (
                    run_id TEXT PRIMARY KEY,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO enhanced_privacy_runs(run_id, summary_json, created_at)
                VALUES (?, ?, ?)
                """,
                (run_id, json.dumps(summary, ensure_ascii=False), datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()


def main() -> None:
    args = parse_args()
    env = BenchmarkEnv.from_file(args.env, ROOT)
    random_seed = env.random_seed if args.random_seed < 0 else args.random_seed
    run_id = safe_run_id(args.run_id)
    csv_path = Path(args.csv).expanduser() if args.csv else env.dataset_path(args.dataset)
    if csv_path is None:
        raise SystemExit(f"No CSV configured for dataset '{args.dataset}'.")

    sk = __import__("run_benchmark_suite").require_sklearn()
    torch_mods = require_torch()

    started = time.perf_counter()
    df = sample_csv(csv_path, args.sample_rows, random_seed)
    X, y = feature_target(df)
    X_train, X_test, y_train, y_test = sk["train_test_split"](
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    preprocessor = make_preprocessor(sk, X_train)
    X_train_tx = to_dense_float32(preprocessor.fit_transform(X_train))
    X_test_tx = to_dense_float32(preprocessor.transform(X_test))
    client_ids = stable_client_ids(X_train["From Bank"].to_numpy(), args.num_clients)

    summary = {
        "dataset": args.dataset,
        "csv": str(csv_path),
        "sample_rows": len(df),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "test_rows": len(y_test),
        "num_clients": args.num_clients,
        "fed_rounds": args.fed_rounds,
        "local_epochs": args.local_epochs,
        "random_seed": random_seed,
        "feature_count_after_encoding": int(X_train_tx.shape[1]),
        "prep_seconds": time.perf_counter() - started,
    }

    privacy_records = run_fedavg_variants(
        torch_mods,
        X_train_tx,
        y_train,
        X_test_tx,
        y_test,
        client_ids,
        num_clients=args.num_clients,
        rounds=args.fed_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        delta=args.delta,
        random_seed=random_seed,
    )
    he_record = run_paillier_microbenchmark(
        vector_size=args.he_vector_size,
        num_clients=args.num_clients,
        key_bits=args.he_key_bits,
        random_seed=random_seed,
    )
    write_outputs(env, run_id, summary, privacy_records, he_record)
    print(json.dumps({"summary": summary, "privacy_results": privacy_records, "he_microbenchmark": he_record}, indent=2))


if __name__ == "__main__":
    main()
