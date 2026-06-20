#!/usr/bin/env python3
"""Build a SQLite benchmark database from configured CSV datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_survey.env_config import BenchmarkEnv  # noqa: E402


IBM_AML_COLUMNS = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=str(ROOT / "local.config"))
    parser.add_argument("--dataset", choices=["ibm_aml"], default="ibm_aml")
    parser.add_argument("--csv", default="", help="Override source CSV path")
    parser.add_argument("--sqlite", default="", help="Output SQLite DB path")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--if-exists", choices=["replace", "append"], default="replace")
    return parser.parse_args()


def write_metadata(conn: sqlite3.Connection, metadata: dict[str, Any]) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_metadata (
            dataset TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (dataset, key)
        )
        """
    )
    for key, value in metadata.items():
        conn.execute(
            """
            INSERT OR REPLACE INTO dataset_metadata(dataset, key, value)
            VALUES (?, ?, ?)
            """,
            (metadata["dataset"], key, json.dumps(value, ensure_ascii=False)),
        )


def build_ibm_aml(csv_path: Path, sqlite_path: Path, chunksize: int, if_exists: str) -> dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = 0
    positive_count = 0

    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")

        if if_exists == "replace":
            conn.execute("DROP TABLE IF EXISTS ibm_aml_transactions")

        for chunk_index, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
            chunk = chunk.rename(columns=IBM_AML_COLUMNS)
            missing = [name for name in IBM_AML_COLUMNS.values() if name not in chunk.columns]
            if missing:
                raise SystemExit(f"Missing expected IBM AML columns: {missing}")

            chunk.insert(0, "row_id", range(rows + 1, rows + len(chunk) + 1))
            positive_count += int(chunk["is_laundering"].sum())
            rows += len(chunk)
            chunk.to_sql(
                "ibm_aml_transactions",
                conn,
                if_exists="append" if chunk_index or if_exists == "append" else "replace",
                index=False,
            )
            print(f"ingested_rows={rows}", flush=True)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_ibm_aml_target ON ibm_aml_transactions(is_laundering)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ibm_aml_timestamp ON ibm_aml_transactions(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ibm_aml_from_bank ON ibm_aml_transactions(from_bank)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ibm_aml_to_bank ON ibm_aml_transactions(to_bank)")

        elapsed = time.perf_counter() - started
        metadata = {
            "dataset": "ibm_aml",
            "source_csv": str(csv_path),
            "sqlite_path": str(sqlite_path),
            "table": "ibm_aml_transactions",
            "rows": rows,
            "positive_count": positive_count,
            "positive_rate": positive_count / rows if rows else None,
            "elapsed_seconds": elapsed,
        }
        write_metadata(conn, metadata)
        conn.commit()

    return metadata


def main() -> None:
    args = parse_args()
    env = BenchmarkEnv.from_file(args.env, ROOT)
    csv_path = Path(args.csv).expanduser() if args.csv else env.dataset_path(args.dataset)
    if csv_path is None:
        raise SystemExit(f"No CSV configured for dataset '{args.dataset}'.")

    sqlite_path = Path(args.sqlite).expanduser() if args.sqlite else env.root / "data" / "financial_survey.sqlite"
    metadata = build_ibm_aml(csv_path, sqlite_path, args.chunksize, args.if_exists)

    output_path = env.output_dir / "database" / "ibm_aml_sqlite_build.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": metadata, "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
