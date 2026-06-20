#!/usr/bin/env python3
"""Profile a configured benchmark dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_survey.dataset_profile import profile_csv, save_profile  # noqa: E402
from financial_survey.env_config import BenchmarkEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=str(ROOT / "local.config"), help="Path to the local config file")
    parser.add_argument(
        "--dataset",
        default="ibm_aml",
        choices=["ibm_aml", "ulb_creditcard", "uci_default_credit", "paysim"],
    )
    parser.add_argument("--csv", default="", help="Override CSV path")
    parser.add_argument("--target", default="", help="Override target column")
    parser.add_argument("--sample", type=int, default=-1, help="Override BENCHMARK_SAMPLE_ROWS")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = BenchmarkEnv.from_file(args.env, ROOT)
    csv_path = Path(args.csv).expanduser() if args.csv else env.dataset_path(args.dataset)
    if csv_path is None:
        raise SystemExit(f"No CSV configured for dataset '{args.dataset}'. Set it in the local config file or pass --csv.")

    sample_rows = env.sample_rows if args.sample < 0 else args.sample
    try:
        profile = profile_csv(
            csv_path,
            dataset=args.dataset,
            target=args.target or None,
            sample_rows=sample_rows,
            random_seed=env.random_seed,
        )
    except FileNotFoundError:
        raise SystemExit(f"CSV not found: {csv_path}")
    print(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))

    output_path = Path(args.output).expanduser() if args.output else env.output_dir / "profiles" / f"{args.dataset}_profile.json"
    save_profile(profile, output_path)
    print(f"Saved profile: {output_path}")


if __name__ == "__main__":
    main()
