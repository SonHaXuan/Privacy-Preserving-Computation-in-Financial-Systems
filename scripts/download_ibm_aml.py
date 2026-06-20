#!/usr/bin/env python3
"""Download and link the IBM AML Kaggle dataset on the execution host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_survey.env_config import BenchmarkEnv  # noqa: E402


DATASET_HANDLE = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
DEFAULT_FILES = [
    "HI-Small_Trans.csv",
    "HI-Small_Patterns.txt",
    "HI-Small_accounts.csv",
    "LI-Small_Trans.csv",
    "LI-Small_Patterns.txt",
    "LI-Small_accounts.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=str(ROOT / "local.config"))
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path, mode: str, force: bool) -> None:
    if destination.exists() or destination.is_symlink():
        if not force:
            return
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source)


def main() -> None:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("kagglehub is required. Install requirements-benchmark.txt on the execution host.") from exc

    args = parse_args()
    env = BenchmarkEnv.from_file(args.env, ROOT)
    dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))

    linked: list[dict[str, str | int]] = []
    for name in args.files:
        source = dataset_dir / name
        if not source.exists():
            raise SystemExit(f"Dataset file not found after download: {source}")
        destination = env.data_dir / name
        link_or_copy(source, destination, args.mode, args.force)
        linked.append(
            {
                "file": name,
                "source": str(source),
                "destination": str(destination),
                "bytes": source.stat().st_size,
                "mode": args.mode,
            }
        )

    manifest_path = env.output_dir / "dataset_downloads" / "ibm_aml_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"dataset": DATASET_HANDLE, "files": linked}, indent=2), encoding="utf-8")
    print(json.dumps({"dataset_dir": str(dataset_dir), "manifest": str(manifest_path), "files": linked}, indent=2))


if __name__ == "__main__":
    main()
