#!/usr/bin/env python3
"""Run lightweight source checks for the FinancialSurvey package."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> int:
    print("+ " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT, text=True)
    return result.returncode


def main() -> None:
    checks = [
        [
            sys.executable,
            "-m",
            "py_compile",
            "scripts/check_dataset_files.py",
            "scripts/run_classical_baselines.py",
            "scripts/download_ibm_aml.py",
            "scripts/build_benchmark_database.py",
            "scripts/run_benchmark_suite.py",
            "scripts/run_enhanced_privacy_benchmarks.py",
            "scripts/profile_dataset.py",
            "scripts/self_check.py",
            "src/financial_survey/env_config.py",
            "src/financial_survey/dataset_profile.py",
        ],
    ]

    failures = 0
    for command in checks:
        failures += int(run(command) != 0)

    if failures:
        raise SystemExit(f"{failures} check(s) failed")

    print("Self-check completed.")


if __name__ == "__main__":
    main()
