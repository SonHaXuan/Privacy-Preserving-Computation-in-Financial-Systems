#!/usr/bin/env python3
"""Check whether expected local dataset files are present.

This script does not download data. Kaggle/IBM/UCI terms should be handled
outside the repo, then files can be placed in FinancialSurvey/data/raw/.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED = {
    "ibm_aml": [
        "HI-Small_Trans.csv",
        "HI-Medium_Trans.csv",
        "LI-Small_Trans.csv",
        "LI-Medium_Trans.csv",
    ],
    "ulb_credit_card_fraud": ["creditcard.csv"],
    "uci_default_credit": ["default of credit card clients.xls", "default_credit_card_clients.csv"],
    "paysim": ["PS_20174392719_1491204439457_log.csv"],
    "ieee_cis": [
        "train_transaction.csv",
        "train_identity.csv",
        "test_transaction.csv",
        "test_identity.csv",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data" / "raw"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    print(f"Checking: {data_dir}")
    for dataset, files in EXPECTED.items():
        present = [name for name in files if (data_dir / name).exists()]
        status = "OK" if present else "MISSING"
        print(f"{dataset}: {status}")
        for name in files:
            marker = "found" if (data_dir / name).exists() else "not found"
            print(f"  - {name}: {marker}")


if __name__ == "__main__":
    main()

