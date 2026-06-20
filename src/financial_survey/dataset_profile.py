"""Dataset profiling helpers for benchmark handoff."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd


TARGET_CANDIDATES: dict[str, list[str]] = {
    "ibm_aml": ["Is Laundering", "is_laundering", "laundering", "Label", "label", "target"],
    "ulb_creditcard": ["Class", "class", "target"],
    "uci_default_credit": ["default payment next month", "Y", "target", "default"],
    "paysim": ["isFraud", "is_fraud", "target"],
}


@dataclass
class DatasetProfile:
    dataset: str
    path: str
    rows: int
    columns: int
    target: str | None
    positive_count: int | None
    positive_rate: float | None
    numeric_columns: int
    categorical_columns: int
    missing_cells: int
    memory_mb: float
    sample_rows_used: int | None
    columns_preview: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_target(dataset: str, columns: list[str]) -> str | None:
    normalized = {col.lower().strip(): col for col in columns}
    for candidate in TARGET_CANDIDATES.get(dataset, []):
        hit = normalized.get(candidate.lower().strip())
        if hit:
            return hit
    return None


def profile_csv(
    path: str | Path,
    dataset: str,
    target: str | None = None,
    sample_rows: int = 0,
    random_seed: int = 42,
) -> DatasetProfile:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    if sample_rows and sample_rows > 0:
        df = pd.read_csv(csv_path)
        if len(df) > sample_rows:
            df = df.sample(sample_rows, random_state=random_seed)
        sample_rows_used = len(df)
    else:
        df = pd.read_csv(csv_path)
        sample_rows_used = None

    target_name = target or infer_target(dataset, list(df.columns))
    positive_count: int | None = None
    positive_rate: float | None = None
    if target_name and target_name in df.columns:
        target_series = df[target_name]
        if pd.api.types.is_numeric_dtype(target_series) or target_series.dropna().isin([0, 1, "0", "1", False, True]).all():
            numeric_target = pd.to_numeric(target_series, errors="coerce")
            positive_count = int((numeric_target == 1).sum())
            positive_rate = float((numeric_target == 1).mean())

    numeric_columns = len(df.select_dtypes(include=[np.number]).columns)
    categorical_columns = df.shape[1] - numeric_columns
    missing_cells = int(df.isna().sum().sum())
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))

    return DatasetProfile(
        dataset=dataset,
        path=str(csv_path),
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        target=target_name,
        positive_count=positive_count,
        positive_rate=positive_rate,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        missing_cells=missing_cells,
        memory_mb=memory_mb,
        sample_rows_used=sample_rows_used,
        columns_preview=list(df.columns[:25]),
    )


def save_profile(profile: DatasetProfile, output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

