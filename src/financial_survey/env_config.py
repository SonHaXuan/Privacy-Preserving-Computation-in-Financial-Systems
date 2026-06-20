"""Minimal key-value reader for local benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def load_dotenv(path: str | Path | None) -> dict[str, str]:
    """Load simple KEY=VALUE lines from a local configuration file.

    Existing environment variables take precedence over file values.
    Quotes are stripped for convenience. Comments and empty lines are ignored.
    """
    values: dict[str, str] = {}
    if path is None:
        return values

    env_path = Path(path)
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = os.environ.get(key, value)
    return values


@dataclass(frozen=True)
class BenchmarkEnv:
    root: Path
    data_dir: Path
    output_dir: Path
    ibm_aml_csv: Path | None
    ulb_creditcard_csv: Path | None
    uci_default_credit_csv: Path | None
    paysim_csv: Path | None
    sample_rows: int
    random_seed: int
    num_clients: int
    device: str

    @classmethod
    def from_file(cls, env_file: str | Path | None, fallback_root: str | Path) -> "BenchmarkEnv":
        values = load_dotenv(env_file)
        root = Path(values.get("FINANCIAL_SURVEY_ROOT", str(fallback_root))).expanduser()
        data_dir = Path(values.get("FINANCIAL_SURVEY_DATA_DIR", str(root / "data" / "raw"))).expanduser()
        output_dir = Path(values.get("FINANCIAL_SURVEY_OUTPUT_DIR", str(root / "outputs"))).expanduser()

        def optional_path(key: str) -> Path | None:
            value = values.get(key, "").strip()
            return Path(value).expanduser() if value else None

        return cls(
            root=root,
            data_dir=data_dir,
            output_dir=output_dir,
            ibm_aml_csv=optional_path("IBM_AML_CSV"),
            ulb_creditcard_csv=optional_path("ULB_CREDITCARD_CSV"),
            uci_default_credit_csv=optional_path("UCI_DEFAULT_CREDIT_CSV"),
            paysim_csv=optional_path("PAYSIM_CSV"),
            sample_rows=int(values.get("BENCHMARK_SAMPLE_ROWS", "0") or 0),
            random_seed=int(values.get("BENCHMARK_RANDOM_SEED", "42") or 42),
            num_clients=int(values.get("BENCHMARK_NUM_CLIENTS", "10") or 10),
            device=values.get("BENCHMARK_DEVICE", "cpu") or "cpu",
        )

    def dataset_path(self, dataset: str) -> Path | None:
        mapping = {
            "ibm_aml": self.ibm_aml_csv,
            "ulb_creditcard": self.ulb_creditcard_csv,
            "uci_default_credit": self.uci_default_credit_csv,
            "paysim": self.paysim_csv,
        }
        return mapping.get(dataset)
