# FinancialSurvey

Source code for running financial transaction benchmark experiments.

## Contents

- `src/financial_survey/`: shared dataset profiling and configuration helpers.
- `scripts/check_dataset_files.py`: checks whether expected dataset files exist.
- `scripts/profile_dataset.py`: profiles a configured CSV dataset.
- `scripts/build_benchmark_database.py`: builds a local SQLite database.
- `scripts/run_benchmark_suite.py`: runs baseline and federated benchmark methods.
- `scripts/run_enhanced_privacy_benchmarks.py`: runs enhanced privacy-preserving benchmark methods.
- `scripts/run_classical_baselines.py`: runs standalone classical baselines.
- `scripts/download_ibm_aml.py`: downloads the IBM AML dataset through `kagglehub`.

Generated datasets, local configuration files, outputs, logs, caches, and result
folders are excluded from version control.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-benchmark.txt
```

Optional homomorphic-encryption dependencies:

```bash
pip install -r requirements-he.txt
```

## Local Configuration

Create a local key-value config file, for example `local.config`. Keep it out of
Git.

Required values for the IBM AML workflow:

```text
FINANCIAL_SURVEY_ROOT=/path/to/FinancialSurvey
FINANCIAL_SURVEY_DATA_DIR=/path/to/FinancialSurvey/data/raw
FINANCIAL_SURVEY_OUTPUT_DIR=/path/to/FinancialSurvey/outputs
IBM_AML_CSV=/path/to/FinancialSurvey/data/raw/HI-Small_Trans.csv
```

## Data

Place CSV datasets under:

```text
data/raw/
```

Check local dataset placement:

```bash
python scripts/check_dataset_files.py --data-dir data/raw
```

To download the IBM AML dataset with `kagglehub`:

```bash
python scripts/download_ibm_aml.py --env local.config
```

## Run

Source check:

```bash
python scripts/self_check.py
```

Profile a dataset:

```bash
python scripts/profile_dataset.py --env local.config --dataset ibm_aml
```

Build SQLite:

```bash
python scripts/build_benchmark_database.py \
  --env local.config \
  --dataset ibm_aml \
  --if-exists replace
```

Run the standard benchmark suite:

```bash
python scripts/run_benchmark_suite.py \
  --env local.config \
  --dataset ibm_aml \
  --sample-rows 300000 \
  --forest-rows 120000 \
  --num-clients 10 \
  --fed-rounds 5 \
  --run-id standard_300k
```

Run the enhanced privacy benchmark suite:

```bash
python scripts/run_enhanced_privacy_benchmarks.py \
  --env local.config \
  --dataset ibm_aml \
  --sample-rows 500000 \
  --num-clients 20 \
  --fed-rounds 8 \
  --local-epochs 1 \
  --run-id enhanced_privacy_500k
```

## Makefile

```bash
make self-check
make data-check
make profile-ibm CONFIG=local.config
make db-ibm CONFIG=local.config
make benchmark-ibm CONFIG=local.config
make enhanced-privacy CONFIG=local.config
make clean
```
