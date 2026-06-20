PYTHON ?= python3
CONFIG ?= local.config

.PHONY: self-check data-check profile-ibm download-ibm db-ibm benchmark-ibm enhanced-privacy clean

self-check:
	$(PYTHON) scripts/self_check.py

data-check:
	$(PYTHON) scripts/check_dataset_files.py --data-dir data/raw

profile-ibm:
	$(PYTHON) scripts/profile_dataset.py --env $(CONFIG) --dataset ibm_aml

download-ibm:
	$(PYTHON) scripts/download_ibm_aml.py --env $(CONFIG)

db-ibm:
	$(PYTHON) scripts/build_benchmark_database.py --env $(CONFIG) --dataset ibm_aml

benchmark-ibm:
	$(PYTHON) scripts/run_benchmark_suite.py --env $(CONFIG) --dataset ibm_aml

enhanced-privacy:
	$(PYTHON) scripts/run_enhanced_privacy_benchmarks.py --env $(CONFIG) --dataset ibm_aml

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
