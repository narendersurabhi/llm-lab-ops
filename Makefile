.PHONY: up test test-uv uv-test lint lint-uv kind-up loadtest release index ingest eval uv-venv uv-install

PYTHON ?= python3
UV ?= uv
UV_PYTHON ?= 3.11
UV_RUN = $(UV) run --python $(UV_PYTHON)

up:
	docker compose up --build

test:
	cd policy-llm-lab && $(PYTHON) -m pytest
	cd policy-llm-ops && $(PYTHON) -m pytest
	cd policy-llm-lab && $(PYTHON) -m ruff check .
	cd policy-llm-ops && $(PYTHON) -m ruff check .
	cd policy-llm-lab && $(PYTHON) -m mypy llm_lab
	cd policy-llm-ops && $(PYTHON) -m mypy llm_ops

test-uv:
	cd policy-llm-lab && $(UV_RUN) -m pytest
	cd policy-llm-ops && $(UV_RUN) -m pytest
	cd policy-llm-lab && $(UV_RUN) -m ruff check .
	cd policy-llm-ops && $(UV_RUN) -m ruff check .
	cd policy-llm-lab && $(UV_RUN) -m mypy llm_lab
	cd policy-llm-ops && $(UV_RUN) -m mypy llm_ops

uv-test: uv-install test-uv

lint:
	cd policy-llm-lab && $(PYTHON) -m ruff check .
	cd policy-llm-ops && $(PYTHON) -m ruff check .

lint-uv:
	cd policy-llm-lab && $(UV_RUN) -m ruff check .
	cd policy-llm-ops && $(UV_RUN) -m ruff check .

kind-up:
	kind create cluster --config infra/kind/kind-config.yaml
	docker build -t policy-llm-lab:dev -f policy-llm-lab/Dockerfile .
	docker build -t policy-llm-ops:dev -f policy-llm-ops/Dockerfile .
	kind load docker-image policy-llm-lab:dev
	kind load docker-image policy-llm-ops:dev
	helm upgrade --install llm-stack infra/helm/llm-stack

loadtest:
	$(PYTHON) scripts/loadtest.py

release:
	cd policy-llm-lab && $(PYTHON) -m llm_lab.release.packager

index:
	cd policy-llm-lab && $(PYTHON) -m llm_lab.indexer

ingest:
	cd policy-llm-lab && $(PYTHON) -m llm_lab.ingest

eval:
	cd policy-llm-lab && $(PYTHON) -m llm_lab.eval

uv-venv:
	$(UV) venv

uv-install:
	$(UV) venv
	$(UV) pip install -e ./policy-llm-lab[dev]
	$(UV) pip install -e ./policy-llm-ops[dev]
