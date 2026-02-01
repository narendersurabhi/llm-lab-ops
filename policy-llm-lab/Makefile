.PHONY: test unit component contract lint typecheck index eval release ingest ingest-pdfs deps train-local train-cloud train-hf train-qlora train-dpo train-mlx

VENV_PYTHON := ../.venv/bin/python
ifneq ("$(wildcard $(VENV_PYTHON))","")
PYTHON ?= $(VENV_PYTHON)
else
PYTHON ?= python3
endif
PYTEST ?= $(PYTHON) -m pytest
RELEASE_ID ?= local-dev
RELEASE_DIR ?= dist/$(RELEASE_ID)

test:
	$(PYTEST) -m "unit or component or contract"

unit:
	$(PYTEST) -m unit

component:
	$(PYTEST) -m component

contract:
	$(PYTEST) -m contract

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy llm_lab

deps:
	uv pip install -p $(PYTHON) -e ".[dev]"

index:
	$(PYTHON) -m llm_lab.indexer

eval:
	$(PYTHON) -m llm_lab.eval

ingest:
	$(PYTHON) -m llm_lab.ingest

ingest-pdfs:
	$(PYTHON) -m llm_lab.ingest_pdfs

release:
	RELEASE_ID=$(RELEASE_ID) RELEASE_DIR=$(RELEASE_DIR) $(PYTHON) -m llm_lab.release.packager

train-local:
	./train/scripts/train_local.sh

train-cloud:
	./train/scripts/train_cloud.sh

train-hf:
	$(PYTHON) train/scripts/train_transformers.py \
		--base-model meta-llama/Llama-3.1-8B-Instruct \
		--dataset data/finetune/train.jsonl

train-qlora:
	$(PYTHON) train/scripts/train_transformers.py \
		--base-model meta-llama/Llama-3.1-8B-Instruct \
		--dataset data/finetune/train.jsonl \
		--quantization qlora

train-dpo:
	$(PYTHON) train/scripts/train_dpo.py \
		--base-model meta-llama/Llama-3.1-8B-Instruct \
		--dataset data/finetune/dpo.jsonl

train-mlx:
	PYTHON=$(PYTHON) ./train/scripts/train_mlx.sh
