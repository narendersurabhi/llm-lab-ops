.PHONY: test unit component contract lint typecheck index eval release ingest

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

index:
	$(PYTHON) -m llm_lab.indexer

eval:
	$(PYTHON) -m llm_lab.eval

ingest:
	$(PYTHON) -m llm_lab.ingest

release:
	RELEASE_ID=$(RELEASE_ID) RELEASE_DIR=$(RELEASE_DIR) $(PYTHON) -m llm_lab.release.packager
