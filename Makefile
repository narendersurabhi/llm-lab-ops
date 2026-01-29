.PHONY: up test lint typecheck kind-up loadtest release index ingest eval contract-all test-integration e2e uv-venv uv-install deps

VENV_PYTHON := .venv/bin/python
ifneq ("$(wildcard $(VENV_PYTHON))","")
PYTHON ?= $(VENV_PYTHON)
else
PYTHON ?= python3
endif
UV ?= uv
UV_PYTHON ?= 3.11
UV_RUN = $(UV) run --python $(UV_PYTHON)

up:
	docker compose up --build

test:
	$(MAKE) -C policy-llm-lab test
	$(MAKE) -C policy-llm-ops test

lint:
	$(MAKE) -C policy-llm-lab lint
	$(MAKE) -C policy-llm-ops lint

typecheck:
	$(MAKE) -C policy-llm-lab typecheck
	$(MAKE) -C policy-llm-ops typecheck

kind-up:
	kind create cluster --config infra/kind/kind-config.yaml
	docker build -t policy-llm-lab:dev -f policy-llm-lab/Dockerfile .
	docker build -t policy-llm-ops:dev -f policy-llm-ops/Dockerfile .
	kind load docker-image policy-llm-lab:dev
	kind load docker-image policy-llm-ops:dev
	helm upgrade --install llm-stack infra/helm/llm-stack

loadtest:
	python scripts/loadtest.py

release:
	$(MAKE) -C policy-llm-lab release

index:
	$(MAKE) -C policy-llm-lab index

ingest:
	$(MAKE) -C policy-llm-lab ingest

eval:
	$(MAKE) -C policy-llm-lab eval

test-integration:
	$(MAKE) -C policy-llm-ops test-integration

e2e:
	$(MAKE) -C policy-llm-ops e2e

contract-all:
	$(MAKE) -C policy-llm-lab release RELEASE_ID=local-dev RELEASE_DIR=dist/local-dev
	RELEASE_PATH=$(CURDIR)/policy-llm-lab/dist/local-dev $(MAKE) -C policy-llm-ops contract-test

uv-venv:
	$(UV) venv

uv-install:
	$(UV) venv
	$(UV) pip install -e ./policy-llm-lab[dev]
	$(UV) pip install -e ./policy-llm-ops[dev]

deps:
	$(UV) venv --python $(UV_PYTHON)
	$(UV) pip install -e ./policy-llm-lab[dev]
	$(UV) pip install -e ./policy-llm-ops[dev]
