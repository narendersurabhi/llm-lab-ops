.PHONY: up test lint typecheck kind-up loadtest release index ingest eval contract-all test-integration e2e uv-venv uv-install deps pipeline-local pipeline-cloud run-mlx run-mlx-bg stop-mlx pipeline-local-mlx pipeline-local-mlx-down pipeline-mlx serve-latest-mlx

VENV_PYTHON := .venv/bin/python
ifneq ("$(wildcard $(VENV_PYTHON))","")
PYTHON ?= $(VENV_PYTHON)
else
PYTHON ?= python3
endif
UV ?= uv
UV_PYTHON ?= 3.11
UV_RUN = $(UV) run --python $(UV_PYTHON)
OPS_PORT ?= 8002
UI_PORT ?= 8502

up:
	OPS_PORT=$(OPS_PORT) UI_PORT=$(UI_PORT) docker compose up -d --build

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
	OPS_PORT=8002 $(MAKE) -C policy-llm-ops test-integration

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

pipeline-local:
	$(MAKE) -C policy-llm-lab ingest
	$(MAKE) -C policy-llm-lab index
	$(MAKE) -C policy-llm-lab train-mlx
	$(MAKE) -C policy-llm-lab eval
	$(MAKE) -C policy-llm-lab release RELEASE_ID=local-dev RELEASE_DIR=dist/local-dev
	RELEASE_PATH=$(CURDIR)/policy-llm-lab/dist/local-dev OPS_PORT=$(OPS_PORT) $(MAKE) up

pipeline-cloud:
	$(MAKE) -C policy-llm-lab ingest
	$(MAKE) -C policy-llm-lab index
	$(MAKE) -C policy-llm-lab train-cloud
	$(MAKE) -C policy-llm-lab eval
	$(MAKE) -C policy-llm-lab release RELEASE_ID=cloud-dev RELEASE_DIR=dist/cloud-dev
	RELEASE_PATH=$(CURDIR)/policy-llm-lab/dist/cloud-dev OPS_PORT=$(OPS_PORT) $(MAKE) up

run-mlx:
	LLM_PROVIDER=mlx \
	RELEASE_PATH=$(CURDIR)/policy-llm-lab/dist/local-dev \
	MLX_MODEL=Qwen/Qwen2.5-3B-Instruct \
	MLX_ADAPTER_PATH=$(CURDIR)/policy-llm-lab/dist/local-dev/model/adapter \
	$(PYTHON) -m uvicorn llm_ops.gateway:app --host 0.0.0.0 --port $(OPS_PORT)

run-mlx-bg:
	OPS_PORT=$(OPS_PORT) MLX_MODEL=Qwen/Qwen2.5-3B-Instruct ./scripts/run_mlx_host.sh

stop-mlx:
	./scripts/stop_mlx_host.sh

pipeline-local-mlx:
	$(MAKE) -C policy-llm-lab ingest
	$(MAKE) -C policy-llm-lab index
	$(MAKE) -C policy-llm-lab train-mlx
	$(MAKE) -C policy-llm-lab eval
	$(MAKE) -C policy-llm-lab release RELEASE_ID=local-dev RELEASE_DIR=dist/local-dev
	OPS_PORT=$(OPS_PORT) $(MAKE) run-mlx-bg
	OPS_PORT=$(OPS_PORT) UI_PORT=$(UI_PORT) docker compose stop policy-llm-ops policy-llm-lab || true
	OPS_PORT=$(OPS_PORT) UI_PORT=$(UI_PORT) OPS_BASE_URL=http://host.docker.internal:$(OPS_PORT) docker compose up -d --build grafana prometheus jaeger loki policy-llm-ui

pipeline-local-mlx-down:
	OPS_PORT=$(OPS_PORT) UI_PORT=$(UI_PORT) docker compose down -v
	$(MAKE) stop-mlx

pipeline-mlx: pipeline-local-mlx

serve-latest-mlx:
	@RELEASE_PATH=$$(./scripts/latest_release.sh) && \
	OPS_PORT=$(OPS_PORT) RELEASE_PATH=$$RELEASE_PATH MLX_MODEL=Qwen/Qwen2.5-3B-Instruct \
	MLX_ADAPTER_PATH=$$RELEASE_PATH/model/adapter ./scripts/run_mlx_host.sh && \
	OPS_PORT=$(OPS_PORT) UI_PORT=$(UI_PORT) OPS_BASE_URL=http://host.docker.internal:$(OPS_PORT) \
	docker compose up -d --build grafana prometheus jaeger loki policy-llm-ui
