.PHONY: test test-integration e2e contract-test up down lint typecheck

VENV_PYTHON := ../.venv/bin/python
ifneq ("$(wildcard $(VENV_PYTHON))","")
PYTHON ?= $(VENV_PYTHON)
else
PYTHON ?= python3
endif
PYTEST ?= $(PYTHON) -m pytest
COMPOSE ?= docker compose
COMPOSE_FILE ?= ../docker-compose.yml

test:
	$(PYTEST) -m unit

contract-test:
	$(PYTEST) -m contract

e2e:
	$(PYTEST) -m e2e

test-integration:
	$(MAKE) -C ../policy-llm-lab release RELEASE_ID=local-dev RELEASE_DIR=dist/local-dev
	$(COMPOSE) -f $(COMPOSE_FILE) up -d --build
	$(PYTEST) -m integration
	$(COMPOSE) -f $(COMPOSE_FILE) down -v

up:
	$(COMPOSE) -f $(COMPOSE_FILE) up --build

down:
	$(COMPOSE) -f $(COMPOSE_FILE) down -v

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy llm_ops
