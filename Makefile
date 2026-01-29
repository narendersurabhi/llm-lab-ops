.PHONY: up test lint kind-up loadtest release index

up:
	docker compose up --build

test:
	cd policy-llm-lab && python -m pytest
	cd policy-llm-ops && python -m pytest
	cd policy-llm-lab && python -m ruff check .
	cd policy-llm-ops && python -m ruff check .
	cd policy-llm-lab && python -m mypy llm_lab
	cd policy-llm-ops && python -m mypy llm_ops

lint:
	cd policy-llm-lab && python -m ruff check .
	cd policy-llm-ops && python -m ruff check .

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
	cd policy-llm-lab && python -m llm_lab.release

index:
	cd policy-llm-lab && python -m llm_lab.indexer
