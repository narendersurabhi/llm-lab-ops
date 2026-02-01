# policy-llm-ops

OpenAI-compatible gateway, LangGraph agent runtime, observability, and canary/rollback logic.

## Testing
```
make test                 # unit tests
make contract-test        # cross-repo contract tests
make test-integration     # docker compose integration tests
make e2e                  # rollout/rollback simulation
make lint
make typecheck
```

Integration tests expect a release bundle mounted at `policy-llm-lab/dist/local-dev`.
Use `make test-integration` to build it and run docker compose tests end-to-end.

## Run locally
```
uvicorn llm_ops.gateway:app --reload --host 0.0.0.0 --port 8000
```

## Run with MLX (Apple Silicon, local inference)
MLX runs on the host (not inside Docker). Use the release bundle adapter produced by
`policy-llm-lab` and set `LLM_PROVIDER=mlx`.

```
export LLM_PROVIDER=mlx
export RELEASE_PATH=../policy-llm-lab/dist/local-dev
export MLX_MODEL=Qwen/Qwen2.5-3B-Instruct
export MLX_ADAPTER_PATH=../policy-llm-lab/dist/local-dev/model/adapter
uvicorn llm_ops.gateway:app --host 0.0.0.0 --port 8002
```

## Streamlit UI
Install the UI extras:
```
pip install -e ".[ui]"
```

Run the UI (expects ops on `http://localhost:8002`):
```
make run-ui
```

The UI includes a "Models" tab that scans the release bundles directory
(`RELEASES_DIR`, default `/releases` in Docker) and shows key stats.

## Docker Compose UI
The UI is included in `docker-compose.yml` and will start with `make up` or
`make pipeline-local`. It defaults to `http://localhost:8502`.
Override with `UI_PORT=8502 make up`.

## Host MLX + Docker UI (Option A)
Run MLX inference on the host and keep UI/observability in Docker:
```
make pipeline-local-mlx
```

Stop MLX host server:
```
make stop-mlx
```

Stop UI/observability + MLX:
```
make pipeline-local-mlx-down
```

## Serve latest release (no retraining)
Use the most recent bundle from `policy-llm-lab/dist` and start host MLX + UI:
```
make serve-latest-mlx
```
