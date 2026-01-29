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
