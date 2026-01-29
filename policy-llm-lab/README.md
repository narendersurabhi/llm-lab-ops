# policy-llm-lab

Data → indexing → eval → release artifacts for the LLMOps pipeline.

## Release bundle layout
```
dist/<release_id>/
  manifest.json
  model/
    model_card.json
    model.gguf
  index/
    index.sqlite
  eval/
    eval_report.json
  contracts/
    manifest.schema.json
    model_card.schema.json
    eval_report.schema.json
  meta/
    sbom.json
    checksums.json
    attestation.json
    CHANGELOG.md
```

Schemas for `manifest.json`, `model_card.json`, and `eval_report.json` live in `contracts/`.

## Testing
```
make test          # unit + component + contract
make unit
make component
make contract
make lint
make typecheck
```

## Build pipeline
```
make ingest
make index
make eval
make release RELEASE_ID=local-dev
```
