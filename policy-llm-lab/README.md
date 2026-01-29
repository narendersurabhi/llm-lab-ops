# policy-llm-lab

Data → indexing → eval → release artifacts for the LLMOps pipeline.

## Release bundle layout
```
release/
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
