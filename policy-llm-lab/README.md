# policy-llm-lab

Data → indexing → eval → release artifacts for the LLMOps pipeline.

## Release bundle layout
```
release/
  manifest.json
  model/
    model.gguf
    model_card.json
    eval_report.json
  index/
    index.sqlite
```

Schemas for `manifest.json`, `model_card.json`, and `eval_report.json` live in `contracts/`.
