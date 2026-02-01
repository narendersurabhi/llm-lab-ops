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

## PDF ingestion (no OCR)
1) Drop PDFs into `policy-llm-lab/data/pdfs/`
2) Run:
```
make ingest-pdfs
make index
```

PDFs are extracted to `policy-llm-lab/artifacts/ingest/pdfs/` as `.txt` files
and automatically included in indexing.

## Training (local + cloud)
```
make train-local   # LlamaFactory/Unsloth if installed
make train-cloud   # LlamaFactory on GPU if installed
make train-hf      # PyTorch + Transformers + PEFT (LoRA)
make train-qlora   # QLoRA (CUDA GPU only; uses bitsandbytes)
make train-dpo     # DPO alignment (requires preference dataset)
make train-mlx     # Apple Silicon MLX LoRA
```

Configs live in `policy-llm-lab/train/configs/` and output adapters go to
`policy-llm-lab/train/outputs/`. Use eval + release after training.
