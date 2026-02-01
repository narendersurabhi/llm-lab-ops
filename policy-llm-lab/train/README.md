# Training pipeline (local + cloud)

This folder provides a simple, reproducible layout for fine-tuning locally (Apple Silicon)
or in the cloud (GPU), then feeding results into eval + release packaging.

Layout:
- configs/: training configs (LlamaFactory/Unsloth style)
- scripts/: runner scripts for local and cloud
- outputs/: adapters and merged weights

Typical flow:
1) make train-local
2) make eval
3) make release

Cloud flow:
1) make train-cloud
2) make eval
3) make release
