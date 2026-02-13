from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.service_name = os.getenv("SERVICE_NAME", "policy-llm-ops-gateway")
        self.retrieval_url = os.getenv("RETRIEVAL_URL", "http://policy-llm-lab:8001")
        self.retrieval_db_path = os.getenv("RETRIEVAL_DB_PATH", "/release/index/index.sqlite")
        self.release_base_dir = os.getenv("RELEASE_BASE_DIR", "/release")
        self.release_path = os.getenv("RELEASE_PATH")
        self.release_id = os.getenv("RELEASE_ID")
        self.llama_cpp_url = os.getenv("LLAMA_CPP_URL", "http://llama:8080")
        self.llm_provider = os.getenv("LLM_PROVIDER", "llama_cpp")
        self.llama_timeout_s = float(os.getenv("LLAMA_TIMEOUT_S", "30"))
        self.llama_max_retries = int(os.getenv("LLAMA_MAX_RETRIES", "1"))
        self.llama_retry_backoff_ms = float(os.getenv("LLAMA_RETRY_BACKOFF_MS", "100"))
        self.model_dir = Path(os.getenv("MODEL_DIR", "/models"))
        self.mlx_model = os.getenv("MLX_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        self.mlx_adapter_path = os.getenv("MLX_ADAPTER_PATH")
        self.mlx_trust_remote_code = (
            os.getenv("MLX_TRUST_REMOTE_CODE", "true").lower() == "true"
        )
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.prompt_sample_rate = float(os.getenv("PROMPT_SAMPLE_RATE", "0.0"))
        self.canary_enabled = os.getenv("CANARY_ENABLED", "true").lower() == "true"
        self.canary_fraction = float(os.getenv("CANARY_FRACTION", "0.05"))
        self.canary_min_samples = int(os.getenv("CANARY_MIN_SAMPLES", "30"))
        self.canary_slo_window = int(os.getenv("CANARY_SLO_WINDOW", "200"))
        self.p95_regression_max = float(os.getenv("P95_REGRESSION_MAX", "0.2"))
        self.citation_min_coverage = float(os.getenv("CITATION_MIN_COVERAGE", "0.5"))
        self.gateway_max_inflight = int(os.getenv("GATEWAY_MAX_INFLIGHT", "64"))
        self.gateway_max_queue = int(os.getenv("GATEWAY_MAX_QUEUE", "256"))
        self.gateway_queue_timeout_ms = float(os.getenv("GATEWAY_QUEUE_TIMEOUT_MS", "250"))
        self.retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))
        self.fake_model_delay_ms = float(os.getenv("FAKE_MODEL_DELAY_MS", "0"))
        self.fake_model_error_every = int(os.getenv("FAKE_MODEL_ERROR_EVERY", "0"))
        self.redact_logs = os.getenv("REDACT_LOGS", "true").lower() == "true"


settings = Settings()
