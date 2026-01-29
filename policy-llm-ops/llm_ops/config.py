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
        self.model_dir = Path(os.getenv("MODEL_DIR", "/models"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.prompt_sample_rate = float(os.getenv("PROMPT_SAMPLE_RATE", "0.0"))
        self.canary_enabled = os.getenv("CANARY_ENABLED", "true").lower() == "true"
        self.canary_fraction = float(os.getenv("CANARY_FRACTION", "0.05"))
        self.canary_min_samples = int(os.getenv("CANARY_MIN_SAMPLES", "30"))
        self.canary_slo_window = int(os.getenv("CANARY_SLO_WINDOW", "200"))
        self.citation_min_coverage = float(os.getenv("CITATION_MIN_COVERAGE", "0.5"))
        self.redact_logs = os.getenv("REDACT_LOGS", "true").lower() == "true"


settings = Settings()
