from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

from jsonschema import validate

from llm_ops.config import settings

CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "contracts"

@dataclass
class CanaryConfig:
    pass_eval: bool
    baseline_p95_ms: float
    baseline_error_rate: float
    baseline_tool_success: float
    min_citation_coverage: float
    max_error_rate: float
    max_p95_regression: float
    min_tool_success: float


@dataclass
class CanaryState:
    fraction: float
    mode: str


class CanaryController:
    def __init__(self, model_dir: Path) -> None:
        self.config = self._load_eval(model_dir)
        self.state = CanaryState(
            fraction=settings.canary_fraction if self.config.pass_eval else 0.0,
            mode="canary" if self.config.pass_eval else "stable",
        )
        self.latencies: Deque[float] = deque(maxlen=settings.canary_slo_window)
        self.errors: Deque[int] = deque(maxlen=settings.canary_slo_window)
        self.tool_success: Deque[int] = deque(maxlen=settings.canary_slo_window)
        self.citation_coverage: Deque[float] = deque(maxlen=settings.canary_slo_window)

    def _load_eval(self, model_dir: Path) -> CanaryConfig:
        path = model_dir / "eval_report.json"
        if not path.exists():
            return CanaryConfig(
                False,
                1200.0,
                0.01,
                0.98,
                settings.citation_min_coverage,
                0.02,
                0.2,
                0.95,
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        schema_path = CONTRACTS_DIR / "eval_report.schema.json"
        if schema_path.exists():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            validate(instance=data, schema=schema)
        baseline = data.get("baseline", {})
        thresholds = data.get("thresholds", {})
        return CanaryConfig(
            pass_eval=bool(data.get("pass")),
            baseline_p95_ms=float(baseline.get("p95_latency_ms", 1200.0)),
            baseline_error_rate=float(baseline.get("error_rate", 0.01)),
            baseline_tool_success=float(baseline.get("tool_success_rate", 0.98)),
            min_citation_coverage=float(
                thresholds.get("citation_coverage_min", settings.citation_min_coverage)
            ),
            max_error_rate=float(thresholds.get("runtime_error_rate_max", 0.02)),
            max_p95_regression=float(thresholds.get("runtime_p95_regression_max", 0.2)),
            min_tool_success=float(thresholds.get("runtime_tool_success_min", 0.95)),
        )

    def choose_variant(self) -> str:
        if self.state.fraction <= 0:
            return "stable"
        return "canary" if random.random() < self.state.fraction else "stable"

    def record(
        self, latency_ms: float, is_error: bool, tool_success: bool, citation_coverage: float
    ) -> None:
        self.latencies.append(latency_ms)
        self.errors.append(1 if is_error else 0)
        self.tool_success.append(1 if tool_success else 0)
        self.citation_coverage.append(citation_coverage)
        self._evaluate()

    def _evaluate(self) -> None:
        if len(self.latencies) < settings.canary_min_samples:
            return
        p95 = self._percentile(list(self.latencies), 0.95)
        error_rate = sum(self.errors) / len(self.errors)
        tool_success_rate = sum(self.tool_success) / len(self.tool_success)
        citation_coverage = sum(self.citation_coverage) / len(self.citation_coverage)

        regression = p95 > self.config.baseline_p95_ms * (1 + self.config.max_p95_regression)
        error_bad = error_rate > self.config.max_error_rate
        tool_bad = tool_success_rate < self.config.min_tool_success
        cite_bad = citation_coverage < self.config.min_citation_coverage

        if regression or error_bad or tool_bad or cite_bad:
            self.state.fraction = 0.0
            self.state.mode = "rolled_back"
            return

        # Promote if SLO holds and we are still in canary mode.
        if self.state.fraction < 1.0:
            self.state.fraction = 1.0
            self.state.mode = "promoted"

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        values_sorted = sorted(values)
        k = int(round((len(values_sorted) - 1) * percentile))
        return values_sorted[k]
