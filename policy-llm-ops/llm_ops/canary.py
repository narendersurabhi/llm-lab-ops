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
    error_rate_max: float
    tool_success_min: float
    citation_coverage_min: float


@dataclass
class CanaryState:
    fraction: float
    mode: str


class CanaryController:
    def __init__(self, eval_report_path: Path) -> None:
        self.config = self._load_eval(eval_report_path)
        self.state = CanaryState(
            fraction=settings.canary_fraction if self.config.pass_eval else 0.0,
            mode="canary" if self.config.pass_eval else "stable",
        )
        self.latencies: Deque[float] = deque(maxlen=settings.canary_slo_window)
        self.errors: Deque[int] = deque(maxlen=settings.canary_slo_window)
        self.tool_success: Deque[int] = deque(maxlen=settings.canary_slo_window)
        self.citation_coverage: Deque[float] = deque(maxlen=settings.canary_slo_window)

    def _load_eval(self, eval_report_path: Path) -> CanaryConfig:
        if not eval_report_path.exists():
            return CanaryConfig(False, 1200.0, 0.02, 0.95, settings.citation_min_coverage)
        data = json.loads(eval_report_path.read_text(encoding="utf-8"))
        schema_path = CONTRACTS_DIR / "eval_report.schema.json"
        if schema_path.exists():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            validate(instance=data, schema=schema)
        metrics = data.get("metrics", {})
        thresholds = data.get("thresholds", {})
        baseline_p95 = float(metrics.get("latency_p95_ms", 1200.0))
        error_rate_max = float(_threshold_value(thresholds, "error_rate", "max", 0.02))
        tool_success_min = float(
            _threshold_value(thresholds, "tool_call_success_rate", "min", 0.95)
        )
        citation_min = float(
            _threshold_value(
                thresholds, "citation_coverage", "min", settings.citation_min_coverage
            )
        )
        return CanaryConfig(
            pass_eval=bool(data.get("pass")),
            baseline_p95_ms=baseline_p95,
            error_rate_max=error_rate_max,
            tool_success_min=tool_success_min,
            citation_coverage_min=citation_min,
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

        regression = p95 > self.config.baseline_p95_ms * (1 + settings.p95_regression_max)
        error_bad = error_rate > self.config.error_rate_max
        tool_bad = tool_success_rate < self.config.tool_success_min
        cite_bad = citation_coverage < self.config.citation_coverage_min

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


def _threshold_value(thresholds: dict, key: str, bound: str, default: float) -> float:
    if not isinstance(thresholds, dict):
        return default
    entry = thresholds.get(key, {})
    if not isinstance(entry, dict):
        return default
    value = entry.get(bound)
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
