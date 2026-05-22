"""Machine-readable PRO threshold telemetry (JSONL) for post-run analysis."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Tunable thresholds and flags recorded once per run in ``run_config``.
THRESHOLD_CONFIG_KEYS = (
    "nll_min",
    "nll_max",
    "pro_score_threshold",
    "pro_early_stop_patience",
    "pro_early_stop_min_delta",
    "pro_goal_similarity_floor",
    "pro_strategy_embed_min_sim",
    "pro_enable_strategy_embed_match",
    "pro_enable_prompt_strategy_attribution",
    "pro_fast_judge_min_len",
    "pro_enable_fast_judge",
    "pro_four_tier_eval",
    "pro_verifier_top_n",
    "pro_dynamic_pattern_select",
    "pro_pattern_exploit_n",
    "pro_pattern_explore_n",
    "pro_pattern_rank_w_rate",
    "pro_pattern_rank_w_avg",
    "pro_pattern_rank_w_req",
    "pro_pattern_explore_seed",
    "pro_phase_split",
    "pro_n_candidates",
    "pro_top_k",
    "pro_explore_n_candidates",
    "pro_exploit_n_candidates",
    "pro_explore_top_k",
    "pro_exploit_top_k",
    "pro_explore_max_new_tokens",
    "pro_exploit_max_new_tokens",
    "target_max_new_tokens",
    "pro_staged_eval_enabled",
    "pro_staged_filter_keep_ratio",
    "pro_staged_probe_keep_ratio",
    "pro_staged_uncertainty_band",
    "pro_feedback_every",
    "pro_feedback_cooldown_repeats",
    "pro_per_candidate_strategy_bundles",
    "pro_rotate_explore_across_candidates",
    "target_model_key",
)

# Keys present in config but not used by pipeline logic (report warns).
UNUSED_CONFIG_KEYS = ("pro_score_threshold",)


def threshold_config_snapshot(obj: Any) -> Dict[str, Any]:
    """Extract tunable thresholds from a pipeline instance or config dataclass."""
    out: Dict[str, Any] = {}
    if is_dataclass(obj):
        raw = asdict(obj)
        for k in THRESHOLD_CONFIG_KEYS:
            if k in raw:
                out[k] = raw[k]
        return out
    for k in THRESHOLD_CONFIG_KEYS:
        if hasattr(obj, k):
            out[k] = getattr(obj, k)
    return out


def effective_score_loss_threshold(v: float) -> float:
    """Map CLI legacy 0–1 values to score_loss scale 0–10 for reporting."""
    x = float(v)
    if 0.0 < x < 1.0:
        return x * 10.0
    return x


class ProThresholdTelemetry:
    """Append-only JSONL writer; one object per line."""

    def __init__(self, path: Optional[str], *, enabled: bool = True) -> None:
        self.enabled = bool(enabled and path)
        self.path = Path(path) if path else None
        self._fh = None
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    def emit(self, kind: str, **fields: Any) -> None:
        if not self.enabled or self._fh is None:
            return
        row = {
            "ts": time.time(),
            "kind": kind,
            **fields,
        }
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
