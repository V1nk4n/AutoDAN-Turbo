"""
Throughput tuning for PRO **with four_tier eval** (~c00e940-style: short decode, few candidates).

``--pro_fast_profile`` does **not** switch to a separate tier1-only path. It sets CLI knobs so
``_evaluate_candidates_four_tier`` runs with the same funnel (tier1 gate → NLL rank → fast/dual top-N)
but at lower cost per repeat.
"""
from __future__ import annotations

import argparse
from typing import Any, Mapping


# Keys match ``argparse.Namespace`` / ``main.py`` dest names.
FAST_PROFILE_OVERRIDES: Mapping[str, Any] = {
    "pro_fast_profile": True,
    "pro_four_tier_eval": True,
    "pro_staged_eval_enabled": False,
    "pro_enable_eval_cache": True,
    "pro_enable_fast_judge": True,
    "pro_enable_feedback_scheduler": True,
    # Short target decode (dominant speedup inside four_tier batch decode).
    "pro_explore_max_new_tokens": 64,
    "pro_exploit_max_new_tokens": 128,
    "pro_explore_n_candidates": 2,
    "pro_explore_top_k": 1,
    "pro_exploit_n_candidates": 4,
    "pro_exploit_top_k": 2,
    "pro_eval_batch_size": 2,
    "pro_verifier_top_n": 1,
    "pro_early_stop_min_delta": 0.01,
    "pro_early_stop_patience": 5,
    "pro_phase_split": 0.7,
    "pro_feedback_every": 2,
    "pro_dynamic_pattern_select": False,
    "pro_per_candidate_strategy_bundles": False,
    "pro_rotate_explore_across_candidates": False,
}


def apply_pro_fast_profile(args: argparse.Namespace, *, logger: Any = None) -> None:
    """Mutate parsed CLI namespace to throughput-tuned four_tier settings."""
    for key, value in FAST_PROFILE_OVERRIDES.items():
        setattr(args, key, value)
    if logger is not None:
        logger.info(
            "[PRO] pro_fast_profile: four_tier=ON with throughput tune "
            "(explore 64/exploit 128 tokens, explore_n=2 top_k=1, verifier_top_n=1, eval_cache=ON)"
        )


def fast_profile_baseline_cli() -> dict:
    """Snapshot for telemetry when fast profile is active."""
    return dict(FAST_PROFILE_OVERRIDES)
