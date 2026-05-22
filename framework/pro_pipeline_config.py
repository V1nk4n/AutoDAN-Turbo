"""Single configuration object for ``AutoDANTurboPro`` (CLI / experiments)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ProPipelineConfig:
    epochs: int = 150
    warm_up_iterations: int = 1
    lifelong_iterations: int = 4
    log_every: int = 10
    pro_n_candidates: int = 4
    pro_top_k: int = 2
    pro_score_threshold: float = 0.5
    target_max_new_tokens: int = 150
    target_model_key: str = ""
    nll_min: float = 0.0
    nll_max: float = 10.0
    # Deprecated: PRO always uses ``epochs`` as repeats-per-request. Kept for CLI compat.
    repeat_shots_per_request: bool = True
    pro_early_stop_patience: int = 5
    pro_early_stop_min_delta: float = 0.1
    pro_feedback_every: int = 2
    pro_phase_split: float = 0.7
    pro_explore_n_candidates: int = 2
    pro_explore_top_k: int = 1
    pro_exploit_n_candidates: int = 4
    pro_exploit_top_k: int = 2
    pro_explore_max_new_tokens: int = 64
    pro_exploit_max_new_tokens: int = 128
    pro_enable_eval_cache: bool = False
    pro_eval_batch_size: int = 2
    pro_enable_retrieval_cache: bool = True
    pro_retrieval_cache_max_entries: int = 4096
    pro_eval_cache_max_entries: int = 4096
    pro_enable_fast_judge: bool = True
    pro_fast_judge_min_len: int = 24
    pro_enable_feedback_scheduler: bool = True
    pro_feedback_cooldown_repeats: int = 1
    pro_enable_strategy_embed_match: bool = True
    pro_strategy_embed_min_sim: float = 0.22
    # Credit library strategies by embedding the jailbreak prompt vs bundle strategies used to generate it.
    pro_enable_prompt_strategy_attribution: bool = True
    pro_goal_similarity_floor: float = 0.15
    pro_staged_eval_enabled: bool = False
    pro_staged_eval_profile: str = "balanced"
    pro_staged_filter_keep_ratio: float = 0.5
    pro_staged_probe_keep_ratio: float = 0.5
    pro_staged_short_max_new_tokens: int = 32
    pro_staged_min_candidates_for_full_eval: int = 1
    pro_staged_uncertainty_band: float = 0.1
    pro_staged_eval_budget_ms: float = 0.0
    pro_staged_weight_filter: float = 0.35
    pro_staged_weight_probe: float = 0.65
    pro_staged_uncertainty_penalty: float = 0.2
    # When False (default), pipeline_stage / PRO events log as short human-readable lines on the console.
    # When True, restore one-line JSON for machine parsing or deep debugging.
    pro_verbose_pipeline_logs: bool = False
    # Append-only JSONL for threshold tuning (see scripts/analyze_pro_thresholds.py).
    pro_enable_threshold_telemetry: bool = True
    pro_telemetry_jsonl: Optional[str] = None
    # --- Optional "decoupled" PRO architecture (embedding rank + 4-tier eval + metadata) ---
    # Dynamic pattern selection: S_rank = w_rate*norm(rate) + w_avg*norm(avg_score) + w_req*norm(req_sim).
    pro_dynamic_pattern_select: bool = False
    pro_pattern_exploit_n: int = 3
    pro_pattern_explore_n: int = 2
    pro_pattern_rank_w_rate: float = 0.3
    pro_pattern_rank_w_avg: float = 0.3
    pro_pattern_rank_w_req: float = 0.4
    pro_pattern_explore_seed: Optional[int] = None
    # Four-tier evaluation: relevance (strict) → NLL score_loss sort → dual judge on top-N only →
    # success = (J==1). Incompatible with staged eval (staged skipped).
    pro_four_tier_eval: bool = False
    pro_verifier_top_n: int = 2
    # Rotate explore order + focus per slot (same explore pair for the whole repeat).
    pro_rotate_explore_across_candidates: bool = False
    # Resample explore per PRO candidate (shared exploit block); needs dynamic select + embeddings.
    pro_per_candidate_strategy_bundles: bool = False

    def __post_init__(self) -> None:
        self.log_every = max(1, int(self.log_every))
        self.pro_early_stop_patience = max(1, int(self.pro_early_stop_patience))
        self.pro_feedback_every = max(1, int(self.pro_feedback_every))
        self.pro_phase_split = max(0.0, min(1.0, float(self.pro_phase_split)))
        self.pro_explore_n_candidates = max(1, int(self.pro_explore_n_candidates))
        self.pro_explore_top_k = max(1, int(self.pro_explore_top_k))
        self.pro_exploit_n_candidates = max(1, int(self.pro_exploit_n_candidates))
        self.pro_exploit_top_k = max(1, int(self.pro_exploit_top_k))
        self.pro_explore_max_new_tokens = max(1, int(self.pro_explore_max_new_tokens))
        self.pro_exploit_max_new_tokens = max(1, int(self.pro_exploit_max_new_tokens))
        self.pro_retrieval_cache_max_entries = max(0, int(self.pro_retrieval_cache_max_entries))
        self.pro_eval_cache_max_entries = max(0, int(self.pro_eval_cache_max_entries))
        self.pro_eval_batch_size = max(1, int(self.pro_eval_batch_size))
        self.pro_strategy_embed_min_sim = max(-1.0, min(1.0, float(self.pro_strategy_embed_min_sim)))
        self.pro_goal_similarity_floor = max(0.0, min(1.0, float(self.pro_goal_similarity_floor)))
        prof = str(self.pro_staged_eval_profile or "balanced").strip().lower()
        if prof not in {"conservative", "balanced", "aggressive"}:
            prof = "balanced"
        self.pro_staged_eval_profile = prof
        self.pro_staged_filter_keep_ratio = max(0.0, min(1.0, float(self.pro_staged_filter_keep_ratio)))
        self.pro_staged_probe_keep_ratio = max(0.0, min(1.0, float(self.pro_staged_probe_keep_ratio)))
        self.pro_staged_short_max_new_tokens = max(1, int(self.pro_staged_short_max_new_tokens))
        self.pro_staged_min_candidates_for_full_eval = max(1, int(self.pro_staged_min_candidates_for_full_eval))
        self.pro_staged_uncertainty_band = max(0.0, min(1.0, float(self.pro_staged_uncertainty_band)))
        self.pro_staged_eval_budget_ms = max(0.0, float(self.pro_staged_eval_budget_ms))
        self.pro_pattern_exploit_n = max(0, int(self.pro_pattern_exploit_n))
        self.pro_pattern_explore_n = max(0, int(self.pro_pattern_explore_n))
        self.pro_pattern_rank_w_rate = max(0.0, min(1.0, float(self.pro_pattern_rank_w_rate)))
        self.pro_pattern_rank_w_avg = max(0.0, min(1.0, float(self.pro_pattern_rank_w_avg)))
        self.pro_pattern_rank_w_req = max(0.0, min(1.0, float(self.pro_pattern_rank_w_req)))
        wsum = (
            self.pro_pattern_rank_w_rate
            + self.pro_pattern_rank_w_avg
            + self.pro_pattern_rank_w_req
        )
        if wsum > 1e-9:
            self.pro_pattern_rank_w_rate = float(self.pro_pattern_rank_w_rate) / wsum
            self.pro_pattern_rank_w_avg = float(self.pro_pattern_rank_w_avg) / wsum
            self.pro_pattern_rank_w_req = float(self.pro_pattern_rank_w_req) / wsum
        self.pro_verifier_top_n = max(1, int(self.pro_verifier_top_n))
        self.pro_early_stop_min_delta = self._normalize_score_loss_threshold(self.pro_early_stop_min_delta)

    @staticmethod
    def _normalize_score_loss_threshold(v: float) -> float:
        """Map legacy fractional thresholds (e.g. 0.1) to score_loss 0–10 scale (1.0).

        Values already on the 0–10 scale (e.g. 1.0, 2.0) are left unchanged.
        """
        x = float(v)
        if 0.0 < x < 1.0:
            return x * 10.0
        return x

    @classmethod
    def from_argparse(cls, args: Any, *, target_model_key: str = "") -> ProPipelineConfig:
        """Build from ``argparse.Namespace`` (``main.py`` / ``eval_pro.py``)."""
        strat_embed = not bool(getattr(args, "pro_disable_strategy_embed_match", False))
        prompt_attrib = not bool(getattr(args, "pro_disable_prompt_strategy_attribution", False))
        return cls(
            epochs=int(getattr(args, "epochs", 150)),
            warm_up_iterations=int(getattr(args, "warm_up_iterations", 1)),
            lifelong_iterations=int(getattr(args, "lifelong_iterations", 4)),
            log_every=int(getattr(args, "log_every", 10)),
            pro_n_candidates=int(getattr(args, "pro_n_candidates", 4)),
            pro_top_k=int(getattr(args, "pro_top_k", 2)),
            pro_score_threshold=float(getattr(args, "pro_score_threshold", 0.5)),
            target_max_new_tokens=int(getattr(args, "target_max_new_tokens", 150)),
            target_model_key=str(target_model_key or getattr(args, "target_model_key", "") or ""),
            nll_min=float(getattr(args, "nll_min", 0.0)),
            nll_max=float(getattr(args, "nll_max", 10.0)),
            repeat_shots_per_request=bool(getattr(args, "pro_repeat_shots_per_request", False)),
            pro_early_stop_patience=int(getattr(args, "pro_early_stop_patience", 5)),
            pro_early_stop_min_delta=float(getattr(args, "pro_early_stop_min_delta", 0.1)),
            pro_feedback_every=int(getattr(args, "pro_feedback_every", 2)),
            pro_phase_split=float(getattr(args, "pro_phase_split", 0.7)),
            pro_explore_n_candidates=int(getattr(args, "pro_explore_n_candidates", 2)),
            pro_explore_top_k=int(getattr(args, "pro_explore_top_k", 1)),
            pro_exploit_n_candidates=int(getattr(args, "pro_exploit_n_candidates", 4)),
            pro_exploit_top_k=int(getattr(args, "pro_exploit_top_k", 2)),
            pro_explore_max_new_tokens=int(getattr(args, "pro_explore_max_new_tokens", 64)),
            pro_exploit_max_new_tokens=int(getattr(args, "pro_exploit_max_new_tokens", 128)),
            pro_enable_eval_cache=bool(getattr(args, "pro_enable_eval_cache", False)),
            pro_eval_batch_size=int(getattr(args, "pro_eval_batch_size", 2)),
            pro_enable_retrieval_cache=bool(getattr(args, "pro_enable_retrieval_cache", True)),
            pro_retrieval_cache_max_entries=int(getattr(args, "pro_retrieval_cache_max_entries", 4096)),
            pro_eval_cache_max_entries=int(getattr(args, "pro_eval_cache_max_entries", 4096)),
            pro_enable_fast_judge=bool(getattr(args, "pro_enable_fast_judge", True)),
            pro_fast_judge_min_len=int(getattr(args, "pro_fast_judge_min_len", 24)),
            pro_enable_feedback_scheduler=bool(getattr(args, "pro_enable_feedback_scheduler", True)),
            pro_feedback_cooldown_repeats=int(getattr(args, "pro_feedback_cooldown_repeats", 1)),
            pro_enable_strategy_embed_match=strat_embed,
            pro_strategy_embed_min_sim=float(getattr(args, "pro_strategy_embed_min_sim", 0.22)),
            pro_enable_prompt_strategy_attribution=prompt_attrib,
            pro_goal_similarity_floor=float(getattr(args, "pro_goal_similarity_floor", 0.15)),
            pro_staged_eval_enabled=bool(getattr(args, "pro_staged_eval_enabled", False)),
            pro_staged_eval_profile=str(getattr(args, "pro_staged_eval_profile", "balanced")),
            pro_staged_filter_keep_ratio=float(getattr(args, "pro_staged_filter_keep_ratio", 0.5)),
            pro_staged_probe_keep_ratio=float(getattr(args, "pro_staged_probe_keep_ratio", 0.5)),
            pro_staged_short_max_new_tokens=int(getattr(args, "pro_staged_short_max_new_tokens", 32)),
            pro_staged_min_candidates_for_full_eval=int(
                getattr(args, "pro_staged_min_candidates_for_full_eval", 1),
            ),
            pro_staged_uncertainty_band=float(getattr(args, "pro_staged_uncertainty_band", 0.1)),
            pro_staged_eval_budget_ms=float(getattr(args, "pro_staged_eval_budget_ms", 0.0)),
            pro_staged_weight_filter=float(getattr(args, "pro_staged_weight_filter", 0.35)),
            pro_staged_weight_probe=float(getattr(args, "pro_staged_weight_probe", 0.65)),
            pro_staged_uncertainty_penalty=float(getattr(args, "pro_staged_uncertainty_penalty", 0.2)),
            pro_verbose_pipeline_logs=bool(getattr(args, "pro_verbose_pipeline_logs", False)),
            pro_enable_threshold_telemetry=not bool(
                getattr(args, "pro_disable_threshold_telemetry", False),
            ),
            pro_telemetry_jsonl=getattr(args, "pro_telemetry_jsonl", None),
            pro_dynamic_pattern_select=bool(getattr(args, "pro_dynamic_pattern_select", False)),
            pro_pattern_exploit_n=int(getattr(args, "pro_pattern_exploit_n", 3)),
            pro_pattern_explore_n=int(getattr(args, "pro_pattern_explore_n", 2)),
            pro_pattern_rank_w_rate=float(getattr(args, "pro_pattern_rank_w_rate", 0.3)),
            pro_pattern_rank_w_avg=float(getattr(args, "pro_pattern_rank_w_avg", 0.3)),
            pro_pattern_rank_w_req=float(getattr(args, "pro_pattern_rank_w_req", 0.4)),
            pro_pattern_explore_seed=getattr(args, "pro_pattern_explore_seed", None),
            pro_four_tier_eval=bool(getattr(args, "pro_four_tier_eval", False)),
            pro_verifier_top_n=int(getattr(args, "pro_verifier_top_n", 2)),
            pro_rotate_explore_across_candidates=bool(
                getattr(args, "pro_rotate_explore_across_candidates", False),
            ),
            pro_per_candidate_strategy_bundles=bool(
                getattr(args, "pro_per_candidate_strategy_bundles", False),
            ),
        )
