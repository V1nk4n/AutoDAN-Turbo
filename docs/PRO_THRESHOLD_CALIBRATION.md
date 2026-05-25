# PRO threshold calibration (warm-up)

Use a **warm-up-only** run to tune gates and rank weights, then analyze telemetry before lifelong training.

**Step-by-step workflow:** see [PRO_WARMUP_STEP_BY_STEP.md](PRO_WARMUP_STEP_BY_STEP.md).

**Quick Gate 0 after smoke run:** `python scripts/pro_smoke_gate.py --latest`

## Recommended warm-up command

```bash
python main.py \
  --pro_enabled \
  --only_warm_up \
  --epochs 50 \
  --log_every 5 \
  --use_local_embedding \
  --pattern_filepath ./logs/pattern_library_warmup_calib.json \
  --pro_four_tier_eval \
  --pro_dynamic_pattern_select \
  --pro_per_candidate_strategy_bundles \
  --pro_explore_n_candidates 2 \
  --pro_exploit_n_candidates 4 \
  --pro_explore_max_new_tokens 512 \
  --pro_exploit_max_new_tokens 1024 \
  --pro_eval_batch_size 6 \
  --pro_tier1_min_response_chars 24 \
  --pro_goal_similarity_floor 0.15
```

Notes:

- `--epochs` is **repeats per harmful request**, not the number of warm-up requests.
- `--target_max_new_tokens` does **not** control PRO repeats; decode length is set by `--pro_explore_max_new_tokens` and `--pro_exploit_max_new_tokens` per phase.
- `--pro_eval_batch_size` is auto-raised to at least `max(explore_n, exploit_n)` in config; set it explicitly to match batch decode (e.g. 6 for 4 exploit candidates).

## After the run

```bash
python scripts/analyze_pro_thresholds.py --latest --out threshold_report.md
```

Check:

1. **repeat_summary coverage** ≈ 100% of attack_log repeats (including skipped).
2. **tier1_short_circuit** histogram vs `pro_tier1_min_response_chars`.
3. **goal_response_sim** (not only `goal_prompt_sim`) for floor tuning.
4. **Eval funnel**: tier1_short share; if >85%, raise min response chars or target decode tokens.

## Staged training

```bash
# 1) Warm-up (this doc)
python main.py --pro_enabled --only_warm_up ...

# 2) Lifelong chunk 1 (loads warm-up state)
python main.py --pro_enabled --only_lifelong 1 ...

# 3) Lifelong chunk 2+ (loads previous lifelong)
python main.py --pro_enabled --only_lifelong 2 ...
```

State: `logs/warm_up_*`, `logs/lifelong_*`, `pattern_library.json`, `epoch_refine_memory.json`.

## Key thresholds

| Flag | Role |
|------|------|
| `pro_tier1_min_response_chars` | Skip NLL/dual when target response shorter than this |
| `pro_goal_similarity_floor` | Semantic prune absolute floor + success gate (`goal_response_sim`) |
| `pro_goal_prune_relative_ratio` | Prune also requires sim ≥ ratio × max(candidate sim); always caps to `pro_top_k` |
| `pro_strategy_embed_min_sim` | Strategy attribution from prompt embedding (default 0.28) |
| `pro_strategy_embed_min_margin` | Top-1 credit when best−second ≥ margin; else slow-path combo |
| `pro_pattern_rank_w_*` | Dynamic pattern S_rank blend |
| `pro_pattern_rank_low_rate_penalty` | Down-rank strategies with rate <5% after ≥3 trials |
| `nll_min` / `nll_max` | Linear map NLL → `score_loss` on **0–10** (not a gate; tune from telemetry p5/p95) |
| `pro_early_stop_min_delta` | Min **score_loss** gain vs best-so-far to reset plateau (0–10; CLI &lt;1 is ×10) |
| `pro_score_threshold` | **Unused** in `pipeline_pro.py` (legacy CLI only) |

Telemetry: `logs/logs_per_run/<run>/pro_threshold_telemetry.jsonl`.
