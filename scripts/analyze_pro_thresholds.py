#!/usr/bin/env python3
"""Analyze PRO runs: thresholds, timing, success, and strategy library health.

Data sources (auto-resolved with ``--latest`` / ``--run-dir``):

- ``pro_threshold_telemetry.jsonl`` — thresholds, eval, pattern rank, library credit
- ``running.log`` — ``[PRO timing]`` latency lines
- ``warm_up_attack_log.json`` (or ``--attack-log``) — per-repeat success and timings
- ``pattern_library.json`` (or ``--pattern-library``) — strategy metrics snapshot

Example::

    python scripts/analyze_pro_thresholds.py --latest --out threshold_report.md
    python scripts/analyze_pro_thresholds.py --run-dir logs/logs_per_run/2026-05-22_14-41-47_682198 --out report.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
_tel_path = _ROOT / "framework" / "pro_threshold_telemetry.py"
_spec = importlib.util.spec_from_file_location("pro_threshold_telemetry", _tel_path)
assert _spec and _spec.loader
_tel_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tel_mod)
UNUSED_CONFIG_KEYS = _tel_mod.UNUSED_CONFIG_KEYS
effective_score_loss_threshold = _tel_mod.effective_score_loss_threshold

_TIMING_LINE_RE = re.compile(r"\[PRO timing\]\s+(\S+)\s+(.+)$")
_TIMING_KV_RE = re.compile(r"(\w+)=([^\s]+)")


@dataclass
class RunArtifacts:
    run_dir: Optional[Path]
    telemetry: List[Dict[str, Any]]
    timing_events: List[Dict[str, Any]]
    attack_log: List[Dict[str, Any]]
    pattern_library: Optional[Dict[str, Any]]


def _human_ms(ms: float) -> str:
    if ms < 1000.0:
        return f"{ms:.0f}ms"
    if ms < 60_000.0:
        return f"{ms / 1000.0:.1f}s"
    return f"{ms / 60_000.0:.1f}m"


def _parse_timing_tail(tail: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in _TIMING_KV_RE.findall(tail):
        if key.endswith("_ms"):
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
        elif key in ("success",):
            out[key] = val.lower() in ("true", "1", "yes")
        elif val.isdigit():
            out[key] = int(val)
        else:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


def _load_pro_timing_log(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not path.is_file():
        return events
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _TIMING_LINE_RE.search(line)
            if not m:
                continue
            ev = {"event": m.group(1)}
            ev.update(_parse_timing_tail(m.group(2)))
            events.append(ev)
    return events


def _load_attack_log(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _load_pattern_library(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict) and "strategies" in data:
        return data
    if isinstance(data, dict):
        return {"strategies": data}
    return None


def _resolve_run_artifacts(
    *,
    jsonl: Optional[Path],
    log: Optional[Path],
    run_dir: Optional[Path],
    attack_log: Optional[Path],
    pattern_library: Optional[Path],
    latest: bool,
) -> RunArtifacts:
    root = _ROOT
    tel_path: Optional[Path] = jsonl
    rdir: Optional[Path] = run_dir

    if latest and tel_path is None:
        candidates = sorted((root / "logs/logs_per_run").glob("*/pro_threshold_telemetry.jsonl"))
        if not candidates:
            raise SystemExit("No telemetry under logs/logs_per_run")
        tel_path = candidates[-1]
        print(f"Using latest telemetry: {tel_path}", flush=True)

    if tel_path is not None:
        rdir = rdir or tel_path.parent

    timing_path = log
    if timing_path is None and rdir is not None:
        cand = rdir / "running.log"
        if cand.is_file():
            timing_path = cand

    atk_path = attack_log
    if atk_path is None:
        for cand in (
            (rdir / "warm_up_attack_log.json") if rdir else None,
            root / "logs/warm_up_attack_log.json",
            root / "logs/warm_up_attack_log_debug.json",
        ):
            if cand is not None and cand.is_file():
                atk_path = cand
                break

    lib_path = pattern_library or (root / "logs/pattern_library.json")

    telemetry = _load_jsonl(tel_path) if tel_path else []
    if not telemetry and timing_path is None and atk_path is None:
        raise SystemExit("No input data found.")

    return RunArtifacts(
        run_dir=rdir,
        telemetry=telemetry,
        timing_events=_load_pro_timing_log(timing_path) if timing_path else [],
        attack_log=_load_attack_log(atk_path) if atk_path else [],
        pattern_library=_load_pattern_library(lib_path) if lib_path else None,
    )


def _percentiles(values: Sequence[float], ps: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {f"p{int(p)}": float("nan") for p in ps}
    xs = sorted(float(v) for v in values)
    n = len(xs)
    out: Dict[str, float] = {}
    for p in ps:
        if n == 1:
            out[f"p{int(p)}"] = xs[0]
            continue
        k = (n - 1) * (p / 100.0)
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            out[f"p{int(p)}"] = xs[lo]
        else:
            w = k - lo
            out[f"p{int(p)}"] = xs[lo] * (1.0 - w) + xs[hi] * w
    return out


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_from_running_log(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pat = re.compile(r"\[PRO\]\s*(\{.*\})\s*$")
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if not m:
                continue
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and ("kind" in obj or "event" in obj):
                if "kind" not in obj and "event" in obj:
                    obj = dict(obj)
                    obj["kind"] = obj.pop("event", "unknown")
                rows.append(obj)
    return rows


def _fmt_stats(values: Sequence[float]) -> str:
    if not values:
        return "(empty)"
    p = _percentiles(values, [10, 50, 90])
    return (
        f"n={len(values)} min={min(values):.4f} max={max(values):.4f} "
        f"mean={statistics.mean(values):.4f} median={statistics.median(values):.4f} "
        f"p10={p['p10']:.4f} p90={p['p90']:.4f}"
    )


def _config_notes(config: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for k in UNUSED_CONFIG_KEYS:
        if k in config:
            lines.append(
                f"- **Warning**: `{k}={config[k]}` is recorded in config but "
                "is not applied in `pipeline_pro.py` (PRO uses J / score_loss / gates)."
            )
    raw_delta = config.get("pro_early_stop_min_delta")
    if raw_delta is not None:
        eff = effective_score_loss_threshold(float(raw_delta))
        if eff != float(raw_delta):
            lines.append(
                f"- **Note**: `pro_early_stop_min_delta` CLI value `{raw_delta}` maps to "
                f"effective **{eff:.3f}** on the 0–10 score_loss scale (legacy fractions "
                "in (0,1) are multiplied by 10). Use `--pro_early_stop_min_delta 0.1` "
                "for effective 1.0, not `1.0` unless you intend 1.0 on the 0–10 scale."
            )
        else:
            lines.append(
                f"- `pro_early_stop_min_delta` effective on 0–10 scale: **{eff:.3f}**"
            )
    return lines


def _counterfactual_embed_thresholds(
    events: List[Dict[str, Any]],
    thresholds: Sequence[float],
) -> List[str]:
    lines: List[str] = []
    max_sims: List[float] = []
    for ev in events:
        if ev.get("max_sim") is not None:
            max_sims.append(float(ev["max_sim"]))

    if not max_sims:
        return ["No strategy_attribution events with max_sim."]

    lines.append(f"Attribution events: {len(max_sims)}")
    lines.append(f"max_sim: {_fmt_stats(max_sims)}")
    lines.append("")
    lines.append("| threshold | n_match>=1 | n_match==1 | n_match>=2 | n_match==0 |")
    lines.append("|-----------|------------|------------|------------|------------|")
    for t in thresholds:
        c0 = c1 = c2 = cp = 0
        for ev in events:
            sims = [
                float(x.get("sim", 0.0))
                for x in (ev.get("similarities") or [])
                if isinstance(x, dict)
            ]
            if not sims and ev.get("max_sim") is not None:
                n = 1 if float(ev["max_sim"]) >= t else 0
            else:
                n = sum(1 for s in sims if s >= t)
            if n == 0:
                c0 += 1
            elif n == 1:
                c1 += 1
            else:
                c2 += 1
            if n >= 1:
                cp += 1
        lines.append(f"| {t:.2f} | {cp} | {c1} | {c2} | {c0} |")
    lines.append("")
    lines.append(
        "Tip: prefer thresholds where n_match==1 is high for jailbreak fast-path, "
        "and n_match>=2 is low if you rely on slow-path for combos."
    )
    return lines


def _counterfactual_goal_floor(
    prune_events: List[Dict[str, Any]],
    floors: Sequence[float],
) -> List[str]:
    lines: List[str] = []
    all_sims: List[float] = []
    for ev in prune_events:
        for item in ev.get("similarities") or []:
            if isinstance(item, dict) and "sim" in item:
                all_sims.append(float(item["sim"]))
    if not all_sims:
        return ["No semantic_prune similarity data."]

    lines.append(f"Candidate goal similarities (all slots): n={len(all_sims)}")
    lines.append(_fmt_stats(all_sims))
    lines.append("")
    lines.append("| floor | kept (sim>=floor) | dropped |")
    lines.append("|-------|-------------------|---------|")
    for f in floors:
        kept = sum(1 for s in all_sims if s >= f)
        lines.append(f"| {f:.2f} | {kept} | {len(all_sims) - kept} |")
    return lines


def _success_gate_analysis(
    eval_events: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    floor = float(config.get("pro_goal_similarity_floor", 0.15))

    eval_gs = [float(ev["goal_sim"]) for ev in eval_events if ev.get("goal_sim") is not None]
    if eval_gs:
        lines.append(f"Eval candidate goal_sim: {_fmt_stats(eval_gs)}")

    j_raw = sum(1 for ev in eval_events if ev.get("J_raw_jailbroken") or ev.get("is_jailbroken"))
    j_qual = sum(1 for ev in eval_events if ev.get("success_qualified"))
    gated = j_raw - j_qual
    if eval_events:
        lines.append(
            f"Among eval_candidate rows: J_raw/jailbroken={j_raw}, "
            f"success_qualified={j_qual}, gated_out={gated}"
        )

    rep_gs = [float(s["goal_sim"]) for s in summaries if s.get("goal_sim") is not None]
    if rep_gs:
        lines.append(f"Repeat-summary best goal_sim: {_fmt_stats(rep_gs)}")
        lines.append("")
        lines.append("| floor | repeats goal_pass | repeats would fail gate |")
        lines.append("|-------|-------------------|-------------------------|")
        for f in [round(x * 0.02, 2) for x in range(3, 16)]:
            pass_n = sum(1 for s in summaries if float(s.get("goal_sim", 0)) >= f)
            fail_n = len(summaries) - pass_n
            lines.append(f"| {f:.2f} | {pass_n} | {fail_n} |")
        lines.append(f"\nConfigured floor: **{floor:.3f}**")
    elif not eval_gs:
        return ["No goal_sim in eval_candidate or repeat_summary."]
    return lines


def _pattern_rank_analysis(events: List[Dict[str, Any]]) -> List[str]:
    if not events:
        return ["No pattern_rank events (enable dynamic pattern select + embeddings)."]

    lines: List[str] = [f"Pattern rank snapshots: {len(events)}"]
    selected_req: List[float] = []
    borderline: List[float] = []
    for ev in events:
        table = ev.get("rank_table") or []
        sel = {r.get("strategy_id") for r in table if r.get("selected")}
        for row in table:
            if not isinstance(row, dict):
                continue
            rs = float(row.get("req_sim", 0))
            if row.get("strategy_id") in sel:
                selected_req.append(rs)
            elif row.get("rank", 99) < int(ev.get("n_selected", 0)) + 3:
                borderline.append(rs)
    if selected_req:
        lines.append(f"req_sim (selected strategies): {_fmt_stats(selected_req)}")
    last = events[-1]
    w_rate = last.get("w_rate")
    w_avg = last.get("w_avg")
    w_req = last.get("w_req")
    if w_avg is not None or w_rate is not None:
        lines.append(
            f"Last run weights: w_rate={w_rate}, w_avg={w_avg}, w_req={w_req}"
        )
    lines.append(
        "Tip: if selected req_sim is low vs library top ranks, increase w_req or "
        "check embedding quality; if exploit picks dominate, tune exploit_n/explore_n."
    )
    return lines


def _judge_and_fast_analysis(
    eval_events: List[Dict[str, Any]],
    tier2_events: List[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    all_ev = eval_events + tier2_events
    if not all_ev:
        return ["No eval_candidate / eval_tier2 judge data."]

    lane_ctr = Counter(str(ev.get("judge_lane", "")) for ev in all_ev)
    lines.append("Judge lane counts:")
    for k, v in lane_ctr.most_common():
        if k:
            lines.append(f"  {k}: {v}")

    fj_ctr = Counter(str(ev.get("fast_judge_decision", "")) for ev in all_ev)
    fj_ctr.pop("", None)
    if fj_ctr:
        lines.append("FastJudge decisions:")
        for k, v in fj_ctr.most_common():
            lines.append(f"  {k}: {v}")

    disagree = sum(
        1 for ev in all_ev
        if ev.get("dual_agree") is False and ev.get("dual_called")
    )
    dual_n = sum(1 for ev in all_ev if ev.get("dual_called") or ev.get("judge_a"))
    if dual_n:
        lines.append(f"Dual judge disagreement (when logged): {disagree}/{dual_n}")

    src_ctr = Counter(str(ev.get("score_source", "")) for ev in eval_events if ev.get("score_source"))
    if src_ctr:
        lines.append("score_source (eval_candidate):")
        for k, v in src_ctr.most_common(8):
            lines.append(f"  {k}: {v}")
    return lines


def _nll_score_loss_analysis(
    eval_events: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    nlls_ok = [float(ev["nll"]) for ev in eval_events if ev.get("nll_ok") and ev.get("nll") is not None]
    losses = [float(ev["score_loss"]) for ev in eval_events if ev.get("score_loss") is not None]
    if not nlls_ok and not losses:
        return ["No eval_candidate / eval_tier2 NLL data."]

    lo = float(config.get("nll_min", 0.0))
    hi = float(config.get("nll_max", 10.0))

    if nlls_ok:
        lines.append(f"NLL (n={len(nlls_ok)}, configured map [{lo}, {hi}] -> score_loss 0..10):")
        lines.append(_fmt_stats(nlls_ok))
        p = _percentiles(nlls_ok, [5, 25, 50, 75, 95])
        lines.append(
            "Suggested nll_min/nll_max from percentiles: "
            f"nll_min≈{p['p5']:.4f} (p5), nll_max≈{p['p95']:.4f} (p95)"
        )
    if losses:
        jb = [
            float(ev["score_loss"])
            for ev in eval_events
            if ev.get("success_qualified") or ev.get("is_jailbroken")
        ]
        fail = [
            float(ev["score_loss"])
            for ev in eval_events
            if not (ev.get("success_qualified") or ev.get("is_jailbroken"))
        ]
        lines.append(f"score_loss all (n={len(losses)}): {_fmt_stats(losses)}")
        if jb:
            lines.append(f"  success/qualified subset: {_fmt_stats(jb)}")
        if fail:
            lines.append(f"  failed subset: {_fmt_stats(fail)}")
        if jb and fail:
            med_j = statistics.median(jb)
            med_f = statistics.median(fail)
            lines.append(
                f"  median gap (success - failed): {med_j - med_f:.3f}"
            )
    return lines


def _early_stop_analysis(
    summaries: List[Dict[str, Any]],
    stop_events: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    if stop_events:
        ctr = Counter(str(ev.get("reason", "")) for ev in stop_events)
        lines.append(f"early_stop events: {len(stop_events)}")
        for k, v in ctr.most_common():
            lines.append(f"  {k}: {v}")

    scores = [float(s["best_score_loss"]) for s in summaries if s.get("best_score_loss") is not None]
    if len(scores) < 2:
        if not stop_events:
            return ["Not enough repeat_summary events for early-stop delta analysis."]
        lines.append("(Only one repeat score per request in telemetry; delta sweep skipped.)")
        return lines

    deltas = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
    pos = [d for d in deltas if d > 0]
    lines.append(f"Repeat best_score_loss deltas (n={len(deltas)}): {_fmt_stats(deltas)}")
    if pos:
        lines.append(f"  positive improvements only: {_fmt_stats(pos)}")
    cur_delta = effective_score_loss_threshold(
        float(config.get("pro_early_stop_min_delta", 1.0))
    )
    patience = int(config.get("pro_early_stop_patience", 2))
    lines.append(
        f"Current pro_early_stop_min_delta (effective)={cur_delta:.3f}, "
        f"patience={patience}. "
        f"Deltas >= threshold: {sum(1 for d in deltas if d >= cur_delta)}/{len(deltas)}"
    )
    for cand in (0.05, 0.1, 0.2, 0.5, 1.0):
        eff = effective_score_loss_threshold(cand)
        lines.append(
            f"  if CLI={cand:.2f} (effective {eff:.2f}): "
            f"{sum(1 for d in deltas if d >= eff)} improvements counted"
        )
    for pat in (1, 2, 3, 4):
        lines.append(
            f"  if patience={pat}: plateau stops would be "
            f"{sum(1 for d in deltas if d < cur_delta)} repeats without >=delta improvement "
            f"(heuristic; actual stop uses per-request streak)"
        )
    return lines


def _library_credit_summary(events: List[Dict[str, Any]]) -> List[str]:
    if not events:
        return ["No library_credit events."]
    ctr = Counter(str(ev.get("outcome", "unknown")) for ev in events)
    lines = ["Library credit outcomes:"]
    for k, v in ctr.most_common():
        lines.append(f"  {k}: {v}")
    return lines


def _phase_breakdown(rows: List[Dict[str, Any]]) -> List[str]:
    phases = Counter(str(r.get("phase", "")) for r in rows if r.get("phase"))
    if not phases or phases == Counter({"": len(rows)}):
        return []
    lines = ["Phase distribution (telemetry rows):"]
    for k, v in phases.most_common():
        if k:
            lines.append(f"  {k}: {v}")
    return lines


def _run_overview_section(artifacts: RunArtifacts) -> List[str]:
    lines = ["## Run overview\n"]
    if artifacts.run_dir:
        lines.append(f"- **Run directory**: `{artifacts.run_dir}`")
    lines.append(f"- **Telemetry events**: {len(artifacts.telemetry)}")
    lines.append(f"- **Timing events** (`[PRO timing]`): {len(artifacts.timing_events)}")
    lines.append(f"- **Attack log entries**: {len(artifacts.attack_log)}")
    if artifacts.pattern_library:
        n_strat = len(artifacts.pattern_library.get("strategies") or {})
        lines.append(f"- **Pattern library strategies** (snapshot): {n_strat}")
    else:
        lines.append("- **Pattern library**: not loaded (pass `--pattern-library`)")
    return lines


def _runtime_performance_section(artifacts: RunArtifacts) -> List[str]:
    lines: List[str] = []
    timing = artifacts.timing_events
    attack = artifacts.attack_log
    if not timing and not attack:
        return ["## Runtime & performance\n", "No timing or attack_log data.\n"]

    lines.append("## Runtime & performance\n")

    by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in timing:
        by_event[str(ev.get("event", ""))].append(ev)

    for ev_name in (
        "warm_up_phase_complete",
        "dataset_stage_complete",
        "pipeline_run_complete",
    ):
        for ev in by_event.get(ev_name, []):
            w = ev.get("wall_ms")
            if w is not None:
                lines.append(
                    f"- **{ev_name}** wall time: {_human_ms(float(w))} "
                    f"({float(w):.0f} ms)"
                )

    repeats = by_event.get("repeat_complete", [])
    if repeats:
        totals = [float(e["total_ms"]) for e in repeats if e.get("total_ms") is not None]
        attacks = [float(e["attack_ms"]) for e in repeats if e.get("attack_ms") is not None]
        fbs = [float(e["feedback_ms"]) for e in repeats if e.get("feedback_ms") is not None]
        if totals:
            lines.append(f"- **Repeats** (from log): n={len(totals)}")
            lines.append(f"  - total_ms: {_fmt_stats(totals)}")
            lines.append(
                f"  - median repeat: {_human_ms(statistics.median(totals))}"
            )
        if attacks and totals:
            fb_share = [
                float(f) / float(t) if float(t) > 0 else 0.0
                for f, t in zip(fbs, totals)
            ]
            lines.append(
                f"  - feedback time share of repeat: "
                f"mean={statistics.mean(fb_share):.1%} median={statistics.median(fb_share):.1%}"
            )

    req_done = by_event.get("request_complete", [])
    if req_done:
        walls = [float(e["wall_ms"]) for e in req_done if e.get("wall_ms") is not None]
        avgs = [float(e["avg_repeat_ms"]) for e in req_done if e.get("avg_repeat_ms") is not None]
        if walls:
            lines.append(f"- **Per-request wall** (n={len(walls)}): {_fmt_stats(walls)}")
            slow = sorted(req_done, key=lambda e: float(e.get("wall_ms", 0)), reverse=True)[:5]
            lines.append("  - Slowest requests:")
            for e in slow:
                lines.append(
                    f"    - request_id={e.get('request_id')} wall={_human_ms(float(e['wall_ms']))} "
                    f"repeats={e.get('repeats_completed')}/{e.get('repeats_max')}"
                )
        if avgs:
            lines.append(f"  - avg_repeat_ms per request: {_fmt_stats(avgs)}")

    if attack:
        atk_totals = [
            float(x["time_ms_total"])
            for x in attack
            if x.get("time_ms_total") is not None
        ]
        if atk_totals:
            lines.append(f"- **Attack log** repeat total_ms: {_fmt_stats(atk_totals)}")
        err_n = sum(1 for x in attack if x.get("error"))
        if err_n:
            lines.append(f"- **Attack log errors**: {err_n}")

    if req_done and walls:
        total_wall = sum(walls)
        n_req = len(walls)
        if total_wall > 0:
            lines.append(
                f"- **Throughput**: {n_req} requests in {_human_ms(total_wall)} "
                f"≈ {n_req * 3_600_000 / total_wall:.2f} requests/hour (request wall only)"
            )

    return lines


def _success_quality_section(
    artifacts: RunArtifacts,
    by_kind: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    lines = ["## Success & quality\n"]
    attack = artifacts.attack_log
    summaries = by_kind.get("repeat_summary", [])
    eval_ev = by_kind.get("eval_candidate", []) + by_kind.get("eval_tier2", [])
    early = by_kind.get("early_stop", [])

    if attack:
        n_rep = len(attack)
        n_ok_rep = sum(1 for x in attack if x.get("success"))
        req_ids = {int(x["request_id"]) for x in attack if x.get("request_id") is not None}
        ok_req = {
            int(x["request_id"])
            for x in attack
            if x.get("success") and x.get("request_id") is not None
        }
        lines.append(f"- **Repeat-level success**: {n_ok_rep}/{n_rep} ({100*n_ok_rep/n_rep:.2f}%)")
        if req_ids:
            lines.append(
                f"- **Request-level success** (≥1 successful repeat): "
                f"{len(ok_req)}/{len(req_ids)} ({100*len(ok_req)/len(req_ids):.2f}%)"
            )
        phases = Counter(str(x.get("phase", "")) for x in attack if x.get("phase"))
        if phases:
            lines.append("- **Success by phase (repeats)**:")
            for ph, cnt in phases.most_common():
                ok = sum(1 for x in attack if x.get("phase") == ph and x.get("success"))
                lines.append(f"  - {ph or '(none)'}: {ok}/{cnt}")
        scores = [
            float(x["best_score_loss"])
            for x in attack
            if x.get("best_score_loss") is not None
        ]
        if scores:
            lines.append(f"- **best_score_loss** (attack log): {_fmt_stats(scores)}")
        es = Counter(str(x.get("early_stop_reason") or "") for x in attack)
        es.pop("", None)
        if es:
            lines.append("- **early_stop_reason** (attack log):")
            for k, v in es.most_common():
                lines.append(f"  - {k}: {v}")
        fb = sum(1 for x in attack if x.get("feedback_called"))
        lines.append(f"- **Feedback invoked**: {fb}/{n_rep} repeats ({100*fb/n_rep:.1f}%)")

    if summaries:
        tel_ok = sum(1 for s in summaries if s.get("success") or s.get("success_qualified"))
        lines.append(
            f"- **Telemetry repeat_summary success**: {tel_ok}/{len(summaries)}"
        )

    if eval_ev:
        tier1 = sum(1 for e in eval_ev if str(e.get("judge_lane", "")) == "tier1_short")
        dual = sum(1 for e in eval_ev if "dual" in str(e.get("judge_lane", "")))
        lines.append(
            f"- **Eval funnel**: tier1_short={tier1}/{len(eval_ev)} "
            f"({100*tier1/len(eval_ev):.1f}%), dual path={dual}/{len(eval_ev)}"
        )
        j_qual = sum(1 for e in eval_ev if e.get("success_qualified"))
        j_raw = sum(1 for e in eval_ev if e.get("J_raw_jailbroken") or e.get("is_jailbroken"))
        if j_raw:
            lines.append(
                f"- **Candidate eval**: J_raw={j_raw}, success_qualified={j_qual}, "
                f"gated={j_raw - j_qual}"
            )

    if early:
        ctr = Counter(str(e.get("reason", "")) for e in early)
        lines.append(f"- **Early-stop events**: {len(early)}")
        for k, v in ctr.most_common():
            lines.append(f"  - {k}: {v}")

    if not attack and not summaries:
        lines.append("No attack_log or repeat_summary — success stats unavailable.")

    return lines


def _strategy_usage_section(
    by_kind: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    lines = ["## Strategy usage (selection & attribution)\n"]
    rank_ev = by_kind.get("pattern_rank", [])
    attrib = by_kind.get("strategy_attribution", [])
    credit = by_kind.get("library_credit", [])

    selected_ctr: Counter[str] = Counter()
    for ev in rank_ev:
        for sid in ev.get("selected_ids") or []:
            selected_ctr[str(sid)] += 1

    if selected_ctr:
        lines.append(f"- **Top selected strategies** (from `pattern_rank`, n={len(rank_ev)} repeats):")
        for sid, cnt in selected_ctr.most_common(12):
            lines.append(f"  - `{sid}`: {cnt}")
        never = set()
        if rank_ev:
            all_ids = {str(r.get("strategy_id")) for r in (rank_ev[-1].get("rank_table") or []) if r.get("strategy_id")}
            picked = set(selected_ctr.keys())
            never = all_ids - picked
            if never and len(all_ids) <= 40:
                lines.append(f"- **Never selected** in this run ({len(never)}): {', '.join(sorted(never)[:15])}{'…' if len(never)>15 else ''}")

    if attrib:
        match0 = sum(1 for a in attrib if int(a.get("match_count", 0)) == 0)
        match1 = sum(1 for a in attrib if int(a.get("match_count", 0)) == 1)
        match2p = sum(1 for a in attrib if int(a.get("match_count", 0)) >= 2)
        lines.append(
            f"- **Prompt attribution** (n={len(attrib)}): "
            f"0 match={match0}, 1 match={match1}, ≥2 match={match2p}"
        )

    if credit:
        lines.append("- **Library credit events** (expanded):")
        for outcome, cnt in Counter(str(c.get("outcome", "")) for c in credit).most_common():
            lines.append(f"  - {outcome}: {cnt}")
        fail_ids: Counter[str] = Counter()
        for c in credit:
            if str(c.get("outcome", "")) == "failure_save_attempt":
                for sid in c.get("strategy_ids") or []:
                    fail_ids[str(sid)] += 1
        if fail_ids:
            lines.append("  - Top strategies on failed attempts (save_attempt):")
            for sid, n in fail_ids.most_common(8):
                lines.append(f"    - `{sid}`: {n}")

    if not rank_ev and not attrib:
        lines.append("No pattern_rank / strategy_attribution telemetry.")

    return lines


def _strategy_metrics_from_lib(info: Dict[str, Any]) -> Dict[str, float]:
    m = info.get("metrics", {}) if isinstance(info.get("metrics"), dict) else {}
    freq = int(m.get("freq", 0))
    trials = max(int(m.get("trial_count", 0)), freq)
    rate = float(freq) / float(trials) if trials > 0 else 0.0
    return {
        "avg_score": float(m.get("avg_score", 0.0)),
        "freq": float(freq),
        "trials": float(trials),
        "rate": rate,
    }


def _strategy_effectiveness_section(
    artifacts: RunArtifacts,
    by_kind: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    lines = ["## Strategy effectiveness (library snapshot)\n"]
    lib = artifacts.pattern_library
    if not lib:
        lines.append("Pattern library not loaded.\n")
        return lines

    strategies = lib.get("strategies") or {}
    if not isinstance(strategies, dict) or not strategies:
        lines.append("Empty pattern library.\n")
        return lines

    rows: List[Tuple[str, Dict[str, float], str]] = []
    for sid, info in strategies.items():
        if not isinstance(info, dict):
            continue
        met = _strategy_metrics_from_lib(info)
        name = str(info.get("name", "") or sid)[:48]
        rows.append((str(sid), met, name))

    selected_ctr: Counter[str] = Counter()
    for ev in by_kind.get("pattern_rank", []):
        for sid in ev.get("selected_ids") or []:
            selected_ctr[str(sid)] += 1

    credit = by_kind.get("library_credit", [])
    success_ids: Counter[str] = Counter()
    for c in credit:
        if str(c.get("outcome", "")) in ("single_strategy_fast", "slow_path"):
            for sid in c.get("credited_ids") or []:
                success_ids[str(sid)] += 1

    lines.append(f"- **Library size**: {len(rows)} strategies")

    by_rate = sorted(rows, key=lambda x: x[1]["rate"], reverse=True)
    tried = [r for r in rows if r[1]["trials"] >= 1]
    if tried:
        lines.append("- **Top historical success rate** (freq/trial_count, min 1 trial):")
        for sid, met, name in by_rate[:8]:
            lines.append(
                f"  - `{sid}`: rate={met['rate']:.2%} trials={int(met['trials'])} "
                f"avg_score={met['avg_score']:.2f} picks={selected_ctr.get(sid, 0)}"
            )

    weak = [
        r for r in rows
        if r[1]["trials"] >= 3 and r[1]["rate"] < 0.05 and selected_ctr.get(r[0], 0) >= 2
    ]
    weak.sort(key=lambda x: selected_ctr.get(x[0], 0), reverse=True)
    if weak:
        lines.append("- **Likely ineffective** (≥3 trials, rate<5%, selected ≥2 times):")
        for sid, met, name in weak[:8]:
            lines.append(
                f"  - `{sid}`: rate={met['rate']:.2%} picks={selected_ctr[sid]} "
                f"avg_score={met['avg_score']:.2f}"
            )
    else:
        low_avg = sorted(rows, key=lambda x: x[1]["avg_score"])[:5]
        lines.append("- **Lowest avg_score** (snapshot):")
        for sid, met, name in low_avg:
            lines.append(
                f"  - `{sid}`: avg_score={met['avg_score']:.2f} rate={met['rate']:.2%} "
                f"picks={selected_ctr.get(sid, 0)}"
            )

    if success_ids:
        lines.append("- **Credited on jailbreak success**:")
        for sid, n in success_ids.most_common(8):
            met = next((m for s, m, _ in rows if s == sid), None)
            extra = f" rate={met['rate']:.2%}" if met else ""
            lines.append(f"  - `{sid}`: {n}{extra}")

    unused = [sid for sid, _, _ in rows if selected_ctr.get(sid, 0) == 0]
    if unused and len(unused) <= 20:
        lines.append(f"- **Never picked in pattern_rank** ({len(unused)}): {', '.join(unused[:12])}{'…' if len(unused)>12 else ''}")

    return lines


def _new_strategy_discovery_section(by_kind: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    lines = ["## New strategy discovery (slow path)\n"]
    credit = by_kind.get("library_credit", [])
    if not credit:
        lines.append("No library_credit events.\n")
        return lines

    slow = [c for c in credit if str(c.get("outcome", "")) == "slow_path"]
    fast = [c for c in credit if str(c.get("outcome", "")) == "single_strategy_fast"]
    fail = [c for c in credit if str(c.get("outcome", "")) == "failure_save_attempt"]
    n_credit = len(credit)
    n_slow = len(slow)

    lines.append(f"- **slow_path** (summarize new pattern): {n_slow}")
    lines.append(f"- **single_strategy_fast** (known strategy success): {len(fast)}")
    lines.append(f"- **failure_save_attempt**: {len(fail)}")
    if n_credit:
        lines.append(
            f"- **New-pattern rate**: {n_slow}/{len(fast) + n_slow} of success credits "
            f"({100*n_slow/(len(fast)+n_slow) if fast or slow else 0:.1f}%)"
        )
        lines.append(
            f"- **slow_path / all credit events**: {100*n_slow/n_credit:.2f}%"
        )

    if slow:
        lines.append("- **slow_path score_loss** (when logged):")
        sl = [float(c["score_loss"]) for c in slow if c.get("score_loss") is not None]
        if sl:
            lines.append(f"  - {_fmt_stats(sl)}")

    return lines


def _health_checks_section(
    artifacts: RunArtifacts,
    by_kind: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> List[str]:
    lines = ["## Health checks & anomalies\n"]
    warnings: List[str] = []

    eval_ev = by_kind.get("eval_candidate", []) + by_kind.get("eval_tier2", [])
    if eval_ev:
        tier1 = sum(1 for e in eval_ev if str(e.get("judge_lane", "")) == "tier1_short")
        if tier1 / len(eval_ev) > 0.85:
            warnings.append(
                f">85% eval candidates stop at tier1_short ({tier1}/{len(eval_ev)}) — "
                "target responses often too short; check max_new_tokens / model."
            )
        gs = [float(e["goal_sim"]) for e in eval_ev if e.get("goal_sim") is not None]
        if gs and min(gs) == max(gs) == 1.0:
            warnings.append(
                "All goal_sim=1.0 — goal floor tuning may be meaningless on this run."
            )

    attack = artifacts.attack_log
    if attack:
        if not any(x.get("feedback_called") for x in attack):
            warnings.append("No repeat invoked feedback — refine loop may be inactive.")
        if sum(1 for x in attack if x.get("error")):
            warnings.append("Errors present in attack_log — inspect failed repeats.")

    timing = artifacts.timing_events
    repeats = [e for e in timing if e.get("event") == "repeat_complete"]
    if repeats:
        totals = [float(e["total_ms"]) for e in repeats if e.get("total_ms") is not None]
        if totals and max(totals) > 5 * statistics.median(totals):
            warnings.append(
                "Large repeat latency outliers — check slow requests in Runtime section."
            )

    if config.get("pro_score_threshold") is not None:
        warnings.append("`pro_score_threshold` is set but unused in PRO pipeline.")

    if not warnings:
        lines.append("No major anomalies flagged.")
    else:
        for w in warnings:
            lines.append(f"- ⚠ {w}")

    return lines


def build_report(artifacts: RunArtifacts) -> str:
    rows = artifacts.telemetry
    by_kind: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_kind[str(r.get("kind", "unknown"))].append(r)

    config: Dict[str, Any] = {}
    if by_kind.get("run_config"):
        cfg = by_kind["run_config"][0].get("config")
        if isinstance(cfg, dict):
            config = cfg

    sections: List[str] = []
    sections.append("# PRO run analysis report\n")

    sections.extend(_run_overview_section(artifacts))
    sections.append("")
    sections.extend(_runtime_performance_section(artifacts))
    sections.append("")
    sections.extend(_success_quality_section(artifacts, by_kind))
    sections.append("")
    sections.extend(_strategy_usage_section(by_kind))
    sections.append("")
    sections.extend(_strategy_effectiveness_section(artifacts, by_kind))
    sections.append("")
    sections.extend(_new_strategy_discovery_section(by_kind))
    sections.append("")
    sections.extend(_health_checks_section(artifacts, by_kind, config))
    sections.append("")

    sections.append("---\n")
    sections.append("# Threshold tuning\n")

    if config:
        sections.append("## Active config (from run_config)\n")
        for k in sorted(config.keys()):
            sections.append(f"- **{k}**: `{config[k]}`")
        sections.append("")
        notes = _config_notes(config)
        if notes:
            sections.append("### Config notes\n")
            sections.extend(notes)
            sections.append("")

    phase_lines = _phase_breakdown(rows)
    if phase_lines:
        sections.append("## Phase (explore / exploit)\n")
        sections.extend(phase_lines)
        sections.append("")

    embed_thresholds = [round(x * 0.02, 2) for x in range(5, 21)]
    goal_floors = [round(x * 0.02, 2) for x in range(3, 16)]

    eval_ev = by_kind.get("eval_candidate", []) + by_kind.get("eval_tier2", [])

    sections.append("## Strategy embedding (`pro_strategy_embed_min_sim`)\n")
    sections.extend(
        _counterfactual_embed_thresholds(by_kind.get("strategy_attribution", []), embed_thresholds)
    )

    sections.append("\n## Goal similarity prune (`pro_goal_similarity_floor`)\n")
    sections.extend(_counterfactual_goal_floor(by_kind.get("semantic_prune", []), goal_floors))

    sections.append("\n## Success gate (J vs qualified success)\n")
    sections.extend(
        _success_gate_analysis(
            by_kind.get("eval_candidate", []),
            by_kind.get("repeat_summary", []),
            config,
        )
    )

    sections.append(
        "\n## Pattern rank weights (`pro_pattern_rank_w_rate`, `w_avg`, `w_req`)\n"
    )
    sections.extend(_pattern_rank_analysis(by_kind.get("pattern_rank", [])))

    sections.append("\n## Judge / FastJudge\n")
    sections.extend(
        _judge_and_fast_analysis(
            by_kind.get("eval_candidate", []),
            by_kind.get("eval_tier2", []),
        )
    )

    sections.append("\n## NLL / score_loss (`nll_min`, `nll_max`, ranking)\n")
    sections.extend(_nll_score_loss_analysis(eval_ev, config))

    sections.append("\n## Early stop (`pro_early_stop_min_delta`, `pro_early_stop_patience`)\n")
    sections.extend(
        _early_stop_analysis(
            by_kind.get("repeat_summary", []),
            by_kind.get("early_stop", []),
            config,
        )
    )

    sections.append("\n## Library credit routing\n")
    sections.extend(_library_credit_summary(by_kind.get("library_credit", [])))

    sections.append("\n## Event counts\n")
    for kind, evs in sorted(by_kind.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        sections.append(f"- `{kind}`: {len(evs)}")

    sections.append(
        "\n---\n"
        "**Tip:** Compare two runs with separate reports; watch Runtime, Success, and Health sections first. "
        "Tune thresholds in the bottom section one group at a time."
    )

    return "\n".join(sections) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze PRO runs: thresholds, timing, success, strategies"
    )
    ap.add_argument("--jsonl", type=str, default=None, help="Path to pro_threshold_telemetry.jsonl")
    ap.add_argument(
        "--log",
        type=str,
        default=None,
        help="running.log for [PRO timing] (default: alongside --jsonl in run dir)",
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="logs/logs_per_run/<stamp> directory (infers telemetry + running.log)",
    )
    ap.add_argument(
        "--attack-log",
        type=str,
        default=None,
        help="warm_up_attack_log.json (default: logs/warm_up_attack_log.json)",
    )
    ap.add_argument(
        "--pattern-library",
        type=str,
        default=None,
        help="pattern_library.json snapshot (default: logs/pattern_library.json)",
    )
    ap.add_argument("--out", type=str, default=None, help="Write markdown report here")
    ap.add_argument(
        "--latest",
        action="store_true",
        help="Use newest logs/logs_per_run/*/pro_threshold_telemetry.jsonl",
    )
    args = ap.parse_args()

    if not any((args.jsonl, args.log, args.latest, args.run_dir)):
        raise SystemExit("Provide --jsonl, --log, --run-dir, or --latest")

    artifacts = _resolve_run_artifacts(
        jsonl=Path(args.jsonl) if args.jsonl else None,
        log=Path(args.log) if args.log else None,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        attack_log=Path(args.attack_log) if args.attack_log else None,
        pattern_library=Path(args.pattern_library) if args.pattern_library else None,
        latest=bool(args.latest),
    )

    if not artifacts.telemetry and not artifacts.timing_events and not artifacts.attack_log:
        raise SystemExit("No telemetry, timing, or attack_log data loaded.")

    report = build_report(artifacts)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
