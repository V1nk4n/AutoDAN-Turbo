#!/usr/bin/env python3
"""Quick Gate 0 check after a short PRO smoke run (no full report needed).

Usage:
  python scripts/pro_smoke_gate.py --latest
  python scripts/pro_smoke_gate.py logs/logs_per_run/2026-05-23_07-28-22_970729
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _latest_run_dir(logs_root: Path) -> Optional[Path]:
    candidates = sorted(
        (p for p in logs_root.glob("logs_per_run/*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def evaluate(run_dir: Path) -> int:
    tel_path = run_dir / "pro_threshold_telemetry.jsonl"
    rows = _load_jsonl(tel_path)
    by_kind: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_kind[str(r.get("kind", ""))].append(r)

    gens = by_kind.get("generation_summary", [])
    pfb = by_kind.get("parser_goal_fallback", [])
    n = max(len(gens), 1)
    fallback_rate = len(pfb) / n
    valid_ns = [int(g.get("valid_n", 0) or 0) for g in gens]
    med_valid = statistics.median(valid_ns) if valid_ns else 0.0

    prunes = by_kind.get("semantic_prune", [])
    sel = [int(p.get("n_selected", 0) or 0) for p in prunes]
    med_sel = statistics.median(sel) if sel else 0.0

    stages = Counter()
    for g in gens:
        for st, cnt in (g.get("extraction_stage_counts") or {}).items():
            stages[str(st)] += int(cnt)

    print(f"Run: {run_dir}")
    print(f"  generation_summary events: {len(gens)}")
    print(f"  parser_goal_fallback: {len(pfb)} ({100*fallback_rate:.1f}%)")
    print(f"  valid_n median: {med_valid:.1f}")
    print(f"  prune n_selected median: {med_sel:.1f}")
    if stages:
        print(f"  extraction stages: {dict(stages.most_common(5))}")

    g0 = fallback_rate < 0.10 and med_valid >= 2.0
    n_sel_zero = sum(1 for x in sel if x == 0)
    n_sel_two_plus = sum(1 for x in sel if x >= 2)
    # Median can be 1 when explore_top_k=2 but relative floor was too tight; require
    # most repeats with top_k candidates and few empty prunes.
    g1 = med_sel >= 2.0 or (
        n_sel_two_plus >= max(1, int(0.5 * len(sel))) and n_sel_zero <= max(1, int(0.1 * len(sel)))
    )

    print()
    print(f"Gate 0 (generation): {'PASS' if g0 else 'FAIL'}")
    print(f"Gate 1 (prune≥2):    {'PASS' if g1 else 'FAIL'}  (n_selected=0: {n_sel_zero}, ≥2: {n_sel_two_plus})")

    if g0 and g1:
        print("\nOK — proceed to warm-up calibration run.")
        return 0
    print("\nNOT READY — fix attacker XML parsing / generation before warm-up.")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="PRO smoke Gate 0 checker")
    parser.add_argument("run_dir", nargs="?", help="logs/logs_per_run/<id>")
    parser.add_argument("--latest", action="store_true", help="Use newest run under logs/logs_per_run")
    args = parser.parse_args()

    if args.latest:
        root = Path(__file__).resolve().parents[1] / "logs"
        run_dir = _latest_run_dir(root)
        if run_dir is None:
            print("No run directory found under logs/logs_per_run/", file=sys.stderr)
            sys.exit(2)
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        parser.error("Provide run_dir or --latest")

    sys.exit(evaluate(run_dir))


if __name__ == "__main__":
    main()
