#!/usr/bin/env python3
"""Verify frozen baseline (c7becfb) and PRO (9a14912) release parity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "c7becfb25dca954c030eee7d98a4b76b7f4078e9"
PRO_COMMIT = "9a14912d3cdaf23da663f151662bbb94fda31e3c"

BASELINE_PAIRS = [
    ("framework_baseline/attacker.py", "framework/attacker.py"),
    ("framework_baseline/scorer.py", "framework/scorer.py"),
    ("framework_baseline/summarizer.py", "framework/summarizer.py"),
    ("framework_baseline/target.py", "framework/target.py"),
    ("llm/huggingface_models_baseline.py", "llm/huggingface_models.py"),
]

PRO_STRICT_FILES = [
    "framework/summarizer.py",
]

# Allowed to differ from 9a14912 only by feedback-refine ablation flag wiring.
PRO_ABLATION_FILES = [
    "pipeline_pro.py",
    "framework/pro_threshold_telemetry.py",
    "eval_pro.py",
]

_ABLATION_LINE_MARKERS = (
    "pro_enable_feedback_refine",
    "pro_disable_feedback_refine",
    "_goat_improved_variable",
    "Feedback & Refine disabled",
)


def git_show(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO_ROOT)


def files_equal(local_path: Path, commit: str, remote_path: str) -> bool:
    if not local_path.is_file():
        return False
    return local_path.read_bytes() == git_show(commit, remote_path)


def _strip_ablation_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if not any(marker in line for marker in _ABLATION_LINE_MARKERS)
    ]


def _canonicalize_pipeline_pro(text: str) -> str:
    """Fold feedback-refine ablation so default-on matches 9a14912."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if "def __init__" in line:
            out.append(line.replace(", pro_enable_feedback_refine: bool = True", ""))
            i += 1
            continue

        if "self.pro_enable_feedback_refine = bool" in line:
            i += 1
            while i < len(lines) and (
                "if not self.pro_enable_feedback_refine" in lines[i]
                or "Feedback & Refine disabled" in lines[i]
                or "improved_variable hints cleared" in lines[i]
                or "prior_attempt still active" in lines[i]
                or lines[i].strip().startswith("self.logger.info(")
                or lines[i].strip() in {")", '")', '")'}
            ):
                i += 1
            continue

        if line.strip() == "if not self.pro_enable_feedback_refine:":
            i += 1
            while i < len(lines) and lines[i].startswith(" " * 12):
                i += 1
            continue

        if line.strip().startswith("def _goat_improved_variable"):
            i += 1
            while i < len(lines) and not (
                lines[i].startswith("    def ") and "_goat_improved_variable" not in lines[i]
            ):
                i += 1
            continue

        if line.strip() == "if self.pro_enable_feedback_refine:":
            i += 1
            while i < len(lines):
                inner = lines[i]
                if inner.startswith(" " * 8) and not inner.startswith(" " * 12):
                    break
                if inner.strip() and not inner.startswith(" " * 12):
                    break
                out.append(inner[4:] if inner.startswith("    ") else inner)
                i += 1
            continue

        if "if self.per_request_epochs and self.pro_enable_feedback_refine:" in line:
            out.append(line.replace(
                "if self.per_request_epochs and self.pro_enable_feedback_refine:",
                "if self.per_request_epochs:",
            ))
            i += 1
            continue

        if line.strip() == "elif not self.pro_enable_feedback_refine:":
            i += 1
            while i < len(lines) and lines[i].startswith(" " * 20):
                i += 1
            continue

        if line.strip().startswith("_global_cross_epoch_hint = ("):
            out.append('        _global_cross_epoch_hint = (getattr(self, "epoch_refine_hint", None) or "").strip()')
            i += 1
            while i < len(lines) and lines[i].strip() != ")":
                i += 1
            i += 1  # skip closing ")"
            continue

        if line.strip() == "self.pro_enable_feedback_refine":
            i += 1
            if i < len(lines) and "and isinstance(result, dict)" in lines[i]:
                out.append(lines[i].replace("and isinstance", "isinstance", 1))
                i += 1
            continue

        if "improved_variable = self._goat_improved_variable()" in line:
            out.append('        improved_variable = (getattr(self, "epoch_refine_hint", None) or "").strip()')
            i += 1
            continue

        if any(marker in line for marker in _ABLATION_LINE_MARKERS):
            i += 1
            continue

        out.append(line)
        i += 1

    return "\n".join(out) + "\n"


def files_equal_modulo_ablation(local_path: Path, commit: str, remote_path: str) -> bool:
    if not local_path.is_file():
        return False
    local_text = local_path.read_text(encoding="utf-8")
    remote_text = git_show(commit, remote_path).decode("utf-8")
    if remote_path == "pipeline_pro.py":
        local_text = _canonicalize_pipeline_pro(local_text)
    else:
        local_lines = _strip_ablation_lines(local_text)
        remote_lines = _strip_ablation_lines(remote_text)
        return local_lines == remote_lines
    return local_text == remote_text


def pipeline_baseline_equal() -> bool:
    local_lines = (REPO_ROOT / "pipeline_baseline.py").read_text(encoding="utf-8").splitlines()
    remote_lines = git_show(BASELINE_COMMIT, "pipeline.py").decode("utf-8").splitlines()
    normalized = []
    for line in local_lines:
        if line.startswith("# Frozen baseline pipeline"):
            continue
        normalized.append(line.replace("AutoDANTurboBaseline", "AutoDANTurbo"))
    return normalized == remote_lines


def main() -> int:
    failed = []

    print(f"Checking baseline freeze vs {BASELINE_COMMIT[:7]} ...")
    for local_rel, remote_rel in BASELINE_PAIRS:
        ok = files_equal(REPO_ROOT / local_rel, BASELINE_COMMIT, remote_rel)
        status = "OK" if ok else "MISMATCH"
        print(f"  [{status}] {local_rel}")
        if not ok:
            failed.append(local_rel)

    ok = pipeline_baseline_equal()
    status = "OK" if ok else "MISMATCH"
    print(f"  [{status}] pipeline_baseline.py")
    if not ok:
        failed.append("pipeline_baseline.py")

    print(f"Checking PRO freeze vs {PRO_COMMIT[:7]} ...")
    for rel in PRO_STRICT_FILES:
        ok = files_equal(REPO_ROOT / rel, PRO_COMMIT, rel)
        status = "OK" if ok else "MISMATCH"
        print(f"  [{status}] {rel}")
        if not ok:
            failed.append(rel)

    for rel in PRO_ABLATION_FILES:
        ok = files_equal_modulo_ablation(REPO_ROOT / rel, PRO_COMMIT, rel)
        status = "OK (ablation-only diff)" if ok else "MISMATCH"
        print(f"  [{status}] {rel}")
        if not ok:
            failed.append(rel)

    if failed:
        print("\nParity check FAILED:")
        for path in failed:
            print(f"  - {path}")
        return 1

    print("\nParity check PASSED.")
    print("Baseline command uses: framework_baseline + pipeline_baseline + HuggingFaceModelBaseline")
    print("PRO command uses: framework + pipeline_pro + HuggingFaceModel (9a14912 + optional --pro_disable_feedback_refine ablation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
