#!/usr/bin/env python3
"""
Adversarial NeuroSearch demo (does not call the full PRO pipeline):
  PatternManager selects strategy → Attacker generates jailbreak prompt → Target responds.

Does not call Scorer / Feedback / Refiner / Summarizer / AutoDANTurboPro.attack_*.

Examples:
  python demo_chat.py
  python demo_chat.py --cli --request_id 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from framework import Attacker, PatternManager, Retrieval, Target
from llm import HuggingFaceModel, LocalEmbeddingModel
from llm.target_resolve import resolve_hf_token, resolve_target_model

DEFAULT_RUN_DIR = (
    "./logs/logs_per_run/2026-06-10_22-41-03_719516_pro_Qwen_Qwen2.5-1.5B-Instruct"
)
DEFAULT_DATA = "./data/harmbench_eval.json"

# Demo model picker: (label, HF repo id). Prefer small / already-cached checkpoints.
MODEL_PRESETS: List[Tuple[str, str]] = [
    ("Qwen2.5-1.5B-Instruct (default)", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B"),
    ("SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("Gemma-2-2B-IT", "google/gemma-2-2b-it"),
    ("Gemma-3-1B-IT", "google/gemma-3-1b-it"),
    ("Phi-1.5", "microsoft/phi-1_5"),
]
SAME_AS_TARGET = "__same__"
CUSTOM_PRESET = "__custom__"
DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class DemoKit:
    """Demo-only kit: attacker + target + pattern library."""

    attacker: Attacker
    target: Target
    pattern_manager: PatternManager
    retrieval: Optional[Retrieval]
    target_model_key: str
    agent_repo: str
    target_repo: str
    pattern_path: str
    max_new_tokens: int = 512
    strategy_k: int = 4
    use_dynamic_select: bool = True

    def list_strategies(self) -> List[Tuple[str, str]]:
        """[(strategy_id, display_name), ...]"""
        out = []
        for sid, info in self.pattern_manager.strategies.items():
            name = str(info.get("name") or sid)
            out.append((sid, name))
        out.sort(key=lambda x: x[1].lower())
        return out

    def _embed(self, text: str):
        if self.retrieval is None:
            return None
        return self.retrieval.embed(text)

    def select_strategies(
        self,
        request: str,
        strategy_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Step 1: select strategies from the pattern library (or a fixed strategy_id)."""
        if strategy_id and strategy_id != "auto":
            info = self.pattern_manager.strategies.get(strategy_id)
            if not info:
                raise ValueError(f"Unknown strategy_id={strategy_id}")
            return [
                {
                    "strategy_id": strategy_id,
                    "Strategy": info.get("name", strategy_id),
                    "Definition": info.get("description", ""),
                    "Example": info.get("examples", []),
                }
            ]

        if self.use_dynamic_select and self.retrieval is not None:
            try:
                exploit_n, explore_n = 3, 1
                return self.pattern_manager.select_top_k_dynamic(
                    request,
                    self._embed,
                    self.target_model_key,
                    library_round=1,
                    k=exploit_n + explore_n,
                    exploit_n=exploit_n,
                    explore_n=explore_n,
                )
            except Exception:
                pass
        return self.pattern_manager.select_top_k(
            self.target_model_key, turn=1, k=self.strategy_k
        )

    @staticmethod
    def _primary_strategy(strategies: List[Dict[str, Any]]) -> Dict[str, str]:
        if not strategies:
            return {"strategy_id": "", "strategy_name": "", "strategy_description": ""}
        s0 = strategies[0]
        return {
            "strategy_id": str(s0.get("strategy_id") or ""),
            "strategy_name": str(s0.get("Strategy") or s0.get("strategy_id") or ""),
            "strategy_description": str(s0.get("Definition") or ""),
        }

    def generate_prompt(
        self,
        request: str,
        strategies: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Step 2: attacker generates a jailbreak prompt conditioned on selected strategies."""
        if not strategies:
            raise ValueError("No strategy selected — cannot generate attacker prompt.")
        # use_strategy: strategies must be selected before generation
        prompt, system = self.attacker.use_strategy(request, strategies)
        return str(prompt or "").strip(), system

    def run(
        self,
        request: str,
        strategy_id: Optional[str] = None,
        strategies: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Demo path (fixed order):
          1) select strategy (pattern library / user)
          2) attacker.use_strategy(strategies) → jailbreak prompt
          3) target.respond(prompt)
        """
        t0 = time.perf_counter()

        # --- 1. Select strategy first ---
        if strategies is None:
            strategies = self.select_strategies(request, strategy_id=strategy_id)
        if not strategies:
            raise ValueError("Pattern library returned no strategies.")
        primary = self._primary_strategy(strategies)
        t_select = time.perf_counter()

        # --- 2. Generate prompt after strategies are fixed ---
        prompt, attacker_system = self.generate_prompt(request, strategies)
        t_attack = time.perf_counter()

        # --- 3. Query target ---
        response = self.target.respond(prompt, max_new_tokens=self.max_new_tokens)
        t_end = time.perf_counter()

        return {
            "mode": "full",
            "strategies_selected": [
                {
                    "strategy_id": s.get("strategy_id"),
                    "name": s.get("Strategy"),
                    "description": s.get("Definition"),
                }
                for s in strategies
            ],
            "strategy_id": primary["strategy_id"],
            "strategy_name": primary["strategy_name"],
            "strategy_description": primary["strategy_description"],
            "final_prompt": prompt,
            "final_response": response,
            "attacker_system_preview": (attacker_system or "")[:500],
            "elapsed_s": t_end - t0,
            "timing": {
                "select_s": t_select - t0,
                "attacker_s": t_attack - t_select,
                "target_s": t_end - t_attack,
            },
        }

    def run_prompt_only(self, prompt: str) -> Dict[str, Any]:
        """Skip strategy + attacker; send a user-provided prompt to the target."""
        text = (prompt or "").strip()
        if not text:
            raise ValueError("Attacker prompt is empty.")
        t0 = time.perf_counter()
        response = self.target.respond(text, max_new_tokens=self.max_new_tokens)
        t_end = time.perf_counter()
        return {
            "mode": "prompt_only",
            "strategies_selected": [],
            "strategy_id": "",
            "strategy_name": "(skipped — direct prompt)",
            "strategy_description": "Strategy and attacker generation were skipped.",
            "final_prompt": text,
            "final_response": response,
            "attacker_system_preview": "",
            "elapsed_s": t_end - t0,
            "timing": {
                "select_s": 0.0,
                "attacker_s": 0.0,
                "target_s": t_end - t0,
            },
        }


def parse_args():
    p = argparse.ArgumentParser(
        description="Adversarial NeuroSearch demo: strategy → attacker prompt → target"
    )
    p.add_argument("--data", type=str, default=DEFAULT_DATA)
    p.add_argument("--id_start", type=int, default=None,
                   help="CLI: start request id (default 0). Web UI always shows the full dataset.")
    p.add_argument("--id_end", type=int, default=None,
                   help="CLI: end request id inclusive (default = last). Web UI always shows the full dataset.")
    p.add_argument("--request_id", type=int, default=None)
    p.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR)
    p.add_argument("--model", type=str, default="llama3")
    p.add_argument("--target_repo", type=str, default=None)
    p.add_argument("--target_config", type=str, default=None)
    p.add_argument("--agent_repo", type=str, default=None)
    p.add_argument("--agent_config", type=str, default=None)
    p.add_argument("--chat_config", type=str, default="./llm/chat_templates")
    p.add_argument("--hf_token", type=str, default="your_hf_token")
    p.add_argument("--use_quantization", action="store_true", default=True)
    p.add_argument("--no_quantization", action="store_true")
    p.add_argument("--quantization_type", type=str, default="4bit", choices=["4bit", "8bit"])
    p.add_argument("--local_embedding_model", type=str,
                   default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--no_embedding", action="store_true",
                   help="Skip embedding model; use static strategy ranking")
    p.add_argument("--strategy_k", type=int, default=4,
                   help="Total strategies selected (default 4 = exploit 3 + explore 1)")
    p.add_argument("--target_max_new_tokens", type=int, default=512,
                   help="Max new tokens for target response (raise if replies cut off mid-sentence)")
    p.add_argument("--cli", action="store_true")
    p.add_argument("--server_name", type=str, default="127.0.0.1")
    p.add_argument("--server_port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def load_requests(path: str) -> List[str]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        for key in ("lifelong", "warm_up", "data", "requests"):
            if key in data and isinstance(data[key], list):
                return [str(x) for x in data[key]]
    raise ValueError(f"Unsupported dataset format: {path}")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("DemoLiteLogger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def short_model_name(repo: str) -> str:
    """Human-readable model name from a HF repo id."""
    return (repo or "").split("/")[-1] or (repo or "(unknown)")


def _esc(text: str) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def unload_hf_model(hf_model) -> None:
    """Free a HuggingFaceModel from GPU/CPU memory."""
    if hf_model is None:
        return
    try:
        import gc
        import torch

        if hasattr(hf_model, "model"):
            del hf_model.model
        if hasattr(hf_model, "tokenizer"):
            del hf_model.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def unload_kit_llms(kit: DemoKit) -> None:
    agent = getattr(kit.attacker, "model", None)
    target = getattr(kit.target, "model", None)
    shared = agent is target
    unload_hf_model(agent)
    if not shared:
        unload_hf_model(target)


def model_cache_ready(config_dir: str, repo: str) -> bool:
    path = os.path.join(
        config_dir, "model_ckpt", repo.replace("/", "_"), "config.json"
    )
    return os.path.isfile(path)


def resolve_repo_choice(
    preset: str,
    custom_repo: str,
    *,
    model_fallback: str,
    config_dir: str,
) -> Tuple[str, str]:
    """Resolve UI preset/custom text to (repo, generation_config)."""
    custom = (custom_repo or "").strip()
    # Custom text only applies when Custom is selected (ignore leftover textboxes).
    if preset == CUSTOM_PRESET:
        if not custom:
            raise ValueError("Custom HF repo is empty.")
        repo = custom
    elif preset == SAME_AS_TARGET:
        raise ValueError("Cannot resolve Same-as-Target as a standalone model.")
    elif preset:
        repo = preset.strip()
    else:
        # CLI-style fallback via --model preset key
        return resolve_target_model(
            model_preset=model_fallback,
            target_repo=None,
            target_config=None,
            config_dir=config_dir,
        )

    return resolve_target_model(
        model_preset=model_fallback,
        target_repo=repo,
        target_config=None,
        config_dir=config_dir,
    )


def load_hf_model(args, repo: str, config_name: str, logger: logging.Logger) -> HuggingFaceModel:
    use_quant = bool(args.use_quantization) and not args.no_quantization
    hf_token = resolve_hf_token(args.hf_token)
    cached = model_cache_ready(args.chat_config, repo)
    logger.info(
        "Loading HuggingFace model: %s (%s)%s",
        repo,
        config_name,
        "" if cached else " — will download (may take a while)",
    )
    return HuggingFaceModel(
        repo,
        args.chat_config,
        config_name,
        hf_token,
        use_quantization=use_quant,
        quantization_type=args.quantization_type,
    )


def apply_models_to_kit(
    kit: DemoKit,
    args,
    logger: logging.Logger,
    target_preset: str,
    target_custom: str,
    attacker_preset: str,
    attacker_custom: str,
) -> DemoKit:
    """Unload current LLMs and load new Attacker / Target into the same kit."""
    target_repo, target_cfg = resolve_repo_choice(
        target_preset,
        target_custom,
        model_fallback=args.model,
        config_dir=args.chat_config,
    )

    if attacker_preset == SAME_AS_TARGET:
        agent_repo, agent_cfg = target_repo, target_cfg
        share = True
    else:
        agent_repo, agent_cfg = resolve_repo_choice(
            attacker_preset,
            attacker_custom,
            model_fallback=args.model,
            config_dir=args.chat_config,
        )
        share = agent_repo == target_repo and agent_cfg == target_cfg

    # Skip reload if already loaded with the same repos / sharing
    already_shared = kit.attacker.model is kit.target.model
    if (
        kit.target_repo == target_repo
        and kit.agent_repo == agent_repo
        and share == already_shared
        and getattr(kit.target, "model", None) is not None
    ):
        logger.info("Models already loaded — skip reload")
        return kit

    prev_target_repo, prev_target_cfg = kit.target_repo, None
    prev_agent_repo = kit.agent_repo
    # Keep previous config names via infer for restore
    try:
        from llm.target_resolve import infer_config_name

        prev_target_cfg = infer_config_name(prev_target_repo, args.chat_config)
        prev_agent_cfg = infer_config_name(prev_agent_repo, args.chat_config)
    except Exception:
        prev_target_cfg = target_cfg
        prev_agent_cfg = agent_cfg

    unload_kit_llms(kit)

    try:
        target_model = load_hf_model(args, target_repo, target_cfg, logger)
        if share:
            agent_model = target_model
            logger.info("Attacker shares Target weights: %s", target_repo)
        else:
            agent_model = load_hf_model(args, agent_repo, agent_cfg, logger)
    except Exception as exc:
        logger.error("Failed to load new models (%s). Restoring previous…", exc)
        try:
            if not prev_target_cfg:
                raise RuntimeError("No previous config to restore") from exc
            target_model = load_hf_model(
                args, prev_target_repo, prev_target_cfg, logger
            )
            if prev_agent_repo == prev_target_repo:
                agent_model = target_model
            else:
                agent_model = load_hf_model(
                    args,
                    prev_agent_repo,
                    prev_agent_cfg or prev_target_cfg,
                    logger,
                )
            kit.attacker = Attacker(agent_model)
            kit.target = Target(target_model)
            kit.agent_repo = prev_agent_repo
            kit.target_repo = prev_target_repo
            kit.target_model_key = prev_target_repo
        except Exception as restore_exc:
            raise RuntimeError(
                f"Failed to load {target_repo}: {exc}. "
                f"Also failed to restore previous models: {restore_exc}"
            ) from exc
        raise RuntimeError(
            f"Failed to load {target_repo}: {exc}. "
            f"Restored previous models ({prev_agent_repo} / {prev_target_repo})."
        ) from exc

    kit.attacker = Attacker(agent_model)
    kit.target = Target(target_model)
    kit.agent_repo = agent_repo
    kit.target_repo = target_repo
    kit.target_model_key = target_repo
    return kit

def build_kit(args, logger: logging.Logger) -> DemoKit:
    config_dir = args.chat_config

    target_repo, target_cfg = resolve_target_model(
        model_preset=args.model,
        target_repo=args.target_repo,
        target_config=args.target_config,
        config_dir=config_dir,
    )
    if args.agent_repo:
        agent_repo, agent_cfg = resolve_target_model(
            model_preset=args.model,
            target_repo=args.agent_repo,
            target_config=args.agent_config,
            config_dir=config_dir,
        )
    else:
        agent_repo, agent_cfg = target_repo, target_cfg

    target_model = load_hf_model(args, target_repo, target_cfg, logger)
    if agent_repo == target_repo and agent_cfg == target_cfg:
        agent_model = target_model
        logger.info("Attacker: same as target")
    else:
        agent_model = load_hf_model(args, agent_repo, agent_cfg, logger)

    pattern_path = os.path.join(args.run_dir, "pattern_library.json")
    if not os.path.isfile(pattern_path):
        raise FileNotFoundError(f"Pattern library not found: {pattern_path}")

    pattern_manager = PatternManager(
        filepath=pattern_path,
        force_seed=False,
        frozen=True,
    )
    logger.info(
        "Pattern library: %s (%d strategies) — Adversarial NeuroSearch demo lite",
        pattern_path,
        len(pattern_manager.strategies),
    )

    retrieval = None
    if not args.no_embedding:
        embed_model = LocalEmbeddingModel(
            model_name=args.local_embedding_model,
            device=None,
            logger=logger,
        )
        retrieval = Retrieval(embed_model, logger)

    return DemoKit(
        attacker=Attacker(agent_model),
        target=Target(target_model),
        pattern_manager=pattern_manager,
        retrieval=retrieval,
        target_model_key=target_repo,
        agent_repo=agent_repo,
        target_repo=target_repo,
        pattern_path=pattern_path,
        max_new_tokens=args.target_max_new_tokens,
        strategy_k=args.strategy_k,
        use_dynamic_select=retrieval is not None,
    )


def to_markdown_block(title: str, body: str, empty_hint: str) -> str:
    """Format a titled markdown section for Gradio Markdown display."""
    text = (body or "").strip()
    if not text:
        return f"### {title}\n\n{empty_hint}"
    # Keep model output as markdown; normalize newlines for readability.
    return f"### {title}\n\n{text}"


def format_result_panels(
    request_id: Optional[int],
    request: str,
    result: Dict[str, Any],
    mode_label: str = "",
) -> Tuple[str, str, str]:
    """Return (strategy_html, prompt_md, response_md)."""
    strategy_name = result.get("strategy_name") or "(unknown)"
    strategy_id = result.get("strategy_id") or ""
    strategy_desc = result.get("strategy_description") or ""
    prompt = result.get("final_prompt") or ""
    response = result.get("final_response") or ""
    selected = result.get("strategies_selected") or []
    mode = result.get("mode") or "full"
    prompt_only = mode == "prompt_only"

    chips = []
    for s in selected:
        name = s.get("name") or s.get("strategy_id") or "?"
        sid = s.get("strategy_id") or ""
        chips.append(
            f'<span class="chip" title="{_esc(sid)}">{_esc(name)}</span>'
        )
    chips_html = " ".join(chips) if chips else '<span class="muted">—</span>'

    if prompt_only:
        strategy_html = """
        <div class="panel panel-strategy">
          <div class="panel-label">Strategy</div>
          <div class="panel-title">Skipped</div>
          <p class="panel-desc">Direct attacker prompt — no strategy selection or attacker generation.</p>
        </div>
        """
    else:
        strategy_html = f"""
        <div class="panel panel-strategy">
          <div class="panel-label">Strategy</div>
          <div class="panel-title">{_esc(strategy_name)}</div>
          <p class="panel-desc">{_esc(strategy_desc) or "No description."}</p>
          <div class="chip-row"><span class="chip-label">Fed to attacker:</span> {chips_html}</div>
        </div>
        """

    prompt_md = to_markdown_block(
        "Attacker prompt", prompt, "No prompt generated."
    )
    response_md = to_markdown_block(
        "Target response", response, "No response generated."
    )
    return strategy_html, prompt_md, response_md


def format_result(request_id: int, request: str, result: Dict[str, Any]) -> str:
    """Plain-text result for CLI."""
    selected = result.get("strategies_selected") or []
    selected_txt = ", ".join(
        str(s.get("name") or s.get("strategy_id")) for s in selected
    ) or "(none)"
    timing = result.get("timing") or {}
    return (
        f"Request #{request_id}\n{request}\n\n"
        f"1. Strategy: {result.get('strategy_name')} ({result.get('strategy_id')})\n"
        f"{result.get('strategy_description') or ''}\n"
        f"All selected: {selected_txt}\n\n"
        f"2. Attacker prompt:\n{result.get('final_prompt') or ''}\n\n"
        f"3. Target response:\n{result.get('final_response') or ''}\n\n"
        f"timing select={timing.get('select_s', 0):.1f}s "
        f"attacker={timing.get('attacker_s', 0):.1f}s "
        f"target={timing.get('target_s', 0):.1f}s "
        f"total={float(result.get('elapsed_s', 0)):.1f}s"
    )


DEMO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

:root {
  --ans-ink: #111827;
  --ans-muted: #374151;
  --ans-line: #93c5fd;
  --ans-surface: #ffffff;
  --ans-surface-solid: #ffffff;
  --ans-accent: #1e3a8a;
  --ans-accent-dark: #1e3a8a;
  --ans-accent-hover: #2563eb;
  --ans-accent-light: #3b82f6;
  --ans-accent-soft: #dbeafe;
  --ans-attacker: #1d4ed8;
  --ans-target: #2563eb;
  --ans-radius: 16px;
  --ans-shadow: 0 4px 14px rgba(30, 58, 138, 0.08);
  --ans-page: #e0f2fe;
  --ans-card: #ffffff;
}

.gradio-container {
  max-width: 1120px !important;
  width: min(1120px, 100%) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  float: none !important;
  font-family: "Montserrat", "Segoe UI", sans-serif !important;
}
/* Center Gradio 6 fillable layout */
gradio-app,
body > gradio-app,
.gradio-container.main,
.gradio-container .main,
.gradio-container .contain,
.fillable {
  max-width: 1120px !important;
  width: min(1120px, 100%) !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
body {
  display: block !important;
}
.gradio-container,
.gradio-container *:not(code):not(pre):not(kbd):not(samp) {
  font-family: "Montserrat", "Segoe UI", sans-serif !important;
}

/* Page shell: light blue on outer layout only */
html {
  color-scheme: light only !important;
}
html, body, .app, .app > .main,
.gradio-container,
.gradio-container > .main,
.gradio-container .main,
.gradio-container .contain,
.gradio-container .tabs,
.gradio-container .tabitem,
.main, .contain {
  background: var(--ans-page) !important;
  background-color: var(--ans-page) !important;
  background-image: none !important;
  color: #111827 !important;
}

/* Override Gradio / framework CSS variables (prevents slate/dark reapply) */
:root,
.gradio-container,
gradio-app {
  --body-background-fill: #e0f2fe !important;
  --block-background-fill: #ffffff !important;
  --block-border-color: #93c5fd !important;
  --block-label-text-color: #374151 !important;
  --block-title-text-color: #111827 !important;
  --body-text-color: #111827 !important;
  --input-background-fill: #ffffff !important;
  --input-border-color: #93c5fd !important;
  --background-fill-primary: #e0f2fe !important;
  --background-fill-secondary: #ffffff !important;
  --button-primary-background-fill: #1e3a8a !important;
  --button-primary-text-color: #ffffff !important;
  --button-secondary-background-fill: #ffffff !important;
  --button-secondary-text-color: #111827 !important;
  --neutral-50: #e0f2fe !important;
  --neutral-100: #dbeafe !important;
  --neutral-200: #bfdbfe !important;
  --neutral-800: #111827 !important;
  --neutral-900: #111827 !important;
  --neutral-950: #111827 !important;
}

/* All inner wrappers/blocks: transparent (shows light-blue page) */
.gradio-container .wrap,
.gradio-container .block,
.gradio-container .form,
.gradio-container .gr-group,
.gradio-container .gr-panel,
.gradio-container .gr-box,
.gradio-container .column,
.gradio-container .row,
.gradio-container .gap,
.gradio-container .fillable,
.gradio-container .fill_width,
.gradio-container .gr-column,
.gradio-container .gr-row,
.gradio-container fieldset,
.gradio-container .panel,
.gradio-container .input-container,
.gradio-container .dropdown-container,
.gradio-container .block > .wrap,
.gradio-container div[class*="svelte"],
.gradio-container [class*="block"],
.gradio-container header,
.gradio-container footer {
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  color: #111827 !important;
}

/* White card surfaces */
.app-header,
.panel,
.result-md,
.meta-bar,
.input-mode-box,
.input-stack,
.request-one-box,
.input-mode-box > .wrap,
.input-stack > .wrap,
.request-one-box > .wrap,
.gradio-container .html-container,
.gradio-container .html-container.padding,
.gradio-container .prose,
.gradio-container .block-html,
.gradio-container .block-markdown,
.gradio-container .block-textbox,
.gradio-container .block-dropdown,
.gradio-container .block-radio,
.gradio-container .block-group,
.gradio-container .block-button {
  background: #ffffff !important;
  background-color: #ffffff !important;
  background-image: none !important;
}

/* Force-remove dark/slate at every level */
.gradio-container [class*="dark"],
.gradio-container [class*="slate"],
.gradio-container [class*="neutral-9"],
.gradio-container [class*="neutral-8"],
.gradio-container [data-theme="dark"],
.gradio-container .dark,
.gradio-container [style*="background-color: rgb(15"],
.gradio-container [style*="background-color: rgb(17"],
.gradio-container [style*="background-color: rgb(30"],
.gradio-container [style*="background-color: rgb(31"],
.gradio-container [style*="background-color: rgb(51"],
.gradio-container [style*="background-color:#1"],
.gradio-container [style*="background-color: #1"],
.gradio-container [style*="background-color:#0"],
.gradio-container [style*="background-color: #0"],
.gradio-container [style*="background-color:#2"],
.gradio-container [style*="background-color: #2"],
.gradio-container [style*="background-color:#3"],
.gradio-container [style*="background-color: #3"] {
  background: #ffffff !important;
  background-color: #ffffff !important;
  background-image: none !important;
  color: #111827 !important;
}
/* Page-level dark overrides fall back to light blue */
html [class*="dark"],
body [class*="dark"],
.gradio-container > .main[class*="dark"],
.gradio-container .main.fillable {
  background: var(--ans-page) !important;
  background-color: var(--ans-page) !important;
}
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container .block-label span,
.gradio-container .gr-form label,
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .markdown,
.gradio-container .markdown p,
.gradio-container .section-label,
.gradio-container .results-head,
.gradio-container .panel-label,
.gradio-container .panel-desc,
.gradio-container .model-hint,
.gradio-container .chip-label,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container .panel-title,
.gradio-container .result-md p,
.gradio-container .result-md li,
.gradio-container .result-md em,
.gradio-container .result-md strong {
  color: #111827 !important;
}
.gradio-container .request-title,
.gradio-container .models-panel .section-label {
  color: #1e3a8a !important;
}
.gradio-container .results-head {
  color: #1e3a8a !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap input,
.gradio-container .wrap textarea,
.gradio-container [data-testid="textbox"] textarea,
.gradio-container [data-testid="dropdown"] input {
  background: #ffffff !important;
  background-color: #ffffff !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
  border-color: var(--ans-line) !important;
  padding-right: 2.75rem !important;
}
.gradio-container [data-testid="dropdown"] .wrap-inner {
  position: relative !important;
}
.gradio-container .gr-radio label,
.gradio-container .gr-checkbox label,
.gradio-container .gr-dropdown label {
  color: #111827 !important;
}
.gradio-container ul.options li.item,
.gradio-container ul.options li.item span,
.gradio-container ul.options li.item:hover,
.gradio-container ul.options li.item.selected {
  background: #ffffff !important;
  color: #111827 !important;
}
.gradio-container .selected,
.gradio-container [aria-selected="true"],
.gradio-container label.selected {
  background: #dbeafe !important;
  color: #111827 !important;
}

/* Dark text everywhere (except primary buttons) */

.app-header {
  margin: 0.4rem 0 1.15rem;
  padding: 1.1rem 1.25rem 1.15rem;
  border-radius: var(--ans-radius);
  background: #ffffff;
  border: 1px solid var(--ans-line);
  box-shadow: var(--ans-shadow);
}
.app-header .brand-mark {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  margin-bottom: 0.35rem;
}
.app-header .brand-dot {
  width: 0.7rem; height: 0.7rem; border-radius: 999px;
  background: linear-gradient(135deg, #3b82f6, #1e3a8a);
  box-shadow: 0 0 0 4px rgba(59,130,246,0.2);
}
.app-header .brand-kicker {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: #1e3a8a;
}
.app-header h1 {
  font-size: 1.85rem; font-weight: 750; margin: 0;
  letter-spacing: -0.035em; color: var(--ans-ink);
  line-height: 1.15;
}
.app-header .tagline {
  margin: 0.45rem 0 0;
  color: var(--ans-muted);
  font-size: 0.95rem;
  line-height: 1.45;
}

.section-label {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--ans-muted);
  margin: 0.15rem 0 0.55rem;
}
.models-panel .section-label,
.gradio-container .models-panel .section-label {
  font-size: 1.15rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.08em !important;
  color: #1e3a8a !important;
  margin: 0.15rem 0 0.85rem !important;
}

.meta-bar {
  display: flex; flex-wrap: wrap; gap: 0.45rem 0.55rem;
  padding: 0.65rem 0.75rem; margin: 0.15rem 0 0.85rem;
  background: #ffffff;
  border: 1px solid var(--ans-line);
  border-radius: 999px;
  font-size: 0.84rem; color: #1e293b;
  box-shadow: 0 4px 14px rgba(2, 132, 199, 0.06);
}
.meta-bar span {
  background: #ffffff;
  border: 1px solid var(--ans-line);
  border-radius: 999px;
  padding: 0.18rem 0.65rem;
  color: #111827;
}
.meta-bar b { color: var(--ans-ink); }

.panel {
  border: 1px solid var(--ans-line); border-radius: var(--ans-radius);
  padding: 1rem 1.1rem; background: #ffffff;
  box-shadow: var(--ans-shadow);
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
  margin-bottom: 0;
}
.panel-strategy { border-left: 4px solid var(--ans-accent); }
.panel-label {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--ans-muted); margin-bottom: 0.4rem;
}
.panel-title { font-size: 1.2rem; font-weight: 750; color: var(--ans-ink); letter-spacing: -0.02em; }
.panel-id {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.78rem; color: var(--ans-muted); margin-top: 0.2rem;
}
.panel-desc {
  margin: 0.7rem 0 0.8rem; color: #374151; line-height: 1.55; font-size: 0.85rem;
}
.panel-desc b, .panel-desc strong {
  color: #1e3a8a;
  font-weight: 700;
}
.gradio-container .panel-desc {
  font-size: 0.85rem !important;
  line-height: 1.55 !important;
  color: #374151 !important;
}
.result-md p, .result-md li {
  font-size: 0.85rem !important;
  line-height: 1.55; color: #374151 !important;
}
.gradio-container .result-md p,
.gradio-container .result-md li {
  font-size: 0.85rem !important;
  color: #374151 !important;
}
.chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
.chip-label { font-size: 0.8rem; color: var(--ans-muted); margin-right: 0.15rem; }
.chip {
  display: inline-block; background: #dbeafe; color: #1e3a8a;
  border: 1px solid #93c5fd; border-radius: 999px;
  padding: 0.18rem 0.65rem; font-size: 0.78rem; font-weight: 650;
}
.muted { color: #374151; font-size: 0.85rem; }

.result-md {
  border: 1px solid var(--ans-line); border-radius: var(--ans-radius);
  padding: 1rem 1.1rem; background: #ffffff;
  margin-bottom: 0; box-shadow: var(--ans-shadow);
  color: #111827 !important;
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
}
.result-md h3 {
  margin: 0 0 0.65rem 0; font-size: 0.78rem; font-weight: 750;
  letter-spacing: 0.08em; text-transform: uppercase; color: #1e3a8a;
}
.result-md em, .result-md i, .result-md strong {
  color: #111827 !important;
  font-style: normal;
}
.result-md pre, .result-md code {
  white-space: pre-wrap; word-break: break-word;
  background: #f0f9ff; border-radius: 8px;
  color: #111827 !important;
}

/* Align Strategy / Attacker prompt / Target response — one card layer, same width + equal gaps */
.gradio-container .block-html:has(.panel),
.gradio-container .block-html:has(.panel) > .wrap,
.gradio-container .html-container:has(.panel),
.gradio-container .html-container.padding:has(.panel),
.gradio-container .block-markdown,
.gradio-container .block-markdown > .wrap,
.gradio-container .block-markdown .html-container,
.gradio-container .block-markdown .html-container.padding,
.gradio-container .block-markdown .prose,
.gradio-container .block:has(> .wrap .result-md),
.gradio-container .block:has(.result-md) > .wrap,
.gradio-container .styler:has(.result-md),
.gradio-container .styler:has(.panel) {
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin-left: 0 !important;
  margin-right: 0 !important;
  margin-top: 0 !important;
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
}
/* Equal vertical gap between the 3 result cards (on Gradio blocks, not nested) */
.gradio-container .block-html:has(.panel),
.gradio-container .block-markdown:has(.result-md),
.gradio-container .block:has(.result-md) {
  margin-bottom: 0.85rem !important;
  gap: 0 !important;
}
.gradio-container .panel,
.gradio-container .result-md {
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 !important;
  box-sizing: border-box !important;
}

.request-one-box, .input-mode-box, .input-stack {
  border: 1px solid var(--ans-line);
  border-radius: var(--ans-radius);
  background: #ffffff !important;
  background-color: #ffffff !important;
  box-shadow: var(--ans-shadow);
  overflow: visible;
}
.input-mode-box > .wrap,
.input-stack > .wrap,
.request-one-box > .wrap {
  background: #ffffff !important;
  background-color: #ffffff !important;
  box-shadow: none !important;
}
.input-stack .request-one-box {
  border: none;
  box-shadow: none;
  border-radius: 0;
  margin-bottom: 0;
  background: #ffffff !important;
}
.request-one-box { margin-bottom: 0; }
.request-one-box .request-title,
.request-top-bar .request-title,
.gradio-container .request-title {
  font-size: 1.15rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: #1e3a8a !important;
  padding: 0 !important;
  margin: 0 !important;
}
.request-one-box .request-pick,
.request-one-box .request-body {
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
}
.request-one-box .request-pick > .wrap,
.request-one-box .request-body > .wrap,
.request-one-box .request-pick .column,
.request-one-box .request-body .column {
  padding-left: 0 !important;
  padding-right: 0 !important;
  margin-left: 0 !important;
  margin-right: 0 !important;
  width: 100% !important;
  max-width: 100% !important;
}
.request-one-box .request-pick {
  padding: 0.55rem 0.85rem 0.45rem;
  border-bottom: none !important;
}
.request-one-box .request-pick .wrap,
.request-one-box .request-pick .container,
.request-one-box .request-pick .secondary-wrap {
  box-shadow: none !important;
  border: none !important;
  background: transparent !important;
  background-color: transparent !important;
  padding: 0 !important;
  min-height: 0 !important;
  width: 100% !important;
  max-width: 100% !important;
}
/* Request dropdown: proper select box + centered chevron */
.request-one-box .request-pick [data-testid="dropdown"] .wrap-inner,
.request-pick .wrap-inner {
  position: relative !important;
  display: flex !important;
  align-items: center !important;
  min-height: 2.75rem !important;
  height: auto !important;
  width: 100% !important;
  max-width: 100% !important;
  border: 1px solid var(--ans-line) !important;
  border-radius: 12px !important;
  background: #ffffff !important;
  background-color: #ffffff !important;
  box-shadow: none !important;
  padding: 0 !important;
  box-sizing: border-box !important;
  overflow: visible !important;
}
.request-one-box .request-pick [data-testid="dropdown"] input,
.request-one-box .request-pick [role="combobox"],
.request-pick input[role="combobox"] {
  padding: 0.65rem 2.75rem 0.65rem 0.95rem !important;
  min-height: 2.75rem !important;
  line-height: 1.4 !important;
  border: none !important;
  background: transparent !important;
  background-color: transparent !important;
  width: 100% !important;
  box-sizing: border-box !important;
}
.request-one-box .request-pick .icon-wrap,
.request-pick .icon-wrap {
  position: absolute !important;
  top: 50% !important;
  right: 0.95rem !important;
  transform: translateY(-50%) !important;
  width: 1.05rem !important;
  height: 1.05rem !important;
  margin: 0 !important;
  pointer-events: none !important;
}
.request-one-box .request-body {
  padding: 0.35rem 0.85rem 0.45rem !important;
}
.request-one-box .request-body > .wrap,
.request-one-box .request-body .block,
.request-one-box .request-body .form {
  padding: 0 !important;
  margin: 0 !important;
  width: 100% !important;
  max-width: 100% !important;
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}
.request-one-box .request-body textarea,
.request-one-box .request-body textarea:focus,
.request-one-box .request-body textarea::placeholder {
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}
.request-one-box .request-body textarea::placeholder {
  color: #64748b !important;
  -webkit-text-fill-color: #64748b !important;
  opacity: 1 !important;
}
.request-one-box .request-body textarea {
  border: 1px solid var(--ans-line) !important;
  box-shadow: none !important;
  background: #ffffff !important;
  background-color: #ffffff !important;
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
  line-height: 1.6 !important;
  min-height: 8.5rem !important;
  max-height: none !important;
  resize: vertical !important;
  border-radius: 12px !important;
  font-size: 0.95rem !important;
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
  margin: 0 !important;
}
.request-run-row {
  padding: 0.35rem 0.85rem 0.85rem;
  border-top: none !important;
}
.request-run-row button.primary,
.request-run-row button.primary span {
  width: 100%;
  color: #ffffff !important;
  background: #1e3a8a !important;
  border: 1px solid #1e3a8a !important;
  box-shadow: 0 6px 16px rgba(30,58,138,0.22) !important;
  font-weight: 700 !important;
}
.request-run-row button.primary:hover,
.request-run-row button.primary:hover span {
  color: #ffffff !important;
  background: #2563eb !important;
  border-color: #2563eb !important;
}
/* Hide Gradio progress / ETA inside the input area */
.input-stack .progress-bar,
.input-stack .eta-bar,
.input-stack .progress-text,
.input-stack [class*="timer"],
.input-stack [class*="progress"] {
  display: none !important;
}

.input-mode-box {
  padding: 0;
  height: auto;
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}
.input-row {
  align-items: stretch !important;
  gap: 0.9rem;
  margin: 0.15rem 0 0.9rem;
}
.request-top-bar {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 0.75rem;
  padding: 0.55rem 0.85rem 0.45rem;
  border-bottom: none !important;
}
.request-top-bar .request-title {
  padding: 0 !important;
  margin: 0 !important;
  flex: 1;
}
.request-mode-toggle {
  flex: 0 0 auto;
  min-width: 260px;
}
/* Mode pills: light blue + dark blue text; selected = dark blue + white */
.request-mode-toggle .wrap,
.request-mode-toggle .form,
.request-mode-toggle .block {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.request-mode-toggle label,
.request-mode-toggle .label-wrap {
  display: none !important;
}
.request-mode-toggle .wrap > label,
.request-mode-toggle fieldset,
.request-mode-toggle .form > div {
  display: flex !important;
  flex-direction: row !important;
  gap: 0.45rem !important;
  background: transparent !important;
  border: none !important;
}
.request-mode-toggle label:not(.label-wrap),
.request-mode-toggle .wrap label,
.request-mode-toggle span:has(input[type="radio"]),
.request-mode-toggle label.svelte-1qxc5u1,
.request-mode-toggle .form label {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  margin: 0 !important;
  padding: 0.4rem 0.85rem !important;
  border-radius: 999px !important;
  background: #e0f2fe !important;
  background-color: #e0f2fe !important;
  color: #1e3a8a !important;
  border: 1px solid #93c5fd !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  cursor: pointer !important;
  line-height: 1.2 !important;
}
.request-mode-toggle input[type="radio"] {
  position: absolute !important;
  opacity: 0 !important;
  width: 0 !important;
  height: 0 !important;
}
.request-mode-toggle label:has(input:checked),
.request-mode-toggle label.selected,
.request-mode-toggle .selected {
  background: #1e3a8a !important;
  background-color: #1e3a8a !important;
  color: #ffffff !important;
  border-color: #1e3a8a !important;
}
.request-mode-toggle label:has(input:checked) span,
.request-mode-toggle label.selected span {
  color: #ffffff !important;
}

/* Gradio outer wrapper: no extra chrome — only .models-panel draws the card */
.gradio-container .styler:has(> .models-panel),
.gradio-container .models-panel.styler,
.gradio-container .styler.models-panel {
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.models-panel {
  padding: 0.95rem 1.05rem 0.95rem;
  margin-bottom: 0.95rem;
  background: #ffffff !important;
  background-color: #ffffff !important;
  background-image: none !important;
  border: 1px solid var(--ans-line) !important;
  border-left: 4px solid #2563eb !important;
  border-radius: var(--ans-radius) !important;
  box-shadow: var(--ans-shadow) !important;
  overflow: visible;
}

/* Active Models: white shell; nested wrappers transparent so white shows */
.models-panel > .wrap,
.models-panel .block,
.models-panel .wrap,
.models-panel .form,
.models-panel .column,
.models-panel .row,
.models-panel .gap,
.models-panel .fillable,
.models-panel .gr-column,
.models-panel .gr-row,
.models-panel .gr-group,
.models-panel .styler,
.models-panel div[class*="svelte"],
.models-panel .html-container,
.models-panel .html-container.padding,
.models-panel .html-container .prose,
.models-panel .block.html,
.models-panel .block.html > .wrap,
.models-panel .block-dropdown,
.models-panel .block-textbox,
.models-panel .block-group,
.models-panel .input-container,
.models-panel .dropdown-container,
.models-panel .single-select,
.models-panel .container,
.models-panel .secondary-wrap,
.models-panel .models-pick-row,
.models-panel .models-pick-row > .wrap,
.models-panel .models-pick-row .column,
.models-panel .models-pick-row .column > .wrap,
.models-panel .models-apply-row,
.models-panel .models-apply-row > .wrap {
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  box-shadow: none !important;
  border: none !important;
  color: #111827 !important;
}
.models-panel .model-pick-card > .wrap {
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  color: #111827 !important;
  border: none !important;
  box-shadow: none !important;
}
.models-panel .html-container,
.models-panel .html-container.padding,
.models-panel .block.html,
.models-panel .block.html > .wrap {
  border: none !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
.models-panel input,
.models-panel textarea,
.models-panel select,
.models-panel .wrap input,
.models-panel .wrap textarea,
.models-panel [role="listbox"],
.models-panel .block-dropdown .wrap-inner,
.model-pick-card [data-testid="dropdown"] .wrap-inner {
  background: #e0f2fe !important;
  background-color: #e0f2fe !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
  border: 1px solid var(--ans-line) !important;
  border-radius: 10px !important;
  box-shadow: none !important;
  min-height: 2.75rem !important;
  box-sizing: border-box !important;
}
.models-panel .block-dropdown > .wrap,
.models-panel .block-dropdown .wrap,
.model-pick-card [data-testid="dropdown"] .wrap {
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  min-height: 0 !important;
}
.models-panel .block-dropdown .wrap-inner,
.model-pick-card [data-testid="dropdown"] .wrap-inner {
  position: relative !important;
  padding: 0 !important;
  overflow: visible !important;
}
.models-panel .block-dropdown .secondary-wrap,
.model-pick-card .secondary-wrap,
.models-panel [data-testid="dropdown"] .secondary-wrap,
.models-panel [data-testid="dropdown"] .container {
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  min-height: 0 !important;
}
.model-pick-card [data-testid="dropdown"] input,
.models-panel [data-testid="dropdown"] input,
.model-pick-card [role="combobox"],
.models-panel [role="combobox"] {
  padding: 0.65rem 2.75rem 0.65rem 0.95rem !important;
  line-height: 1.4 !important;
  border: none !important;
  background: transparent !important;
  background-color: transparent !important;
  min-height: 2.75rem !important;
}
/* Dropdown chevron — blue, inset, stroke-style */
.gradio-container .icon-wrap,
.models-panel .icon-wrap,
.model-pick-card .icon-wrap {
  position: absolute !important;
  top: 50% !important;
  right: 1.15rem !important;
  transform: translateY(-50%) !important;
  width: 1.05rem !important;
  height: 1.05rem !important;
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
  color: #1e3a8a !important;
  pointer-events: none !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  opacity: 0.9 !important;
}
.gradio-container .icon-wrap svg.dropdown-arrow,
.models-panel .icon-wrap svg.dropdown-arrow,
.model-pick-card .icon-wrap svg.dropdown-arrow {
  width: 100% !important;
  height: 100% !important;
  display: block !important;
  /* replace filled triangle with stroke chevron */
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='none'%3E%3Cpath d='M5.5 7.75L10 12.25L14.5 7.75' stroke='%231e3a8a' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") !important;
  background-repeat: no-repeat !important;
  background-position: center !important;
  background-size: contain !important;
}
.gradio-container .icon-wrap svg.dropdown-arrow path,
.models-panel .icon-wrap svg.dropdown-arrow path,
.model-pick-card .icon-wrap svg.dropdown-arrow path {
  display: none !important;
}
.models-panel [class*="dark"],
.models-panel [class*="slate"],
.models-panel [class*="neutral-9"],
.models-panel [class*="neutral-8"] {
  background: #ffffff !important;
  background-color: #ffffff !important;
  color: #111827 !important;
}
.models-panel .section-label {
  margin-bottom: 0.85rem;
}
.models-pick-row {
  gap: 1.25rem !important;
  margin-bottom: 0.35rem;
}
/* No nested cards — labels + dropdown only */
.models-panel .models-pick-row .column,
.models-panel .models-pick-row .column > .wrap,
.models-panel .models-pick-row .styler,
.models-panel .column:has(.model-pick-card),
.models-panel .styler:has(.model-pick-card),
.gradio-container .styler:has(> .model-pick-card),
.gradio-container .column:has(.model-pick-card) {
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
  background-color: transparent !important;
  padding: 0 !important;
}
.gradio-container .model-pick-card,
.gradio-container .gr-group.model-pick-card,
.models-panel .model-pick-card,
.models-panel .gr-group.model-pick-card,
.models-panel .model-pick-card.model-pick-attacker,
.models-panel .model-pick-card.model-pick-target {
  border: 1px solid var(--ans-line) !important;
  border-radius: 12px !important;
  padding: 0.85rem 0.95rem 0.95rem !important;
  background: #ffffff !important;
  background-color: #ffffff !important;
  height: 100%;
  box-shadow: none !important;
  box-sizing: border-box !important;
}
.model-pick-card > .wrap,
.model-pick-card .styler,
.model-pick-card .block,
.model-pick-card .wrap,
.model-pick-card .block-group,
.model-pick-card .form {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  background-color: transparent !important;
  padding: 0 !important;
}
.model-pick-card .model-role {
  font-size: 0.7rem; font-weight: 750; letter-spacing: 0.08em;
  text-transform: uppercase; margin-bottom: 0.15rem;
}
.model-pick-attacker .model-role { color: var(--ans-attacker); }
.model-pick-target .model-role { color: var(--ans-target); }
.model-pick-card .model-hint {
  font-size: 0.8rem; color: var(--ans-muted); margin: 0 0 0.55rem;
}
.model-pick-card label,
.model-pick-card .label-wrap span {
  display: none !important;
}
.models-apply-row {
  margin-top: 0.25rem;
}
.models-panel .models-status-wrap,
.models-panel .models-status-wrap > .wrap,
.models-panel .models-status-wrap .html-container,
.models-panel .models-status-wrap .block {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}
.models-panel button.secondary,
.models-panel button.secondary span,
.models-panel .secondary-wrap button,
.models-panel .secondary-wrap button span,
.models-apply-row button,
.models-apply-row button span,
.models-apply-row button.secondary,
.models-apply-row button.secondary span,
.models-apply-row button.primary,
.models-apply-row button.primary span {
  background: #1e3a8a !important;
  background-color: #1e3a8a !important;
  color: #ffffff !important;
  border: 1px solid #1e3a8a !important;
  border-radius: 12px !important;
  box-shadow: 0 6px 16px rgba(30,58,138,0.2) !important;
  font-weight: 700 !important;
}
.models-panel button.secondary:hover,
.models-panel button.secondary:hover span,
.models-apply-row button:hover,
.models-apply-row button:hover span,
.models-apply-row button.secondary:hover,
.models-apply-row button.secondary:hover span,
.models-apply-row button.primary:hover,
.models-apply-row button.primary:hover span {
  background: #2563eb !important;
  background-color: #2563eb !important;
  border-color: #2563eb !important;
  color: #ffffff !important;
}

.results-head {
  margin: 0.75rem 0 0.85rem;
  padding: 0;
  font-size: 1.25rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #1e3a8a !important;
  text-align: center;
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  border-radius: 0;
  box-shadow: none !important;
}
.gradio-container .results-head {
  font-size: 1.25rem !important;
  font-weight: 800 !important;
  color: #1e3a8a !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.run-status {
  display: none;
  align-items: center;
  gap: 0.7rem;
  margin: 0 0 0.85rem;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  border: 1px solid #93c5fd;
  background: linear-gradient(90deg, #eff6ff, #dbeafe);
  color: #1e3a8a;
  font-weight: 700;
  font-size: 0.95rem;
}
.run-status.is-running {
  display: flex;
}
.model-status {
  display: none;
  align-items: center;
  gap: 0.65rem;
  margin: 0.35rem 0 0.65rem;
  padding: 0.7rem 0.95rem;
  border-radius: 12px;
  border: 1px solid #93c5fd;
  background: linear-gradient(90deg, #eff6ff, #dbeafe);
  color: #1e3a8a;
  font-weight: 650;
  font-size: 0.9rem;
  line-height: 1.35;
}
.model-status.is-visible {
  display: flex;
}
.model-status.is-loading {
  border-color: #93c5fd;
  background: linear-gradient(90deg, #eff6ff, #dbeafe);
  color: #1e3a8a;
}
.model-status.is-ok {
  border-color: #86efac;
  background: linear-gradient(90deg, #f0fdf4, #dcfce7);
  color: #166534;
}
.model-status.is-error {
  border-color: #fca5a5;
  background: linear-gradient(90deg, #fef2f2, #fee2e2);
  color: #991b1b;
}
.models-panel.is-loading-models {
  opacity: 0.92;
}
.run-status .spinner,
.panel-running .spinner,
.model-status .spinner {
  width: 1.05rem;
  height: 1.05rem;
  border: 2.5px solid #93c5fd;
  border-top-color: #1e3a8a;
  border-radius: 50%;
  animation: ans-spin 0.8s linear infinite;
  flex-shrink: 0;
  display: inline-block;
}
.model-status.is-ok .spinner,
.model-status.is-error .spinner {
  display: none;
}
@keyframes ans-spin {
  to { transform: rotate(360deg); }
}
.panel-running {
  border-left: 4px solid #2563eb;
}
.panel-running .panel-title {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
}

/* Primary buttons: dark blue + white text */
button.primary, .primary,
button.primary span, .primary span {
  border-radius: 12px !important;
  background: #1e3a8a !important;
  color: #ffffff !important;
  border: 1px solid #1e3a8a !important;
  box-shadow: 0 6px 16px rgba(30,58,138,0.2) !important;
  font-weight: 700 !important;
}
button.primary:hover, .primary:hover,
button.primary:hover span, .primary:hover span {
  background: #2563eb !important;
  border-color: #2563eb !important;
  color: #ffffff !important;
}
.models-apply-row button.secondary,
.secondary-wrap button, button.secondary,
.models-apply-row button.secondary span,
.secondary-wrap button span, button.secondary span {
  border-radius: 12px !important;
  border-color: #1e3a8a !important;
  background: #1e3a8a !important;
  background-color: #1e3a8a !important;
  color: #ffffff !important;
  font-weight: 700 !important;
}
.models-apply-row button.secondary:hover,
.models-apply-row button.secondary:hover span {
  background: #2563eb !important;
  background-color: #2563eb !important;
  border-color: #2563eb !important;
  color: #ffffff !important;
}
/* Keep primary buttons dark blue (not treated as dark containers) */
button.primary, .primary,
button.primary span, .primary span,
.request-run-row button.primary,
.request-run-row button.primary span {
  background: #1e3a8a !important;
  background-color: #1e3a8a !important;
  color: #ffffff !important;
  border-color: #1e3a8a !important;
}
button.primary:hover, .primary:hover,
button.primary:hover span, .primary:hover span,
.request-run-row button.primary:hover,
.request-run-row button.primary:hover span {
  background: #2563eb !important;
  background-color: #2563eb !important;
  border-color: #2563eb !important;
  color: #ffffff !important;
}
footer { display: none !important; }

@media (max-width: 720px) {
  .app-header h1 { font-size: 1.5rem; }
  .meta-bar { border-radius: 14px; }
}
"""


def resolve_id_range(n_requests: int, args) -> Tuple[int, int]:
    """Return inclusive (id_start, id_end), clamped to dataset size."""
    start = 0 if args.id_start is None else int(args.id_start)
    end = (n_requests - 1) if args.id_end is None else int(args.id_end)
    start = max(0, min(start, n_requests - 1))
    end = max(0, min(end, n_requests - 1))
    if end < start:
        start, end = end, start
    return start, end


def run_cli(kit: DemoKit, requests: List[str], args):
    print("\n" + "=" * 60)
    print("Adversarial NeuroSearch demo — strategy → attacker prompt → target")
    print(f"Attacker: {kit.agent_repo}")
    print(f"Target:   {kit.target_repo}")
    print(f"Pattern:  {kit.pattern_path}")
    print("=" * 60 + "\n")

    if args.request_id is not None:
        ids = [args.request_id]
    else:
        start, end = resolve_id_range(len(requests), args)
        ids = list(range(start, end + 1))
        print(f"CLI range: #{start} .. #{end} ({len(ids)} request(s))\n")

    for rid in ids:
        if rid < 0 or rid >= len(requests):
            print(f"Skip invalid id {rid}")
            continue
        print(f"\n>>> Request #{rid} ...")
        md, result = run_one(kit, rid, requests[rid])
        print(md)
        print(f"strategy={result.get('strategy_name')}")


def run_one(
    kit: DemoKit,
    request_id: int,
    request: str,
    strategy_id: Optional[str] = None,
    strategies: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    result = kit.run(request, strategy_id=strategy_id, strategies=strategies)
    return format_result(request_id, request, result), result


def run_gradio(kit: DemoKit, requests: List[str], args):
    try:
        import gradio as gr
    except ImportError:
        print(
            "Missing package: gradio. Install with:\n"
            "  pip install gradio\n"
            "Or use CLI mode:\n"
            "  python demo_chat.py --cli",
            file=sys.stderr,
        )
        sys.exit(1)

    logger = logging.getLogger("DemoLiteLogger")
    state = {"kit": kit, "skip_pick_clear": False}

    n = len(requests)
    CUSTOM_RID = -1
    dropdown_choices = [
        ("Custom (type your own)", CUSTOM_RID),
    ] + [
        (f"#{i} — {requests[i][:72]}{'…' if len(requests[i]) > 72 else ''}", i)
        for i in range(n)
    ]

    target_preset_choices = MODEL_PRESETS
    attacker_preset_choices = [
        ("Same as Target", SAME_AS_TARGET),
    ] + MODEL_PRESETS

    preset_repos = {repo for _, repo in MODEL_PRESETS}
    if args.target_repo and args.target_repo in preset_repos:
        default_target_preset = args.target_repo
    elif kit.target_repo in preset_repos:
        default_target_preset = kit.target_repo
    else:
        default_target_preset = DEFAULT_MODEL_REPO

    if args.agent_repo and args.agent_repo in preset_repos:
        default_attacker_preset = args.agent_repo
    elif kit.agent_repo in preset_repos:
        # Prefer showing the concrete model (e.g. Qwen), not "Same as Target"
        default_attacker_preset = kit.agent_repo
    else:
        default_attacker_preset = DEFAULT_MODEL_REPO

    MODE_REQUEST = "request"
    MODE_PROMPT = "prompt"

    empty_strategy = (
        '<div class="panel panel-strategy">'
        '<div class="panel-label">Strategy</div>'
        '<p class="panel-desc">Click <b>Run demo</b> to select strategy, '
        "generate an attacker prompt, and query the target.</p>"
        "</div>"
    )
    idle_status = '<div class="run-status" aria-hidden="true"></div>'
    running_status = (
        '<div class="run-status is-running" role="status" aria-live="polite">'
        '<span class="spinner" aria-hidden="true"></span>'
        "<span>Running demo… selecting strategy, generating attacker prompt, "
        "querying target.</span>"
        "</div>"
    )
    running_strategy = (
        '<div class="panel panel-strategy panel-running">'
        '<div class="panel-label">Strategy</div>'
        '<div class="panel-title"><span class="spinner" aria-hidden="true"></span>'
        "Working…</div>"
        '<p class="panel-desc">Please wait — results will appear here when finished.</p>'
        "</div>"
    )
    running_prompt = "### Attacker prompt\n\n_Generating…_"
    running_response = "### Target response\n\n_Waiting for target…_"

    def request_from_id(rid):
        if rid is None:
            return ""
        try:
            rid = int(rid)
        except (TypeError, ValueError):
            return ""
        if rid == CUSTOM_RID or rid < 0 or rid >= n:
            return ""
        return requests[rid]

    def on_mode_change(mode):
        is_prompt = mode == MODE_PROMPT
        title = (
            '<div class="request-title">Attacker prompt</div>'
            if is_prompt
            else '<div class="request-title">Request</div>'
        )
        return (
            title,
            gr.update(visible=not is_prompt),
            gr.update(visible=is_prompt),
        )

    def on_pick(rid):
        """Selecting a dataset id fills request text; Custom clears it (unless switched via edit)."""
        if state.pop("skip_pick_clear", False):
            return gr.update()
        return request_from_id(rid)

    def on_request_edit(text, rid):
        """If user edits text away from the selected dataset request, switch to Custom."""
        if rid is None:
            return gr.update()
        try:
            rid_i = int(rid)
        except (TypeError, ValueError):
            return gr.update()
        if rid_i == CUSTOM_RID or rid_i < 0 or rid_i >= n:
            return gr.update()
        if (text or "") != requests[rid_i]:
            state["skip_pick_clear"] = True
            return gr.update(value=CUSTOM_RID)
        return gr.update()

    def on_apply_models(target_preset, attacker_preset):
        def _preset_label(preset: str) -> str:
            if not preset or preset == SAME_AS_TARGET:
                return "Same as Target"
            return short_model_name(str(preset))

        def _status(kind: str, message: str) -> str:
            return (
                f'<div class="model-status is-visible is-{_esc(kind)}" '
                f'role="status" aria-live="polite">'
                f'<span class="spinner" aria-hidden="true"></span>'
                f"<span>{_esc(message)}</span></div>"
            )

        att_label = _preset_label(attacker_preset)
        tgt_label = _preset_label(target_preset)
        yield (
            _status(
                "loading",
                f"Loading models… Attacker: {att_label} · Target: {tgt_label}",
            ),
            gr.update(interactive=False, value="Loading models…"),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

        try:
            preview_repo = None
            if target_preset and target_preset != SAME_AS_TARGET:
                preview_repo = target_preset
            if preview_repo and not model_cache_ready(args.chat_config, preview_repo):
                logger.warning(
                    "Model %s is not cached locally — download may take a long time",
                    preview_repo,
                )

            before_agent = state["kit"].agent_repo
            before_target = state["kit"].target_repo
            apply_models_to_kit(
                state["kit"],
                args,
                logger,
                target_preset,
                "",
                attacker_preset,
                "",
            )
            k = state["kit"]
            logger.info(
                "Models loaded · Attacker=%s · Target=%s",
                short_model_name(k.agent_repo),
                short_model_name(k.target_repo),
            )
            skipped = (
                before_agent == k.agent_repo and before_target == k.target_repo
            )
            msg = (
                f"Models ready · Attacker: {short_model_name(k.agent_repo)} · "
                f"Target: {short_model_name(k.target_repo)}"
            )
            if skipped:
                msg = (
                    f"Already loaded · Attacker: {short_model_name(k.agent_repo)} · "
                    f"Target: {short_model_name(k.target_repo)}"
                )
            yield (
                _status("ok", msg),
                gr.update(interactive=True, value="Apply Models"),
                gr.update(interactive=True),
                gr.update(interactive=True),
            )
        except SystemExit as exc:
            logger.error("Apply models failed: %s", exc)
            yield (
                _status("error", f"Failed to load models: {exc}"),
                gr.update(interactive=True, value="Apply Models"),
                gr.update(interactive=True),
                gr.update(interactive=True),
            )
        except Exception as exc:
            logger.error("Apply models failed: %s", exc)
            yield (
                _status("error", f"Failed to load models: {exc}"),
                gr.update(interactive=True, value="Apply Models"),
                gr.update(interactive=True),
                gr.update(interactive=True),
            )

    def on_run(mode, rid, request_text, prompt_text):
        # Immediate UI feedback before the long model calls.
        yield (
            running_status,
            running_strategy,
            running_prompt,
            running_response,
            gr.update(interactive=False, value="Running…"),
        )

        k = state["kit"]
        try:
            if mode == MODE_PROMPT:
                result = k.run_prompt_only(prompt_text or "")
                strategy_html, prompt_md, response_md = format_result_panels(
                    None, "", result, mode_label="Direct prompt"
                )
            else:
                request = (request_text or "").strip()
                if not request:
                    raise ValueError("Request is empty.")

                rid_out: Optional[int] = None
                label = "Custom request"
                try:
                    rid_i = int(rid) if rid is not None else CUSTOM_RID
                except (TypeError, ValueError):
                    rid_i = CUSTOM_RID
                if 0 <= rid_i < n and request == requests[rid_i]:
                    rid_out = rid_i
                    label = ""

                # Always Auto from pattern library
                result = k.run(request, strategy_id="auto")
                strategy_html, prompt_md, response_md = format_result_panels(
                    rid_out, request, result, mode_label=label
                )

            yield (
                idle_status,
                strategy_html,
                prompt_md,
                response_md,
                gr.update(interactive=True, value="Run demo"),
            )
        except Exception as exc:
            err = (
                f'<div class="panel"><div class="panel-label">Error</div>'
                f'<pre style="white-space:pre-wrap">{_esc(str(exc))}</pre></div>'
            )
            yield (
                idle_status,
                err,
                "### Attacker prompt\n\nError.",
                "### Target response\n\nError.",
                gr.update(interactive=True, value="Run demo"),
            )

    demo_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky",
        neutral_hue="gray",
        font=gr.themes.GoogleFont("Montserrat"),
        font_mono=gr.themes.GoogleFont("Montserrat"),
    ).set(
        body_background_fill="#e0f2fe",
        body_text_color="#111827",
        block_background_fill="#ffffff",
        block_border_color="#93c5fd",
        block_label_text_color="#374151",
        block_title_text_color="#111827",
        input_background_fill="#ffffff",
        input_border_color="#93c5fd",
        block_radius="*radius_lg",
        button_primary_background_fill="#1e3a8a",
        button_primary_background_fill_hover="#2563eb",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_text_color="#111827",
        border_color_primary="#93c5fd",
        color_accent="#1e3a8a",
        color_accent_soft="#dbeafe",
    )

    with gr.Blocks(
        title="Adversarial NeuroSearch Demo",
        fill_width=False,
    ) as demo:
        gr.HTML(
            """
            <div class="app-header">
              <div class="brand-mark">
                <span class="brand-dot"></span>
                <span class="brand-kicker">Thesis demo</span>
              </div>
              <h1>Adversarial NeuroSearch</h1>
            </div>
            """
        )

        with gr.Group(elem_classes=["models-panel"]):
            gr.HTML('<div class="section-label">Active models</div>')
            with gr.Row(elem_classes=["models-pick-row"]):
                with gr.Column(scale=1, min_width=260):
                    with gr.Group(elem_classes=["model-pick-card", "model-pick-attacker"]):
                        gr.HTML(
                            '<div class="model-role">Attacker Model</div>'
                            '<div class="model-hint">Generates jailbreak prompt</div>'
                        )
                        attacker_preset_dd = gr.Dropdown(
                            choices=attacker_preset_choices,
                            value=default_attacker_preset,
                            label="Attacker Model",
                            show_label=False,
                            container=False,
                        )
                with gr.Column(scale=1, min_width=260):
                    with gr.Group(elem_classes=["model-pick-card", "model-pick-target"]):
                        gr.HTML(
                            '<div class="model-role">Target Model</div>'
                            '<div class="model-hint">Answers jailbreak prompt</div>'
                        )
                        target_preset_dd = gr.Dropdown(
                            choices=target_preset_choices,
                            value=default_target_preset,
                            label="Target Model",
                            show_label=False,
                            container=False,
                        )
            models_status = gr.HTML(
                value='<div class="model-status" aria-hidden="true"></div>',
                elem_classes=["models-status-wrap"],
            )
            with gr.Row(elem_classes=["models-apply-row"]):
                apply_models_btn = gr.Button(
                    "Apply Models", variant="primary", size="lg"
                )

        with gr.Group(elem_classes=["input-stack"]):
            with gr.Row(elem_classes=["request-top-bar"]):
                section_title = gr.HTML(
                    value='<div class="request-title">Request</div>',
                    elem_classes=["request-title-wrap"],
                )
                with gr.Group(elem_classes=["request-mode-toggle"]):
                    input_mode = gr.Radio(
                        choices=[
                            ("Request", MODE_REQUEST),
                            ("Attack prompt", MODE_PROMPT),
                        ],
                        value=MODE_REQUEST,
                        label="Input mode",
                        show_label=False,
                        container=False,
                    )
            with gr.Group(elem_classes=["request-one-box"]) as request_box:
                with gr.Column(elem_classes=["request-pick"]):
                    request_dd = gr.Dropdown(
                        choices=dropdown_choices,
                        value=CUSTOM_RID,
                        show_label=False,
                        label="Request id",
                        filterable=True,
                        container=False,
                    )
                with gr.Column(elem_classes=["request-body"]):
                    request_view = gr.Textbox(
                        value="",
                        show_label=False,
                        label="Request text",
                        placeholder="Type a custom request, or pick one from the dropdown…",
                        lines=8,
                        max_lines=24,
                        interactive=True,
                        container=False,
                    )

            with gr.Group(elem_classes=["request-one-box"], visible=False) as prompt_box:
                with gr.Column(elem_classes=["request-body"]):
                    prompt_input = gr.Textbox(
                        value="",
                        show_label=False,
                        label="Attacker prompt",
                        placeholder="Paste or type the full attacker prompt…",
                        lines=10,
                        max_lines=30,
                        interactive=True,
                        container=False,
                    )

            with gr.Row(elem_classes=["request-run-row"]):
                run_btn = gr.Button("Run demo", variant="primary", size="lg")

        gr.HTML('<div class="results-head">Results</div>')
        run_status = gr.HTML(value=idle_status)
        strategy_html = gr.HTML(value=empty_strategy)
        prompt_md = gr.Markdown(
            value="### Attacker prompt\n\nGenerated prompt will appear here…",
            elem_classes=["result-md"],
        )
        response_md = gr.Markdown(
            value="### Target response\n\nTarget response will appear here…",
            elem_classes=["result-md"],
        )

        apply_models_btn.click(
            on_apply_models,
            inputs=[
                target_preset_dd,
                attacker_preset_dd,
            ],
            outputs=[
                models_status,
                apply_models_btn,
                attacker_preset_dd,
                target_preset_dd,
            ],
            show_progress="full",
        )
        input_mode.change(
            on_mode_change,
            inputs=[input_mode],
            outputs=[section_title, request_box, prompt_box],
        )
        request_dd.change(
            on_pick,
            inputs=[request_dd],
            outputs=[request_view],
        )
        request_view.change(
            on_request_edit,
            inputs=[request_view, request_dd],
            outputs=[request_dd],
        )
        run_btn.click(
            on_run,
            inputs=[input_mode, request_dd, request_view, prompt_input],
            outputs=[run_status, strategy_html, prompt_md, response_md, run_btn],
            show_progress="full",
        )

    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        ssr_mode=False,
        css=DEMO_CSS,
        theme=demo_theme,
    )


def main():
    args = parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    os.environ.setdefault("WANDB_MODE", "disabled")

    requests = load_requests(args.data)
    if not requests:
        raise SystemExit(f"Empty dataset: {args.data}")

    logger = setup_logger()
    logger.info("Loaded %d requests from %s (ids 0..%d)", len(requests), args.data, len(requests) - 1)
    kit = build_kit(args, logger)

    if args.cli:
        run_cli(kit, requests, args)
    else:
        run_gradio(kit, requests, args)


if __name__ == "__main__":
    main()
