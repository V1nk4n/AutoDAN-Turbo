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
    max_new_tokens: int = 128
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
    p.add_argument("--target_max_new_tokens", type=int, default=128)
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


def models_status_html(kit: DemoKit) -> str:
    """Compact Attacker / Target names currently loaded."""
    return f"""
    <div class="models-status">
      <div class="model-card model-card-attacker">
        <div class="model-role">Attacker</div>
        <div class="model-hint">Generates jailbreak prompt</div>
        <div class="model-name">{_esc(short_model_name(kit.agent_repo))}</div>
      </div>
      <div class="model-card model-card-target">
        <div class="model-role">Target</div>
        <div class="model-hint">Answers jailbreak prompt</div>
        <div class="model-name">{_esc(short_model_name(kit.target_repo))}</div>
      </div>
    </div>
    """


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
        return f"### {title}\n\n_{empty_hint}_"
    # Keep model output as markdown; normalize newlines for readability.
    return f"### {title}\n\n{text}"


def format_result_panels(
    request_id: Optional[int],
    request: str,
    result: Dict[str, Any],
    mode_label: str = "",
) -> Tuple[str, str, str, str, str]:
    """Return (strategy_html, prompt_md, response_md, meta_html, status)."""
    strategy_name = result.get("strategy_name") or "(unknown)"
    strategy_id = result.get("strategy_id") or ""
    strategy_desc = result.get("strategy_description") or ""
    prompt = result.get("final_prompt") or ""
    response = result.get("final_response") or ""
    selected = result.get("strategies_selected") or []
    timing = result.get("timing") or {}
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
          <div class="panel-id">{_esc(strategy_id)}</div>
          <p class="panel-desc">{_esc(strategy_desc) or "No description."}</p>
          <div class="chip-row"><span class="chip-label">Fed to attacker:</span> {chips_html}</div>
        </div>
        """

    if request_id is not None:
        req_label = f"Request <b>#{request_id}</b>"
        status_id = f"#{request_id}"
    elif mode_label:
        req_label = f"<b>{_esc(mode_label)}</b>"
        status_id = mode_label
    else:
        req_label = "<b>Custom</b>"
        status_id = "custom"

    meta_html = (
        f'<div class="meta-bar">'
        f'<span>{req_label}</span>'
        f'<span>select {timing.get("select_s", 0):.1f}s</span>'
        f'<span>attacker {timing.get("attacker_s", 0):.1f}s</span>'
        f'<span>target {timing.get("target_s", 0):.1f}s</span>'
        f'<span>total <b>{float(result.get("elapsed_s", 0)):.1f}s</b></span>'
        f"</div>"
    )
    status = (
        f"Done {status_id} · {strategy_name} · "
        f"{float(result.get('elapsed_s', 0)):.1f}s"
    )
    prompt_md = to_markdown_block(
        "Attacker prompt", prompt, "No prompt generated."
    )
    response_md = to_markdown_block(
        "Target response", response, "No response generated."
    )
    return strategy_html, prompt_md, response_md, meta_html, status


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
:root {
  --ans-ink: #0f172a;
  --ans-muted: #64748b;
  --ans-line: #e2e8f0;
  --ans-surface: rgba(255,255,255,0.82);
  --ans-surface-solid: #ffffff;
  --ans-accent: #0f766e;
  --ans-accent-soft: #ccfbf1;
  --ans-attacker: #0e7490;
  --ans-target: #b45309;
  --ans-radius: 16px;
  --ans-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}

.gradio-container {
  max-width: 1120px !important;
  font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif !important;
}

/* Page atmosphere */
.gradio-container,
.main, .wrap {
  background: transparent !important;
}
body, .app {
  background:
    radial-gradient(1200px 500px at 10% -10%, #d8f3ef 0%, transparent 55%),
    radial-gradient(900px 420px at 100% 0%, #e8eef9 0%, transparent 50%),
    linear-gradient(180deg, #f7fafc 0%, #eef2f6 100%) !important;
}

.app-header {
  margin: 0.4rem 0 1.15rem;
  padding: 1.1rem 1.25rem 1.15rem;
  border-radius: var(--ans-radius);
  background:
    linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.65));
  border: 1px solid rgba(226,232,240,0.9);
  box-shadow: var(--ans-shadow);
  backdrop-filter: blur(8px);
}
.app-header .brand-mark {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  margin-bottom: 0.35rem;
}
.app-header .brand-dot {
  width: 0.7rem; height: 0.7rem; border-radius: 999px;
  background: linear-gradient(135deg, #14b8a6, #0e7490);
  box-shadow: 0 0 0 4px rgba(20,184,166,0.18);
}
.app-header .brand-kicker {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--ans-accent);
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

.meta-bar {
  display: flex; flex-wrap: wrap; gap: 0.45rem 0.55rem;
  padding: 0.65rem 0.75rem; margin: 0.15rem 0 0.85rem;
  background: var(--ans-surface-solid);
  border: 1px solid var(--ans-line);
  border-radius: 999px;
  font-size: 0.84rem; color: #334155;
  box-shadow: 0 4px 14px rgba(15,23,42,0.04);
}
.meta-bar span {
  background: #f8fafc;
  border: 1px solid var(--ans-line);
  border-radius: 999px;
  padding: 0.18rem 0.65rem;
}
.meta-bar b { color: var(--ans-ink); }

.panel {
  border: 1px solid var(--ans-line); border-radius: var(--ans-radius);
  padding: 1rem 1.1rem; background: var(--ans-surface-solid);
  box-shadow: var(--ans-shadow);
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
  margin: 0.7rem 0 0.8rem; color: #334155; line-height: 1.55; font-size: 0.95rem;
}
.chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
.chip-label { font-size: 0.8rem; color: var(--ans-muted); margin-right: 0.15rem; }
.chip {
  display: inline-block; background: var(--ans-accent-soft); color: #115e59;
  border: 1px solid #99f6e4; border-radius: 999px;
  padding: 0.18rem 0.65rem; font-size: 0.78rem; font-weight: 650;
}
.muted { color: #94a3b8; font-size: 0.85rem; }

.result-md {
  border: 1px solid var(--ans-line); border-radius: var(--ans-radius);
  padding: 1rem 1.1rem; background: var(--ans-surface-solid);
  margin-bottom: 0.85rem; box-shadow: var(--ans-shadow);
}
.result-md h3 {
  margin: 0 0 0.65rem 0; font-size: 0.78rem; font-weight: 750;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--ans-muted);
}
.result-md p, .result-md li { line-height: 1.6; color: #1e293b; }
.result-md pre, .result-md code {
  white-space: pre-wrap; word-break: break-word;
  background: #f8fafc; border-radius: 8px;
}

.request-one-box, .input-mode-box, .models-panel, .action-panel {
  border: 1px solid var(--ans-line);
  border-radius: var(--ans-radius);
  background: var(--ans-surface-solid);
  box-shadow: var(--ans-shadow);
  /* Do NOT use backdrop-filter/transform here — they offset Gradio dropdown portals */
  overflow: visible;
}
.request-one-box { margin-bottom: 0; }
.request-one-box .request-title {
  font-size: 0.78rem; font-weight: 750; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--ans-muted);
  padding: 0.75rem 1rem 0.35rem;
}
.request-one-box .request-pick {
  padding: 0 0.7rem 0.45rem;
  border-bottom: 1px solid var(--ans-line);
}
.request-one-box .request-pick .wrap {
  box-shadow: none !important;
  border: none !important;
  background: transparent !important;
}
.request-one-box .request-body { padding: 0.25rem 0.45rem 0.45rem; }
.request-one-box .request-body textarea {
  border: none !important;
  box-shadow: none !important;
  background: rgba(248,250,252,0.9) !important;
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
  line-height: 1.6 !important;
  min-height: 8.5rem !important;
  max-height: none !important;
  resize: vertical !important;
  border-radius: 12px !important;
}

.input-mode-box {
  padding: 0.75rem 0.95rem 0.55rem;
  height: 100%;
}
.input-row {
  align-items: stretch !important;
  gap: 0.9rem;
  margin: 0.15rem 0 0.9rem;
}

.models-panel {
  padding: 0.95rem 1.05rem 0.8rem;
  margin-bottom: 0.95rem;
}
.models-panel .md h3, .models-panel h3 {
  margin: 0 0 0.55rem !important;
  font-size: 0.78rem !important;
  font-weight: 750 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--ans-muted) !important;
}
.models-status {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
  margin: 0.2rem 0 0.35rem;
}
.model-card {
  border: 1px solid var(--ans-line);
  border-radius: 14px;
  padding: 0.85rem 0.95rem;
  background: linear-gradient(180deg, #fff, #f8fafc);
  position: relative;
  border-left: 4px solid var(--ans-line);
}
.model-card-attacker { border-left-color: var(--ans-attacker); }
.model-card-target { border-left-color: var(--ans-target); }
.model-role {
  font-size: 0.7rem; font-weight: 750; letter-spacing: 0.08em;
  text-transform: uppercase;
}
.model-card-attacker .model-role { color: var(--ans-attacker); }
.model-card-target .model-role { color: var(--ans-target); }
.model-hint { font-size: 0.8rem; color: var(--ans-muted); margin: 0.2rem 0 0.45rem; }
.model-name {
  font-size: 1.08rem; font-weight: 750; color: var(--ans-ink);
  letter-spacing: -0.02em; line-height: 1.25;
}
.model-shared {
  grid-column: 1 / -1;
  display: inline-flex; align-items: center; gap: 0.4rem;
  font-size: 0.82rem; color: var(--ans-muted);
  padding-top: 0.2rem;
}
.model-shared .dot {
  width: 0.45rem; height: 0.45rem; border-radius: 999px;
  background: #14b8a6;
}

.action-panel {
  padding: 0.75rem 0.9rem;
  margin: 0.15rem 0 0.85rem;
}
.results-head {
  margin: 0.55rem 0 0.65rem;
  font-size: 0.78rem; font-weight: 750; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--ans-muted);
}

/* Soften Gradio chrome */
button.primary, .primary {
  border-radius: 12px !important;
  box-shadow: 0 8px 18px rgba(15,118,110,0.22) !important;
}
.secondary-wrap button, button.secondary {
  border-radius: 12px !important;
}
footer { display: none !important; }

@media (max-width: 720px) {
  .models-status { grid-template-columns: 1fr; }
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

    target_preset_choices = MODEL_PRESETS + [("Custom HF repo…", CUSTOM_PRESET)]
    attacker_preset_choices = [
        ("Same as Target", SAME_AS_TARGET),
    ] + MODEL_PRESETS + [("Custom HF repo…", CUSTOM_PRESET)]

    preset_repos = {repo for _, repo in MODEL_PRESETS}
    if args.target_repo and args.target_repo not in preset_repos:
        default_target_preset = CUSTOM_PRESET
        default_target_custom = args.target_repo
    elif kit.target_repo in preset_repos:
        default_target_preset = kit.target_repo
        default_target_custom = ""
    else:
        default_target_preset = CUSTOM_PRESET
        default_target_custom = kit.target_repo

    if args.agent_repo and args.agent_repo not in preset_repos:
        default_attacker_preset = CUSTOM_PRESET
        default_attacker_custom = args.agent_repo
    elif args.agent_repo and args.agent_repo in preset_repos:
        default_attacker_preset = args.agent_repo
        default_attacker_custom = ""
    elif kit.agent_repo in preset_repos:
        # Prefer showing the concrete model (e.g. Qwen), not "Same as Target"
        default_attacker_preset = kit.agent_repo
        default_attacker_custom = ""
    else:
        default_attacker_preset = DEFAULT_MODEL_REPO
        default_attacker_custom = ""

    ROLE_ATTACKER = "attacker"
    ROLE_TARGET = "target"

    MODE_REQUEST = "request"
    MODE_PROMPT = "prompt"

    empty_meta = (
        '<div class="meta-bar"><span class="muted">Waiting for a run…</span></div>'
    )
    empty_strategy = (
        '<div class="panel panel-strategy">'
        '<div class="panel-label">Strategy</div>'
        '<div class="panel-title">Ready when you are</div>'
        '<p class="panel-desc muted">Click <b>Run demo</b> to select strategy, '
        "generate an attacker prompt, and query the target.</p>"
        "</div>"
    )

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
        return (
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

    def on_role_toggle(role):
        is_attacker = role == ROLE_ATTACKER
        return (
            gr.update(visible=is_attacker),
            gr.update(visible=not is_attacker),
        )

    def on_attacker_preset_change(preset):
        return gr.update(visible=preset == CUSTOM_PRESET)

    def on_target_preset_change(preset):
        return gr.update(visible=preset == CUSTOM_PRESET)

    def on_apply_models(target_preset, target_custom, attacker_preset, attacker_custom):
        try:
            preview_repo = None
            if target_preset == CUSTOM_PRESET:
                preview_repo = (target_custom or "").strip()
            elif target_preset and target_preset != SAME_AS_TARGET:
                preview_repo = target_preset
            if preview_repo and not model_cache_ready(args.chat_config, preview_repo):
                logger.warning(
                    "Model %s is not cached locally — download may take a long time",
                    preview_repo,
                )

            apply_models_to_kit(
                state["kit"],
                args,
                logger,
                target_preset,
                target_custom or "",
                attacker_preset,
                attacker_custom or "",
            )
            k = state["kit"]
            msg = (
                f"Models loaded · Attacker={short_model_name(k.agent_repo)} · "
                f"Target={short_model_name(k.target_repo)}"
            )
            return models_status_html(k), msg
        except SystemExit as exc:
            return models_status_html(state["kit"]), f"Error: {exc}"
        except Exception as exc:
            return models_status_html(state["kit"]), f"Error: {exc}"

    def on_run(mode, rid, request_text, prompt_text):
        k = state["kit"]
        try:
            if mode == MODE_PROMPT:
                result = k.run_prompt_only(prompt_text or "")
                return format_result_panels(
                    None, "", result, mode_label="Direct prompt"
                )

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
            return format_result_panels(rid_out, request, result, mode_label=label)
        except Exception as exc:
            err = (
                f'<div class="panel"><div class="panel-label">Error</div>'
                f'<pre style="white-space:pre-wrap">{_esc(str(exc))}</pre></div>'
            )
            return (
                err,
                "### Attacker prompt\n\n_Error._",
                "### Target response\n\n_Error._",
                empty_meta,
                f"Error: {exc}",
            )

    with gr.Blocks(title="Adversarial NeuroSearch Demo") as demo:
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
            models_html = gr.HTML(value=models_status_html(kit))

        with gr.Row(elem_classes=["input-row"]):
            with gr.Column(scale=1, min_width=240):
                with gr.Group(elem_classes=["input-mode-box"]):
                    input_mode = gr.Radio(
                        choices=[
                            ("Request", MODE_REQUEST),
                            ("Attack prompt", MODE_PROMPT),
                        ],
                        value=MODE_REQUEST,
                        label="Input mode",
                    )
            with gr.Column(scale=3, min_width=360):
                with gr.Group(elem_classes=["request-one-box"]) as request_box:
                    gr.HTML('<div class="request-title">Request</div>')
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
                    gr.HTML('<div class="request-title">Attacker prompt</div>')
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

        with gr.Group(elem_classes=["action-panel"]):
            run_btn = gr.Button("Run demo", variant="primary", size="lg")
            status = gr.Textbox(
                label="Status",
                value="Ready.",
                interactive=False,
                max_lines=1,
                container=True,
            )

        gr.HTML('<div class="results-head">Results</div>')
        meta_html = gr.HTML(value=empty_meta)
        strategy_html = gr.HTML(value=empty_strategy)
        prompt_md = gr.Markdown(
            value="### Attacker prompt\n\n_Generated prompt will appear here…_",
            elem_classes=["result-md"],
        )
        response_md = gr.Markdown(
            value="### Target response\n\n_Target response will appear here…_",
            elem_classes=["result-md"],
        )

        with gr.Accordion("Edit models", open=False):
            with gr.Group(elem_classes=["models-panel"]):
                model_role = gr.Radio(
                    choices=[
                        ("Attacker", ROLE_ATTACKER),
                        ("Target", ROLE_TARGET),
                    ],
                    value=ROLE_TARGET,
                    label="Role",
                    info="Toggle which role to change",
                )
                with gr.Group(visible=False) as attacker_edit:
                    attacker_preset_dd = gr.Dropdown(
                        choices=attacker_preset_choices,
                        value=default_attacker_preset,
                        label="Attacker model",
                        info="Generates the jailbreak prompt",
                    )
                    attacker_custom_tb = gr.Textbox(
                        value=default_attacker_custom,
                        label="Custom Attacker HF repo",
                        placeholder="org/model-name",
                        visible=(default_attacker_preset == CUSTOM_PRESET),
                    )
                with gr.Group(visible=True) as target_edit:
                    target_preset_dd = gr.Dropdown(
                        choices=target_preset_choices,
                        value=default_target_preset,
                        label="Target model",
                        info="Answers the jailbreak prompt",
                    )
                    target_custom_tb = gr.Textbox(
                        value=default_target_custom,
                        label="Custom Target HF repo",
                        placeholder="org/model-name",
                        visible=(default_target_preset == CUSTOM_PRESET),
                    )
                apply_models_btn = gr.Button("Apply / reload models", variant="secondary")

        apply_models_btn.click(
            on_apply_models,
            inputs=[
                target_preset_dd,
                target_custom_tb,
                attacker_preset_dd,
                attacker_custom_tb,
            ],
            outputs=[models_html, status],
        )
        model_role.change(
            on_role_toggle,
            inputs=[model_role],
            outputs=[attacker_edit, target_edit],
        )
        attacker_preset_dd.change(
            on_attacker_preset_change,
            inputs=[attacker_preset_dd],
            outputs=[attacker_custom_tb],
        )
        target_preset_dd.change(
            on_target_preset_change,
            inputs=[target_preset_dd],
            outputs=[target_custom_tb],
        )
        input_mode.change(
            on_mode_change,
            inputs=[input_mode],
            outputs=[request_box, prompt_box],
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
            outputs=[strategy_html, prompt_md, response_md, meta_html, status],
        )

    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        css=DEMO_CSS,
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Plus Jakarta Sans"),
            font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        ).set(
            body_background_fill="*neutral_50",
            block_radius="*radius_lg",
            button_primary_background_fill="#0f766e",
            button_primary_background_fill_hover="#0d9488",
            button_primary_text_color="#ffffff",
            border_color_primary="#e2e8f0",
        ),
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
