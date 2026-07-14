"""Resolve HuggingFace target repo + generation config for a single run."""

from __future__ import annotations

import os
from typing import Optional, Tuple

_PRESET_LLAMA3 = ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct")
_PRESET_SMOLLM2 = (
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "SmolLM2-1.7B-Instruct",
)
_PRESET_DEFAULT = ("google/gemma-1.1-7b-it", "gemma-it")
<<<<<<< HEAD
_PRESET_PHI15 = ("microsoft/phi-1_5", "phi-1_5")
_PHI_PRESETS = {"phi", "phi15", "phi-1.5", "phi-1_5", "phi1.5"}
=======
_HF_TOKEN_PLACEHOLDER = "your_hf_token"


def resolve_hf_token(token: Optional[str] = None) -> Optional[str]:
    """Use CLI token when set; otherwise HF_TOKEN / HUGGING_FACE_HUB_TOKEN from env."""
    if token and token.strip() and token.strip() != _HF_TOKEN_PLACEHOLDER:
        return token.strip()
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env_val = os.environ.get(env_name)
        if env_val and env_val.strip():
            return env_val.strip()
    return None
>>>>>>> origin/dev-time-improve-strategy


def list_generation_configs(config_dir: str) -> list[str]:
    gen_dir = os.path.join(config_dir, "generation_configs")
    if not os.path.isdir(gen_dir):
        return []
    return sorted(f[:-5] for f in os.listdir(gen_dir) if f.endswith(".json"))


def infer_config_name(repo_name: str, config_dir: str) -> Optional[str]:
    tail = repo_name.split("/")[-1]
    candidates = [tail]
    if tail.endswith("-Instruct"):
        candidates.append(tail[: -len("-Instruct")])
    for name in candidates:
        path = os.path.join(config_dir, "generation_configs", f"{name}.json")
        if os.path.isfile(path):
            return name
    return None


def resolve_target_model(
    *,
    model_preset: str,
    target_repo: Optional[str],
    target_config: Optional[str],
    config_dir: str,
) -> Tuple[str, str]:
    if target_repo:
        repo_name = target_repo.strip()
        if target_config:
            config_name = target_config.strip()
        else:
            config_name = infer_config_name(repo_name, config_dir)
            if not config_name:
                available = list_generation_configs(config_dir)
                raise SystemExit(
                    f"Cannot infer --target_config for {repo_name}. "
                    f"Pass --target_config explicitly. "
                    f"Available: {', '.join(available)}"
                )
    elif model_preset.lower() == "smollm2":
        repo_name, config_name = _PRESET_SMOLLM2
    elif model_preset == "llama3":
        repo_name, config_name = _PRESET_LLAMA3
    elif (model_preset or "").strip().lower() in _PHI_PRESETS:
        repo_name, config_name = _PRESET_PHI15
    else:
        repo_name, config_name = _PRESET_DEFAULT

    config_path = os.path.join(config_dir, "generation_configs", f"{config_name}.json")
    if not os.path.isfile(config_path):
        available = list_generation_configs(config_dir)
        raise SystemExit(
            f"Generation config not found: {config_path}. "
            f"Available: {', '.join(available)}"
        )
    return repo_name, config_name
