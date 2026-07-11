from framework import Attacker, Scorer, Summarizer, Retrieval, Target, Feedback, Refiner, PatternManager
from framework.harmbench_classifier import HarmBenchClassifier
from llm import HuggingFaceModel, OpenAIEmbeddingModel
from llm.target_resolve import resolve_hf_token, resolve_target_model
import argparse
import json
import logging
import os
import datetime
import wandb

from main import load_epoch_memory
from pipeline_pro import AutoDANTurboPro


def config():
    parser = argparse.ArgumentParser(
        description="Evaluate trained PRO pipeline (aligned with main.py training flags).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Model preset: llama3, smollm2, or gemma fallback",
    )
    parser.add_argument("--target_repo", type=str, default=None)
    parser.add_argument("--target_config", type=str, default=None)
    parser.add_argument("--agent_repo", type=str, default=None,
                        help="HuggingFace repo id for attacker/summarizer/scorer/feedback/refiner (default: same as target)")
    parser.add_argument("--agent_config", type=str, default=None,
                        help="Generation config for agent model (inferred from repo tail if omitted)")
    parser.add_argument("--chat_config", type=str, default="./llm/chat_templates")

    parser.add_argument("--data", type=str, default="./data/harmful_behavior_requests.json")
    parser.add_argument(
        "--split",
        type=str,
        default="lifelong",
        choices=["warm_up", "lifelong"],
        help="Which split inside the JSON to evaluate on",
    )

    parser.add_argument("--pro_n_candidates", type=int, default=4)
    parser.add_argument("--pro_top_k", type=int, default=2)
    parser.add_argument("--pro_score_threshold", type=float, default=0.5)
    parser.add_argument(
        "--pro_per_request_epochs",
        action="store_true",
        help="Match training: up to `epochs` repeats per request with early stop + feedback",
    )
    parser.add_argument(
        "--single_shot_eval",
        action="store_true",
        help="Force one attack_single_turn per request (ignore --pro_per_request_epochs)",
    )
    parser.add_argument("--pro_early_stop_patience", type=int, default=5)
    parser.add_argument("--pro_early_stop_min_delta", type=float, default=0.15)
    parser.add_argument("--pro_refusal_streak_stop", type=int, default=4)
    parser.add_argument("--pro_feedback_every", type=int, default=2)
    parser.add_argument(
        "--pro_feedback_min_quality",
        type=float,
        default=3.5,
        help="Feedback when best failed score_loss exceeds this (0-10 scale with four-tier)",
    )
    parser.add_argument("--pro_phase_split", type=float, default=0.7)
    parser.add_argument("--pro_explore_n_candidates", type=int, default=2)
    parser.add_argument("--pro_explore_top_k", type=int, default=1)
    parser.add_argument("--pro_exploit_n_candidates", type=int, default=4)
    parser.add_argument("--pro_exploit_top_k", type=int, default=2)
    parser.add_argument("--pro_explore_max_new_tokens", type=int, default=64)
    parser.add_argument("--pro_exploit_max_new_tokens", type=int, default=128)
    parser.add_argument("--pro_enable_eval_cache", action="store_true")
    parser.add_argument("--pro_eval_batch_size", type=int, default=2)
    parser.add_argument("--pro_enable_retrieval_cache", action="store_true")
    parser.add_argument("--pro_enable_fast_judge", action="store_true")
    parser.add_argument("--pro_fast_judge_min_len", type=int, default=24)
    parser.add_argument("--pro_enable_feedback_scheduler", action="store_true")
    parser.add_argument("--pro_feedback_budget_ms", type=float, default=5000.0)
    parser.add_argument("--pro_feedback_min_delta", type=float, default=0.02)
    parser.add_argument("--pro_feedback_cooldown_turns", type=int, default=1)
    parser.add_argument("--pro_disable_feedback_refine", action="store_true")
    parser.add_argument("--pro_disable_dual_scorer", action="store_true", help="Do not load Qwen3-0.6B wrapper model")

    parser.add_argument("--mfps_enabled", action="store_true")
    parser.add_argument("--mfps_profile", type=str, default="balanced", choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--mfps_alpha0", type=float, default=0.5)
    parser.add_argument("--mfps_alpha1", type=float, default=0.5)
    parser.add_argument("--mfps_short_max_new_tokens", type=int, default=32)
    parser.add_argument("--mfps_min_candidates_f2", type=int, default=1)
    parser.add_argument("--mfps_uncertainty_band", type=float, default=0.1)
    parser.add_argument("--mfps_eval_budget_ms", type=float, default=0.0)
    parser.add_argument("--mfps_w_f0", type=float, default=0.35)
    parser.add_argument("--mfps_w_f1", type=float, default=0.65)
    parser.add_argument("--mfps_uncertainty_penalty", type=float, default=0.2)

    parser.add_argument("--target_max_new_tokens", type=int, default=150)
    parser.add_argument("--nll_min", type=float, default=2.0)
    parser.add_argument("--nll_max", type=float, default=5.0)

    parser.add_argument("--pro_four_tier_eval", dest="pro_four_tier_eval", action="store_true")
    parser.add_argument("--pro_no_four_tier_eval", dest="pro_four_tier_eval", action="store_false")
    parser.set_defaults(pro_four_tier_eval=True)
    parser.add_argument("--pro_hybrid_mfps_four_tier", dest="pro_hybrid_mfps_four_tier", action="store_true")
    parser.add_argument("--pro_no_hybrid_mfps_four_tier", dest="pro_hybrid_mfps_four_tier", action="store_false")
    parser.set_defaults(pro_hybrid_mfps_four_tier=True)
    parser.add_argument("--pro_verifier_top_n", type=int, default=1)
    parser.add_argument("--pro_dynamic_pattern_select", action="store_true")
    parser.add_argument("--pro_pattern_exploit_n", type=int, default=3)
    parser.add_argument("--pro_pattern_explore_n", type=int, default=2)
    parser.add_argument("--pro_pattern_rank_w_rate", type=float, default=0.3)
    parser.add_argument("--pro_pattern_rank_w_avg", type=float, default=0.3)
    parser.add_argument("--pro_pattern_rank_w_req", type=float, default=0.4)
    parser.add_argument("--pro_pattern_explore_seed", type=int, default=None)
    parser.add_argument("--pro_goal_similarity_floor", type=float, default=0.15)

    parser.add_argument("--pattern_force_seed", action="store_true", help="Reset pattern library to seed (usually off for eval)")
    parser.add_argument("--pattern_frozen", action="store_true", help="Do not update pattern_library.json during eval")
    parser.add_argument("--pattern_filepath", type=str, default="./logs/pattern_library.json")
    parser.add_argument(
        "--epoch_memory_file",
        type=str,
        default="./logs/epoch_refine_memory.json",
        help="Cross-epoch refine hints from training",
    )
    parser.add_argument("--skip_epoch_memory", action="store_true", help="Do not load epoch_refine_memory for GOAT hints")

    parser.add_argument("--epochs", type=int, default=50, help="Max repeats per request when --pro_per_request_epochs")
    parser.add_argument("--break_score", type=float, default=8.5)
    parser.add_argument("--max_requests", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_json", type=str, default="./logs/eval_results.json")

    parser.add_argument("--use_harmbench_classifier", action="store_true")
    parser.add_argument("--harmbench_classifier_model", type=str, default="cais/HarmBench-Mistral-7b-val-cls")
    parser.add_argument("--harmbench_no_quantization", action="store_true")
    parser.add_argument("--harmbench_quantization_type", type=str, default="4bit", choices=["4bit", "8bit"])
    parser.add_argument("--use_ollama", action="store_true")
    parser.add_argument("--ollama_model_name", type=str, default="mistral:7b")
    parser.add_argument("--ollama_base_url", type=str, default="http://localhost:11434")

    parser.add_argument("--azure", action="store_true")
    parser.add_argument("--azure_endpoint", type=str, default="your_azure_endpoint")
    parser.add_argument("--azure_api_version", type=str, default="2024-02-01")
    parser.add_argument("--azure_deployment_name", type=str, default="your_azure_deployment_name")
    parser.add_argument("--azure_api_key", type=str, default="your_azure_api_key")
    parser.add_argument("--openai_api_key", type=str, default="your_openai_api_key")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-ada-002")
    parser.add_argument("--use_local_embedding", action="store_true")
    parser.add_argument("--local_embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--hf_token", type=str, default="your_hf_token")
    return parser


def load_eval_requests(path: str, split: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and split in data:
        return data[split]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset format in {path}. Expected dict with key '{split}' or list[str].")


def setup_logger():
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    per_run_root = os.path.join(log_dir, "eval_per_run")
    os.makedirs(per_run_root, exist_ok=True)
    run_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    per_run_dir = os.path.join(per_run_root, run_stamp)
    os.makedirs(per_run_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "eval_pro.log")
    per_run_log = os.path.join(per_run_dir, "running.log")

    logger = logging.getLogger("EvalProLogger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    for path in (log_file, per_run_log):
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    logger.info("Eval per-run log: %s", per_run_log)
    return logger, file_formatter, per_run_dir


def build_pipeline_kwargs(args, repo_name: str) -> dict:
    per_request_epochs = bool(args.pro_per_request_epochs) and not args.single_shot_eval
    return dict(
        epochs=args.epochs,
        warm_up_iterations=1,
        lifelong_iterations=1,
        log_every=args.log_every,
        pro_n_candidates=args.pro_n_candidates,
        pro_top_k=args.pro_top_k,
        pro_score_threshold=args.pro_score_threshold,
        target_max_new_tokens=args.target_max_new_tokens,
        nll_min=args.nll_min,
        nll_max=args.nll_max,
        target_model_key=repo_name,
        per_request_epochs=per_request_epochs,
        pro_hybrid_mfps_four_tier=args.pro_hybrid_mfps_four_tier,
        pro_four_tier_eval=args.pro_four_tier_eval,
        pro_verifier_top_n=args.pro_verifier_top_n,
        pro_dynamic_pattern_select=args.pro_dynamic_pattern_select,
        pro_pattern_exploit_n=args.pro_pattern_exploit_n,
        pro_pattern_explore_n=args.pro_pattern_explore_n,
        pro_pattern_rank_w_rate=args.pro_pattern_rank_w_rate,
        pro_pattern_rank_w_avg=args.pro_pattern_rank_w_avg,
        pro_pattern_rank_w_req=args.pro_pattern_rank_w_req,
        pro_pattern_explore_seed=args.pro_pattern_explore_seed,
        pro_goal_similarity_floor=args.pro_goal_similarity_floor,
        pro_early_stop_patience=args.pro_early_stop_patience,
        pro_early_stop_min_delta=args.pro_early_stop_min_delta,
        pro_refusal_streak_stop=args.pro_refusal_streak_stop,
        pro_feedback_every=args.pro_feedback_every,
        pro_feedback_min_quality=args.pro_feedback_min_quality,
        pro_phase_split=args.pro_phase_split,
        pro_explore_n_candidates=args.pro_explore_n_candidates,
        pro_explore_top_k=args.pro_explore_top_k,
        pro_exploit_n_candidates=args.pro_exploit_n_candidates,
        pro_exploit_top_k=args.pro_exploit_top_k,
        pro_explore_max_new_tokens=args.pro_explore_max_new_tokens,
        pro_exploit_max_new_tokens=args.pro_exploit_max_new_tokens,
        pro_enable_eval_cache=args.pro_enable_eval_cache,
        pro_eval_batch_size=args.pro_eval_batch_size,
        pro_enable_retrieval_cache=args.pro_enable_retrieval_cache,
        pro_enable_fast_judge=args.pro_enable_fast_judge,
        pro_fast_judge_min_len=args.pro_fast_judge_min_len,
        pro_enable_feedback_scheduler=args.pro_enable_feedback_scheduler,
        pro_feedback_budget_ms=args.pro_feedback_budget_ms,
        pro_feedback_min_delta=args.pro_feedback_min_delta,
        pro_feedback_cooldown_turns=args.pro_feedback_cooldown_turns,
        pro_enable_feedback_refine=not args.pro_disable_feedback_refine,
        mfps_enabled=args.mfps_enabled,
        mfps_profile=args.mfps_profile,
        mfps_alpha0=args.mfps_alpha0,
        mfps_alpha1=args.mfps_alpha1,
        mfps_short_max_new_tokens=args.mfps_short_max_new_tokens,
        mfps_min_candidates_f2=args.mfps_min_candidates_f2,
        mfps_uncertainty_band=args.mfps_uncertainty_band,
        mfps_eval_budget_ms=args.mfps_eval_budget_ms,
        mfps_w_f0=args.mfps_w_f0,
        mfps_w_f1=args.mfps_w_f1,
        mfps_uncertainty_penalty=args.mfps_uncertainty_penalty,
    )


if __name__ == "__main__":
    args = config().parse_args()
    logger, file_formatter, per_run_dir = setup_logger()

    if os.environ.get("WANDB_MODE", "").lower() != "disabled":
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        wandb.init(project="AutoDAN-Turbo", name=f"eval-{utc_now}")
        try:
            os.makedirs(wandb.run.dir, exist_ok=True)
            wandb_log_file = os.path.join(wandb.run.dir, "eval.log")
            wandb_file_handler = logging.FileHandler(wandb_log_file)
            wandb_file_handler.setLevel(logging.INFO)
            wandb_file_handler.setFormatter(file_formatter)
            logger.addHandler(wandb_file_handler)
            wandb.save(wandb_log_file, policy="live")
            logger.info("Logging to wandb directory: %s", wandb_log_file)
        except Exception as e:
            logger.warning("Failed to setup wandb logging: %s", e)

    eval_requests = load_eval_requests(args.data, args.split)
    logger.info("Loaded %d eval requests from %s split=%s", len(eval_requests), args.data, args.split)

    config_dir = args.chat_config
    hf_token = resolve_hf_token(args.hf_token)

    repo_name, config_name = resolve_target_model(
        model_preset=args.model,
        target_repo=args.target_repo,
        target_config=args.target_config,
        config_dir=config_dir,
    )
    logger.info("Target model: %s (generation_config=%s)", repo_name, config_name)

    target_model = HuggingFaceModel(
        repo_name,
        config_dir,
        config_name,
        hf_token,
        use_quantization=True,
        quantization_type="4bit",
    )

    # Agent model (attacker / summarizer / scorer / feedback / refiner)
    if args.agent_repo:
        agent_repo_name, agent_config_name = resolve_target_model(
            model_preset=args.model,
            target_repo=args.agent_repo,
            target_config=args.agent_config,
            config_dir=config_dir,
        )
        if agent_repo_name == repo_name and agent_config_name == config_name:
            agent_model = target_model
        else:
            agent_model = HuggingFaceModel(
                agent_repo_name,
                config_dir,
                agent_config_name,
                hf_token,
                use_quantization=True,
                quantization_type="4bit",
            )
        logger.info("Agent model: %s (generation_config=%s)", agent_repo_name, agent_config_name)
    else:
        agent_model = target_model
        logger.info("Agent model: same as target (%s)", repo_name)

    attacker = Attacker(agent_model)
    summarizer = Summarizer(agent_model)

    if args.pro_disable_dual_scorer:
        scorer = Scorer(agent_model)
        logger.info("PRO scorer: single model (dual x_model disabled)")
    else:
        x_model_repo = "Qwen/Qwen3-0.6B"
        x_model_config = "Qwen3-0.6B"
        x_model = HuggingFaceModel(x_model_repo, config_dir, x_model_config, hf_token)
        scorer = Scorer(agent_model, x_model)
        logger.info("PRO scorer: dual (main=%s, wrapper=%s)", agent_repo_name if args.agent_repo else repo_name, x_model_repo)

    if args.use_local_embedding:
        from llm import LocalEmbeddingModel

        text_embedding_model = LocalEmbeddingModel(
            model_name=args.local_embedding_model,
            device=None,
            logger=logger,
        )
    elif args.azure:
        text_embedding_model = OpenAIEmbeddingModel(
            azure=True,
            azure_endpoint=args.azure_endpoint,
            azure_api_version=args.azure_api_version,
            azure_deployment_name=args.azure_deployment_name,
            azure_api_key=args.azure_api_key,
            logger=logger,
        )
    else:
        text_embedding_model = OpenAIEmbeddingModel(
            azure=False,
            openai_api_key=args.openai_api_key,
            embedding_model=args.embedding_model,
            logger=logger,
        )

    retrieval = Retrieval(text_embedding_model, logger)
    target = Target(target_model)
    feedback = Feedback(agent_model)
    refiner = Refiner(agent_model)
    pattern_manager = PatternManager(
        filepath=args.pattern_filepath,
        force_seed=args.pattern_force_seed,
        frozen=args.pattern_frozen,
    )
    logger.info(
        "Pattern library: %s (%d strategies, frozen=%s)",
        args.pattern_filepath,
        len(pattern_manager.strategies),
        args.pattern_frozen,
    )

    attack_kit = {
        "attacker": attacker,
        "scorer": scorer,
        "summarizer": summarizer,
        "retrieval": retrieval,
        "logger": logger,
        "feedback": feedback,
        "refiner": refiner,
        "pattern_manager": pattern_manager,
    }

    pipeline = AutoDANTurboPro(
        turbo_framework=attack_kit,
        data={"warm_up": [], "lifelong": []},
        target=target,
        **build_pipeline_kwargs(args, repo_name),
    )

    per_request_epochs = bool(args.pro_per_request_epochs) and not args.single_shot_eval
    logger.info(
        "Eval mode: %s | four_tier=%s hybrid_mfps=%s mfps=%s exploit/explore=%s/%s",
        "repeats" if per_request_epochs else "single_shot",
        pipeline.pro_four_tier_eval,
        pipeline.pro_hybrid_mfps_four_tier,
        pipeline.mfps_enabled,
        pipeline.pro_pattern_exploit_n,
        pipeline.pro_pattern_explore_n,
    )

    if not args.skip_epoch_memory and os.path.exists(args.epoch_memory_file):
        try:
            em = load_epoch_memory(args.epoch_memory_file)
            hint = AutoDANTurboPro.build_epoch_refine_hint_from_memory(em)
            pipeline.set_epoch_refine_hint(hint)
            if hint:
                logger.info(
                    "Loaded epoch refine hint (len=%d, last_iteration=%s) from %s",
                    len(hint),
                    em.get("last_iteration"),
                    args.epoch_memory_file,
                )
            else:
                logger.info("epoch_refine_memory present but empty hint")
        except Exception as ex:
            logger.warning("Could not load epoch_refine_memory: %s", ex)
    elif args.skip_epoch_memory:
        logger.info("Skipping epoch_refine_memory (--skip_epoch_memory)")
    else:
        logger.info("No epoch_refine_memory at %s", args.epoch_memory_file)

    harmbench_classifier = None
    if args.use_harmbench_classifier:
        if args.use_ollama:
            try:
                harmbench_classifier = HarmBenchClassifier(
                    model_name=args.harmbench_classifier_model,
                    device=None,
                    logger=logger,
                    use_ollama=True,
                    ollama_model_name=args.ollama_model_name,
                    ollama_base_url=args.ollama_base_url,
                )
            except Exception as e:
                logger.error("HarmBench Ollama init failed: %s", e)
        else:
            use_quantization = not args.harmbench_no_quantization
            try:
                harmbench_classifier = HarmBenchClassifier(
                    model_name=args.harmbench_classifier_model,
                    device=None,
                    logger=logger,
                    use_quantization=use_quantization,
                    quantization_type=args.harmbench_quantization_type,
                )
            except Exception as e:
                logger.error("HarmBench init failed: %s", e)

    report = pipeline.evaluate_dataset(
        eval_requests,
        max_requests=args.max_requests,
        log_every=args.log_every,
        harmbench_classifier=harmbench_classifier,
        contexts=None,
    )
    report["eval_config"] = {
        "data": args.data,
        "split": args.split,
        "pattern_filepath": args.pattern_filepath,
        "epoch_memory_file": None if args.skip_epoch_memory else args.epoch_memory_file,
        "per_request_epochs": per_request_epochs,
        "epochs": args.epochs,
        "single_shot_eval": args.single_shot_eval,
    }

    method = report.get("evaluation_method", "")
    if "harmbench" in method:
        logger.info(
            "[EVAL DONE] method=%s total=%s successful=%s ASR=%.4f",
            method,
            report["total"],
            report["successful"],
            report["asr"],
        )
    else:
        logger.info(
            "[EVAL DONE] method=%s total=%s successful=%s ASR=%.4f",
            method,
            report["total"],
            report["successful"],
            report["asr"],
        )

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        per_run_report = os.path.join(per_run_dir, "eval_results.json")
        with open(per_run_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Saved eval report to %s and %s", args.save_json, per_run_report)
