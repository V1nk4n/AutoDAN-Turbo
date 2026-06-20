"""Evaluate a trained baseline run (framework_baseline + pipeline_baseline)."""

from framework import Retrieval
from framework.harmbench_classifier import HarmBenchClassifier
from framework_baseline import Attacker as BaselineAttacker
from framework_baseline import Scorer as BaselineScorer
from framework_baseline import Summarizer as BaselineSummarizer
from framework_baseline import Target as BaselineTarget
from llm import OpenAIEmbeddingModel
from llm.huggingface_models_baseline import HuggingFaceModel as HuggingFaceModelBaseline
from llm.target_resolve import resolve_target_model
from pipeline_baseline import AutoDANTurboBaseline
import argparse
import datetime
import json
import logging
import os
import pickle
import wandb


def config():
    parser = argparse.ArgumentParser(
        description="Evaluate trained baseline pipeline (aligned with main.py baseline flags).",
    )
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--target_repo", type=str, default=None)
    parser.add_argument("--target_config", type=str, default=None)
    parser.add_argument("--agent_repo", type=str, default=None)
    parser.add_argument("--agent_config", type=str, default=None)
    parser.add_argument("--chat_config", type=str, default="./llm/chat_templates")

    parser.add_argument("--data", type=str, default="./data/harmful_behavior_requests.json")
    parser.add_argument(
        "--split",
        type=str,
        default="lifelong",
        choices=["warm_up", "lifelong"],
        help="Which split inside the JSON to evaluate on",
    )
    parser.add_argument(
        "--strategy_library_pkl",
        type=str,
        default="./logs/lifelong_strategy_library.pkl",
        help="Frozen strategy library from a baseline training run",
    )

    parser.add_argument("--epochs", type=int, default=50, help="Attack retries per request during eval")
    parser.add_argument("--break_score", type=float, default=8.5)
    parser.add_argument("--max_requests", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_json", type=str, default="./logs/eval_baseline_results.json")

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
    per_run_root = os.path.join(log_dir, "eval_baseline_per_run")
    os.makedirs(per_run_root, exist_ok=True)
    run_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    per_run_dir = os.path.join(per_run_root, run_stamp)
    os.makedirs(per_run_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "eval_baseline.log")
    per_run_log = os.path.join(per_run_dir, "running.log")

    logger = logging.getLogger("EvalBaselineLogger")
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


if __name__ == "__main__":
    args = config().parse_args()
    logger, file_formatter, per_run_dir = setup_logger()

    if os.environ.get("WANDB_MODE", "").lower() != "disabled":
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        wandb.init(project="AutoDAN-Turbo", name=f"eval-baseline-{utc_now}")
        try:
            os.makedirs(wandb.run.dir, exist_ok=True)
            wandb_log_file = os.path.join(wandb.run.dir, "eval_baseline.log")
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

    with open(args.strategy_library_pkl, "rb") as f:
        strategy_library = pickle.load(f)
    logger.info("Loaded strategy library (%d strategies) from %s", len(strategy_library), args.strategy_library_pkl)

    config_dir = args.chat_config
    hf_token = args.hf_token

    repo_name, config_name = resolve_target_model(
        model_preset=args.model,
        target_repo=args.target_repo,
        target_config=args.target_config,
        config_dir=config_dir,
    )
    logger.info("Target model: %s (generation_config=%s)", repo_name, config_name)

    import torch

    def _load_baseline_hf(repo: str, cfg: str) -> HuggingFaceModelBaseline:
        return HuggingFaceModelBaseline(
            repo,
            config_dir,
            cfg,
            hf_token,
            use_quantization=True,
            quantization_type="4bit",
        )

    target_model = _load_baseline_hf(repo_name, config_name)

    if args.agent_repo:
        agent_repo, agent_config_name = resolve_target_model(
            model_preset=args.model,
            target_repo=args.agent_repo,
            target_config=args.agent_config,
            config_dir=config_dir,
        )
        if agent_repo == repo_name and agent_config_name == config_name:
            agent_model = target_model
        else:
            agent_model = _load_baseline_hf(agent_repo, agent_config_name)
        attacker = BaselineAttacker(agent_model)
        summarizer = BaselineSummarizer(agent_model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        scorer = BaselineScorer(agent_model)
        logger.info(
            "Baseline agent stack (attacker/summarizer/scorer): %s (generation_config=%s)",
            agent_repo,
            agent_config_name,
        )
    else:
        attacker = BaselineAttacker(target_model)
        summarizer = BaselineSummarizer(target_model)
        scorer_repo = "Qwen/Qwen2.5-1.5B-Instruct"
        scorer_config = "Qwen2.5-1.5B-Instruct"
        scorer_model = _load_baseline_hf(scorer_repo, scorer_config)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        scorer = BaselineScorer(scorer_model)
        logger.info(
            "Baseline scorer (legacy split): %s (generation_config=%s)",
            scorer_repo,
            scorer_config,
        )

    target = BaselineTarget(target_model)

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
    attack_kit = {
        "attacker": attacker,
        "scorer": scorer,
        "summarizer": summarizer,
        "retrival": retrieval,
        "logger": logger,
    }

    pipeline = AutoDANTurboBaseline(
        turbo_framework=attack_kit,
        data={"warm_up": [], "lifelong": []},
        target=target,
        epochs=args.epochs,
        break_score=args.break_score,
        warm_up_iterations=1,
        lifelong_iterations=1,
        log_every=args.log_every,
    )

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
        strategy_library,
        break_score=args.break_score,
        max_requests=args.max_requests,
        log_every=args.log_every,
        harmbench_classifier=harmbench_classifier,
        contexts=None,
    )
    report["eval_config"] = {
        "data": args.data,
        "split": args.split,
        "strategy_library_pkl": args.strategy_library_pkl,
        "target_repo": repo_name,
        "target_config": config_name,
        "agent_repo": args.agent_repo,
        "agent_config": args.agent_config,
        "epochs": args.epochs,
        "break_score": args.break_score,
    }

    method = report.get("evaluation_method", "llm_scorer")
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
            "[EVAL DONE] method=%s total=%s successful=%s ASR=%.4f avg_score=%.4f break_score=%s",
            method,
            report["total"],
            report["successful"],
            report["asr"],
            report.get("avg_score", 0.0),
            report.get("break_score", args.break_score),
        )

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        per_run_report = os.path.join(per_run_dir, "eval_results.json")
        with open(per_run_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Saved eval report to %s and %s", args.save_json, per_run_report)
