from framework import Attacker, Scorer, Summarizer, Retrieval, Target
from framework.harmbench_classifier import HarmBenchClassifier
from llm import HuggingFaceModel, OpenAIEmbeddingModel
import argparse
import json
import logging
import os
import pickle
import datetime
import wandb

from pipeline import AutoDANTurbo


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--chat_config", type=str, default="./llm/chat_templates")

    # Dataset for evaluation (must be JSON with {"warm_up": [...], "lifelong": [...]}, or list[str])
    parser.add_argument("--data", type=str, default="./data/harmful_behavior_requests.json")
    parser.add_argument(
        "--split",
        type=str,
        default="lifelong",
        choices=["warm_up", "lifelong"],
        help="Which split inside the JSON to evaluate on",
    )

    # Strategy library to evaluate (pkl produced by main.py)
    parser.add_argument("--strategy_library_pkl", type=str, default="./logs/lifelong_strategy_library.pkl")

    # Evaluation params
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--break_score", type=float, default=8.5)
    parser.add_argument("--max_requests", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_json", type=str, default="./logs/eval_results.json")
    
    # HarmBench classifier (Option A: paper-style evaluation)
    parser.add_argument(
        "--use_harmbench_classifier",
        action="store_true",
        help="Use HarmBench classifier for evaluation (paper-style, matches AutoDAN-Turbo paper)",
    )
    parser.add_argument(
        "--harmbench_classifier_model",
        type=str,
        default="cais/HarmBench-Mistral-7b-val-cls",
        help="HarmBench classifier model name from HuggingFace",
    )
    parser.add_argument(
        "--harmbench_no_quantization",
        action="store_true",
        help="Disable quantization for HarmBench classifier (default: uses 4-bit quantization to reduce VRAM)",
    )
    parser.add_argument(
        "--harmbench_quantization_type",
        type=str,
        default="4bit",
        choices=["4bit", "8bit"],
        help="Quantization type for HarmBench classifier (default: 4bit)",
    )
    parser.add_argument(
        "--use_ollama",
        action="store_true",
        help="Use Ollama for HarmBench classifier instead of loading model directly (prevents crashes)",
    )
    parser.add_argument(
        "--ollama_model_name",
        type=str,
        default="mistral:7b",
        help="Name of model in Ollama (e.g., 'mistral:7b', 'llama2:7b')",
    )
    parser.add_argument(
        "--ollama_base_url",
        type=str,
        default="http://localhost:11434",
        help="Base URL for Ollama API (default: http://localhost:11434)",
    )

    # Embeddings
    parser.add_argument("--azure", action="store_true", help="Use azure")
    parser.add_argument("--azure_endpoint", type=str, default="your_azure_endpoint")
    parser.add_argument("--azure_api_version", type=str, default="2024-02-01")
    parser.add_argument("--azure_deployment_name", type=str, default="your_azure_deployment_name")
    parser.add_argument("--azure_api_key", type=str, default="your_azure_api_key")
    parser.add_argument("--openai_api_key", type=str, default="your_openai_api_key")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-ada-002")
    parser.add_argument("--use_local_embedding", action="store_true", help="Use local embedding model instead of OpenAI API")
    parser.add_argument(
        "--local_embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Local embedding model name from sentence-transformers",
    )

    # HF
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


if __name__ == "__main__":
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "eval.log")

    logger = logging.getLogger("EvalLogger")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    utc_now = datetime.datetime.now(datetime.timezone.utc)
    wandb.init(project="AutoDAN-Turbo", name=f"eval-{utc_now}")
    
    # ✅ Thêm file handler cho wandb directory (giống main.py)
    try:
        # wandb.run.dir trỏ đến files/, cần lấy parent để có root của run directory
        wandb_run_root = os.path.dirname(wandb.run.dir)  # Lấy parent directory
        wandb_logs_dir = os.path.join(wandb_run_root, 'logs')
        os.makedirs(wandb_logs_dir, exist_ok=True)
        wandb_log_file = os.path.join(wandb_logs_dir, 'eval.log')  # Dùng eval.log thay vì running.log
        
        wandb_file_handler = logging.FileHandler(wandb_log_file)
        wandb_file_handler.setLevel(logging.INFO)
        wandb_file_handler.setFormatter(file_formatter)
        logger.addHandler(wandb_file_handler)
        logger.info(f"✅ Logging to wandb directory: {wandb_log_file}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to setup wandb logging: {e}")

    args = config().parse_args()

    # Load evaluation requests
    eval_requests = load_eval_requests(args.data, args.split)

    # Load frozen strategy library
    with open(args.strategy_library_pkl, "rb") as f:
        strategy_library = pickle.load(f)

    # Build models (keep similar to test.py, but configurable via --model)
    config_dir = args.chat_config
    hf_token = args.hf_token

    if args.model == "llama3":
        repo_name = "meta-llama/Llama-3.2-1B-Instruct"
        config_name = "llama-3-instruct"
    else:
        repo_name = "google/gemma-1.1-7b-it"
        config_name = "gemma-it"

    model = HuggingFaceModel(repo_name, config_dir, config_name, hf_token)
    attacker = Attacker(model)
    summarizer = Summarizer(model)

    # Scorer model (default same as test.py)
    # scorer_repo_name = "google/gemma-1.1-7b-it"
    # scorer_config_name = "gemma-it"
    scorer_repo_name = "meta-llama/Llama-3.2-1B-Instruct"
    scorer_config_name = "llama-3-instruct"
    scorer_model = HuggingFaceModel(scorer_repo_name, config_dir, scorer_config_name, hf_token)
    scorer = Scorer(scorer_model)

    # Embeddings
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

    retrival = Retrieval(text_embedding_model, logger)
    target = Target(model)

    attack_kit = {
        "attacker": attacker,
        "scorer": scorer,
        "summarizer": summarizer,
        "retrival": retrival,
        "logger": logger,
    }

    # data is not used directly for eval, but required by constructor
    dummy_data = {"warm_up": [], "lifelong": []}
    pipeline = AutoDANTurbo(
        turbo_framework=attack_kit,
        data=dummy_data,
        target=target,
        epochs=args.epochs,
        break_score=args.break_score,
        warm_up_iterations=1,
        lifelong_iterations=1,
        log_every=args.log_every,
    )

    # Initialize HarmBench classifier if requested
    harmbench_classifier = None
    if args.use_harmbench_classifier:
        if args.use_ollama:
            logger.info(f"Using Ollama for HarmBench classifier: {args.ollama_model_name}")
            logger.info(f"Ollama base URL: {args.ollama_base_url}")
            try:
                harmbench_classifier = HarmBenchClassifier(
                    model_name=args.harmbench_classifier_model,
                    device=None,  # Not used with Ollama
                    logger=logger,
                    use_ollama=True,
                    ollama_model_name=args.ollama_model_name,
                    ollama_base_url=args.ollama_base_url,
                )
                logger.info("✅ HarmBench classifier initialized successfully with Ollama")
            except Exception as e:
                logger.error(f"Failed to initialize HarmBench classifier with Ollama: {e}")
                logger.error("Falling back to LLM scorer evaluation")
                harmbench_classifier = None
        else:
            logger.info(f"Initializing HarmBench classifier: {args.harmbench_classifier_model}")
            
            # Determine quantization settings (default: enabled with 4-bit)
            use_quantization = not args.harmbench_no_quantization
            if use_quantization:
                logger.info(f"Will use {args.harmbench_quantization_type} quantization to reduce VRAM")
            else:
                logger.info("Will load classifier in full precision (no quantization)")
            
            try:
                harmbench_classifier = HarmBenchClassifier(
                    model_name=args.harmbench_classifier_model,
                    device=None,  # Auto-detect
                    logger=logger,
                    use_quantization=use_quantization,
                    quantization_type=args.harmbench_quantization_type,
                )
                logger.info("✅ HarmBench classifier initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize HarmBench classifier: {e}")
                logger.error("Falling back to LLM scorer evaluation")
                harmbench_classifier = None

    # Run evaluation
    if harmbench_classifier:
        logger.info("Using HarmBench classifier for evaluation (paper-style)")
        report = pipeline.evaluate_dataset(
            eval_requests,
            strategy_library,
            max_requests=args.max_requests,
            log_every=args.log_every,
            harmbench_classifier=harmbench_classifier,
            contexts=None,  # Can be extended to support contextual evaluation
        )
        logger.info(
            f"[EVAL DONE - HarmBench] total={report['total']} successful={report['successful']} "
            f"ASR={report['asr']:.4f} failed={report['failed']}"
        )
    else:
        logger.info("Using LLM scorer for evaluation (original method)")
        report = pipeline.evaluate_dataset(
            eval_requests,
            strategy_library,
            break_score=args.break_score,
            max_requests=args.max_requests,
            log_every=args.log_every,
        )
        logger.info(
            f"[EVAL DONE - LLM Scorer] total={report['total']} successful={report['successful']} "
            f"ASR={report['asr']:.4f} avg_score={report.get('avg_score', 0):.4f} break_score={report.get('break_score', 0)}"
        )

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved eval report to {args.save_json}")

 