#!/usr/bin/env python3
"""
Script to download and prepare HarmBench dataset for evaluation.

HarmBench dataset is available on GitHub (official source):
- https://github.com/centerforaisafety/HarmBench
- Dataset files: data/behavior_datasets/harmbench_behaviors_text_*.csv

This script downloads the HarmBench dataset and converts it to the format
required by AutoDAN-Turbo: {"warm_up": [...], "lifelong": [...]}
"""

import argparse
import json
import os
from typing import List, Dict, Optional


def download_harmbench_with_datasets(
    output_path: str,
    split: str = "test",
    hf_token: Optional[str] = None
) -> Optional[Dict[str, List[str]]]:
    """
    Download HarmBench dataset using datasets library.
    
    Args:
        output_path: Path to save the converted dataset
        split: Dataset split to download ("test", "train", "val")
        hf_token: HuggingFace token (optional)
    
    Returns:
        Dictionary with "warm_up" and "lifelong" keys, or None if failed
    """
    try:
        from datasets import load_dataset
        
        print(f"📥 Downloading HarmBench dataset using datasets library (split: {split})...")
        
        # Load HarmBench dataset from HuggingFace
        dataset = load_dataset(
            "cais/harmbench",
            split=split,
            token=hf_token
        )
        print(f"✅ Downloaded {len(dataset)} examples")
        
        # Extract behaviors (malicious requests)
        behaviors = []
        for example in dataset:
            # HarmBench format: {"behavior": "...", "context": "...", ...}
            behavior = example.get("behavior", "")
            if behavior and isinstance(behavior, str):
                behaviors.append(behavior.strip())
        
        print(f"✅ Extracted {len(behaviors)} behaviors")
        
        if len(behaviors) == 0:
            print("⚠️  No behaviors found in dataset")
            return None
        
        return behaviors
        
    except ImportError:
        print("⚠️  datasets library not found, trying alternative method...")
        return None
    except Exception as e:
        print(f"⚠️  Error with datasets library: {e}")
        return None


def download_harmbench_from_github(
    output_path: str,
    split: str = "test",
    hf_token: Optional[str] = None
) -> Optional[List[str]]:
    """
    Download HarmBench dataset from GitHub repository (official source).
    
    HarmBench dataset is in CSV format with columns:
    Behavior,FunctionalCategory,SemanticCategory,Tags,ContextString,BehaviorID
    
    Args:
        output_path: Path to save the converted dataset
        split: Dataset split to download ("test", "train", "val", "all")
              "all" uses harmbench_behaviors_text_all.csv (official HarmBench evaluation dataset)
        hf_token: Not used for GitHub, kept for compatibility
    
    Returns:
        List of behaviors, or None if failed
    """
    try:
        import requests
        import csv
        from io import StringIO
        
        print(f"📥 Downloading HarmBench dataset from GitHub (split: {split})...")
        
        # HarmBench official GitHub repository
        # Repository: https://github.com/centerforaisafety/HarmBench
        github_base_url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets"
        
        # HarmBench dataset files (CSV format)
        # Official HarmBench uses harmbench_behaviors_text_all.csv for evaluation
        filename_map = {
            "test": "harmbench_behaviors_text_test.csv",
            "train": "harmbench_behaviors_text_all.csv",
            "all": "harmbench_behaviors_text_all.csv",  # Official HarmBench evaluation dataset
            "val": "harmbench_behaviors_text_val.csv"
        }
        
        filename = filename_map.get(split, "harmbench_behaviors_text_all.csv")
        github_url = f"{github_base_url}/{filename}"
        
        print(f"   URL: {github_url}")
        
        # Download from GitHub
        response = requests.get(github_url, timeout=120)
        response.raise_for_status()
        
        print(f"✅ Downloaded {len(response.text)} bytes from GitHub")
        
        # Parse CSV file
        behaviors = []
        csv_reader = csv.DictReader(StringIO(response.text))
        
        for row_num, row in enumerate(csv_reader, 1):
            behavior = row.get("Behavior", "").strip()
            if behavior:
                # Remove quotes if present
                behavior = behavior.strip('"').strip("'")
                if behavior:
                    behaviors.append(behavior)
        
        print(f"✅ Loaded {len(behaviors)} behaviors from GitHub CSV")
        
        if len(behaviors) == 0:
            print("⚠️  No behaviors found in dataset")
            return None
        
        return behaviors
        
    except ImportError as e:
        missing_lib = str(e).split()[-1].strip("'")
        if "requests" in missing_lib:
            print("⚠️  requests library not found, install with: pip install requests")
        else:
            print(f"⚠️  Missing library: {missing_lib}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Error downloading from GitHub: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Error parsing CSV from GitHub: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def download_harmbench_with_hub(
    output_path: str,
    split: str = "test",
    hf_token: Optional[str] = None
) -> Optional[List[str]]:
    """
    Download HarmBench dataset using huggingface_hub (alternative method).
    
    Args:
        output_path: Path to save the converted dataset
        split: Dataset split to download ("test", "train", "val")
        hf_token: HuggingFace token (optional)
    
    Returns:
        List of behaviors, or None if failed
    """
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"📥 Downloading HarmBench dataset using huggingface_hub (split: {split})...")
        
        # Try different possible repository names
        repo_candidates = [
            "cais/harmbench",
            "centerforaisafety/harmbench",
            "harmbench/behaviors"
        ]
        
        filename_map = {
            "test": "behaviors.jsonl",
            "train": "behaviors.jsonl",
            "val": "behaviors.jsonl"
        }
        
        filename = filename_map.get(split, "behaviors.jsonl")
        
        # Try each repository
        for repo_id in repo_candidates:
            try:
                print(f"   Trying repository: {repo_id}")
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=hf_token
                )
                
                print(f"✅ Downloaded file: {file_path}")
                
                # Read JSONL file
                behaviors = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            example = json.loads(line.strip())
                            behavior = example.get("behavior", "")
                            if behavior and isinstance(behavior, str):
                                behaviors.append(behavior.strip())
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Warning: Failed to parse line {line_num}: {e}")
                            continue
                
                print(f"✅ Loaded {len(behaviors)} behaviors from JSONL")
                
                if len(behaviors) == 0:
                    print("⚠️  No behaviors found in dataset")
                    continue
                
                return behaviors
                
            except Exception as e:
                print(f"   ⚠️  Failed with {repo_id}: {e}")
                continue
        
        print("⚠️  All HuggingFace repositories failed")
        return None
        
    except ImportError:
        print("⚠️  huggingface_hub library not found")
        return None
    except Exception as e:
        print(f"⚠️  Error with huggingface_hub: {e}")
        return None


def split_behaviors(behaviors: List[str], warm_up_size: int = 50) -> Dict[str, List[str]]:
    """
    Split behaviors into warm_up and lifelong splits.
    
    Args:
        behaviors: List of all behaviors
        warm_up_size: Number of behaviors for warm_up (default: 50)
    
    Returns:
        Dictionary with "warm_up" and "lifelong" keys
    """
    if len(behaviors) < warm_up_size:
        print(f"⚠️  Warning: Only {len(behaviors)} behaviors available, less than warm_up_size ({warm_up_size})")
        warm_up_size = len(behaviors) // 2  # Use half for warm_up
    
    warm_up = behaviors[:warm_up_size]
    lifelong = behaviors[warm_up_size:]
    
    return {
        "warm_up": warm_up,
        "lifelong": lifelong
    }


def save_dataset(data: Dict[str, List[str]], output_path: str):
    """
    Save dataset to JSON file.
    
    Args:
        data: Dictionary with "warm_up" and "lifelong" keys
        output_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved dataset to {output_path}")
    print(f"   - warm_up: {len(data['warm_up'])} requests")
    print(f"   - lifelong: {len(data['lifelong'])} requests")


def download_harmbench_dataset(
    output_path: str,
    split: str = "test",
    hf_token: Optional[str] = None,
    warm_up_size: int = 50,
    eval_format: bool = False
) -> Dict[str, List[str]]:
    """
    Download HarmBench dataset and convert to required format.
    
    Args:
        output_path: Path to save the converted dataset
        split: Dataset split to download ("test", "train", "val", "all")
              "all" uses harmbench_behaviors_text_all.csv (official HarmBench evaluation dataset, default)
        hf_token: HuggingFace token (optional)
        warm_up_size: Number of behaviors for warm_up (default: 50, only used if eval_format=False)
        eval_format: If True, save as simple list for evaluation. If False, save as {"warm_up": [...], "lifelong": [...]}
    
    Returns:
        Dictionary with "warm_up" and "lifelong" keys (if eval_format=False)
        or list of behaviors (if eval_format=True)
    """
    print("=" * 60)
    print("HarmBench Dataset Preparation")
    print("=" * 60)
    
    behaviors = None
    
    # Try GitHub first (official source)
    behaviors = download_harmbench_from_github(output_path, split, hf_token)
    
    # Fallback to datasets library
    if behaviors is None:
        behaviors = download_harmbench_with_datasets(output_path, split, hf_token)
    
    # Fallback to huggingface_hub
    if behaviors is None:
        behaviors = download_harmbench_with_hub(output_path, split, hf_token)
    
    if behaviors is None:
        raise RuntimeError(
            "Failed to download HarmBench dataset from all sources.\n"
            "Tried: GitHub, HuggingFace datasets library, and huggingface_hub.\n"
            "Please check:\n"
            "1. Internet connection\n"
            "2. HarmBench repository: https://github.com/centerforaisafety/HarmBench\n"
            "3. Install dependencies: 'pip install requests datasets huggingface_hub'"
        )
    
    # Save in appropriate format
    if eval_format:
        # Simple list format for evaluation
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(behaviors, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved evaluation dataset to {output_path}")
        print(f"   - Total behaviors: {len(behaviors)}")
        return {"behaviors": behaviors}  # Return dict for consistency
    else:
        # Split into warm_up and lifelong for training
        result = split_behaviors(behaviors, warm_up_size)
        save_dataset(result, output_path)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare HarmBench dataset for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all behaviors (default, uses harmbench_behaviors_text_all.csv)
  python scripts/prepare_harmbench_dataset.py --output ./data/harmbench_dataset.json
  
  # Download test split
  python scripts/prepare_harmbench_dataset.py --output ./data/harmbench_dataset.json --split test
  
  # Download with HuggingFace token
  python scripts/prepare_harmbench_dataset.py \\
    --output ./data/harmbench_dataset.json \\
    --hf_token "hf_..."
  
  # Custom warm_up size
  python scripts/prepare_harmbench_dataset.py \\
    --output ./data/harmbench_dataset.json \\
    --warm_up_size 100
        """
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/harmbench_dataset.json",
        help="Output path for converted dataset (default: ./data/harmbench_dataset.json)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["test", "train", "val", "all"],
        help="Dataset split to download. 'all' uses harmbench_behaviors_text_all.csv (official HarmBench evaluation dataset, default)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (optional, for private datasets or rate limiting)"
    )
    parser.add_argument(
        "--warm_up_size",
        type=int,
        default=50,
        help="Number of behaviors for warm_up split (default: 50, only used if --eval_format is False)"
    )
    parser.add_argument(
        "--eval_format",
        action="store_true",
        help="Save as simple list format for evaluation (no warm_up/lifelong split). Use with: python eval.py --data <file> --split <any>"
    )
    
    args = parser.parse_args()
    
    try:
        result = download_harmbench_dataset(
            output_path=args.output,
            split=args.split,
            hf_token=args.hf_token,
            warm_up_size=args.warm_up_size,
            eval_format=args.eval_format
        )
        
        print("\n" + "=" * 60)
        print("✅ Dataset preparation complete!")
        print("=" * 60)
        
        if args.eval_format:
            # Evaluation format (simple list)
            behaviors = result.get("behaviors", [])
            print(f"\n📊 Dataset Statistics:")
            print(f"   - Total behaviors: {len(behaviors)}")
            print(f"\n📁 Output file: {args.output}")
            print(f"\n💡 Usage for evaluation:")
            print(f"   python eval.py --data {args.output} --split <any> ...")
            print(f"   (Note: --split is ignored when dataset is a simple list)")
        else:
            # Training format (with splits)
            print(f"\n📊 Dataset Statistics:")
            print(f"   - Total behaviors: {len(result['warm_up']) + len(result['lifelong'])}")
            print(f"   - Warm-up split: {len(result['warm_up'])} behaviors")
            print(f"   - Lifelong split: {len(result['lifelong'])} behaviors")
            print(f"\n📁 Output file: {args.output}")
            print(f"\n💡 Usage:")
            print(f"   python main.py --data {args.output} ...")
            print(f"   python eval.py --data {args.output} --split lifelong ...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install required libraries:")
        print("   pip install requests datasets huggingface_hub")
        print("\n2. Check HarmBench repository:")
        print("   https://github.com/centerforaisafety/harmbench")
        print("\n3. Check internet connection")
        print("\n4. If GitHub fails, try HuggingFace token:")
        print("   --hf_token 'hf_...'")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
