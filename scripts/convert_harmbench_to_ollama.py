#!/usr/bin/env python3
"""
Script to convert HarmBench classifier model from HuggingFace to Ollama format.

This script downloads the HarmBench model and converts it to GGUF format,
then imports it into Ollama.

Requirements:
- llama-cpp-python (for conversion)
- Or use llama.cpp directly
- Ollama installed and running
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    # Check for llama-cpp-python or llama.cpp
    try:
        import llama_cpp
    except ImportError:
        # Check if llama.cpp binary exists
        result = subprocess.run(["which", "llama-cpp-convert"], capture_output=True)
        if result.returncode != 0:
            missing.append("llama-cpp-python or llama.cpp")
    
    # Check for Ollama
    result = subprocess.run(["which", "ollama"], capture_output=True)
    if result.returncode != 0:
        missing.append("ollama")
    
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nInstallation:")
        print("   - Ollama: curl -fsSL https://ollama.com/install.sh | sh")
        print("   - llama.cpp: https://github.com/ggerganov/llama.cpp")
        print("   - Or: pip install llama-cpp-python")
        return False
    
    return True


def download_model(model_name: str, output_dir: str, hf_token: str = None):
    """Download model from HuggingFace."""
    print(f"📥 Downloading model: {model_name}")
    
    from huggingface_hub import snapshot_download
    
    try:
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=output_dir,
            token=hf_token,
            local_files_only=False
        )
        print(f"✅ Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return None


def convert_to_gguf(model_path: str, output_path: str):
    """Convert model to GGUF format using llama.cpp."""
    print(f"🔄 Converting model to GGUF format...")
    
    # Try using llama-cpp-convert if available
    convert_cmd = [
        "llama-cpp-convert",
        model_path,
        output_path
    ]
    
    try:
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Model converted to: {output_path}")
            return True
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("⚠️  llama-cpp-convert not found. Trying alternative method...")
        # Alternative: use Python llama-cpp-python
        try:
            from llama_cpp import Llama
            
            # This is a simplified approach - actual conversion may need more steps
            print("⚠️  Direct conversion via llama-cpp-python may not work for all models.")
            print("    Consider using llama.cpp binary directly.")
            return False
        except ImportError:
            print("❌ llama-cpp-python not installed")
            return False


def import_to_ollama(gguf_path: str, ollama_model_name: str):
    """Import GGUF model into Ollama."""
    print(f"📦 Importing model into Ollama as: {ollama_model_name}")
    
    # Create Modelfile
    modelfile_content = f"""FROM {gguf_path}
TEMPLATE \"\"\"[INST] <<SYS>>
{{{{ .System }}}}
<</SYS>>

{{{{ .Prompt }}}} [/INST]
\"\"\"
PARAMETER temperature 0.0
PARAMETER top_p 0.9
"""
    
    modelfile_path = "/tmp/harmbench_modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    # Import using ollama create
    cmd = ["ollama", "create", ollama_model_name, "-f", modelfile_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Model imported to Ollama as: {ollama_model_name}")
            print(f"   You can now use it with: ollama run {ollama_model_name}")
            return True
        else:
            print(f"❌ Failed to import to Ollama: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error importing to Ollama: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert HarmBench model from HuggingFace to Ollama format"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="cais/HarmBench-Mistral-7b-val-cls",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--ollama_model_name",
        type=str,
        default="harmbench-mistral-7b",
        help="Name for the model in Ollama"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save downloaded model"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (if model is private)"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download if model already exists"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HarmBench Model to Ollama Converter")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Download model
    model_path = None
    if not args.skip_download:
        model_path = download_model(args.model_name, args.output_dir, args.hf_token)
        if not model_path:
            sys.exit(1)
    else:
            print(f"✅ Using model at: {model_path}")
    
    # Step 2: Convert to GGUF
    gguf_path = os.path.join(args.output_dir, f"{args.ollama_model_name}.gguf")
    
    # Note: Actual conversion requires llama.cpp or similar tools
    # This is a placeholder - you may need to use llama.cpp directly
    print("\n⚠️  Note: GGUF conversion requires llama.cpp")
    print("   Steps to convert manually:")
    print("   1. Install llama.cpp: https://github.com/ggerganov/llama.cpp")
    print("   2. Run: python convert.py <model_path> --outfile <output.gguf>")
    print("   3. Then use: ollama create <name> -f <modelfile>")
    
    # Alternative: Use Ollama's import feature if available
    print("\n💡 Alternative: Use Ollama's built-in HuggingFace import")
    print("   (if supported in your Ollama version)")
    
    # Step 3: Import to Ollama (if GGUF exists)
    if os.path.exists(gguf_path):
        import_to_ollama(gguf_path, args.ollama_model_name)
    else:
        print("\n📝 Manual steps:")
        print(f"   1. Convert model to GGUF format")
        print(f"   2. Create Modelfile with: FROM {gguf_path}")
        print(f"   3. Run: ollama create {args.ollama_model_name} -f <modelfile>")


if __name__ == "__main__":
    main()
