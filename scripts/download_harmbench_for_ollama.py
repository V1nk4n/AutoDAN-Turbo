#!/usr/bin/env python3
"""
Simple script to download HarmBench model for Ollama conversion.

This script downloads the model and provides instructions for conversion.
"""

import os
import argparse
from pathlib import Path


def download_model(model_name: str, output_dir: str, hf_token: str = None):
    """Download model from HuggingFace."""
    print(f"📥 Downloading model: {model_name}")
    print(f"   Output directory: {output_dir}")
    
    try:
        from huggingface_hub import snapshot_download
        
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            token=hf_token,
            local_files_only=False
        )
        
        print(f"✅ Model downloaded successfully!")
        print(f"   Location: {model_path}")
        return model_path
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("   Install with: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return None


def print_conversion_instructions(model_path: str, ollama_name: str):
    """Print instructions for converting to Ollama."""
    print("\n" + "=" * 60)
    print("📋 Next Steps: Convert to Ollama Format")
    print("=" * 60)
    
    print("\n1️⃣  Install llama.cpp:")
    print("   git clone https://github.com/ggerganov/llama.cpp.git")
    print("   cd llama.cpp")
    print("   mkdir build && cd build")
    print("   cmake ..")
    print("   cmake --build . --config Release")
    print("   # Hoặc đơn giản hơn: pip install llama-cpp-python")
    
    print("\n2️⃣  Convert to GGUF:")
    print(f"   cd llama.cpp")
    print(f"   python convert_hf_to_gguf.py {model_path} --outfile ../{ollama_name}.gguf --outtype q4_0")
    print("   (Use q4_0 for 4-bit quantization, f16 for full precision)")
    print("   (Script convert_hf_to_gguf.py có sẵn trong llama.cpp)")
    
    print("\n3️⃣  Create Modelfile:")
    print(f"   echo 'FROM ./{ollama_name}.gguf' > Modelfile")
    print("   echo 'TEMPLATE \"\"\"[INST] <<SYS>>' >> Modelfile")
    print("   echo '{{{{ .System }}}}' >> Modelfile")
    print("   echo '<</SYS>>' >> Modelfile")
    print("   echo '' >> Modelfile")
    print("   echo '{{{{ .Prompt }}}} [/INST]' >> Modelfile")
    print("   echo '\"\"\"' >> Modelfile")
    print("   echo 'PARAMETER temperature 0.0' >> Modelfile")
    print("   echo 'PARAMETER top_p 0.9' >> Modelfile")
    
    print("\n4️⃣  Import to Ollama:")
    print(f"   ollama create {ollama_name} -f Modelfile")
    
    print("\n5️⃣  Test:")
    print(f"   ollama run {ollama_name} \"Hello\"")
    
    print("\n" + "=" * 60)
    print("💡 Alternative: Use Python llama-cpp-python")
    print("=" * 60)
    print("   pip install llama-cpp-python")
    print("   # Then use convert_harmbench_to_ollama.py")


def main():
    parser = argparse.ArgumentParser(
        description="Download HarmBench model for Ollama conversion"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="cais/HarmBench-Mistral-7b-val-cls",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/harmbench-mistral-7b",
        help="Directory to save downloaded model"
    )
    parser.add_argument(
        "--ollama_name",
        type=str,
        default="harmbench-mistral-7b",
        help="Name for the model in Ollama"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (optional, for private models)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HarmBench Model Downloader for Ollama")
    print("=" * 60)
    
    # Download model
    model_path = download_model(args.model_name, args.output_dir, args.hf_token)
    
    if model_path:
        print_conversion_instructions(model_path, args.ollama_name)
    else:
        print("\n❌ Download failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
