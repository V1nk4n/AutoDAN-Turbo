# Quick Start: Convert HarmBench Model to Ollama

Hướng dẫn nhanh để convert model HarmBench sang Ollama format.

## Bước 1: Cài đặt dependencies

```bash
# Đảm bảo đã có Python environment
pip install transformers torch huggingface_hub
```

## Bước 2: Download model

```bash
python scripts/download_harmbench_for_ollama.py \
  --model_name cais/HarmBench-Mistral-7b-val-cls \
  --output_dir ./models/harmbench-mistral-7b \
  --ollama_name harmbench-mistral-7b \
  --hf_token <your_hf_token>
```

## Bước 3: Clone llama.cpp (nếu chưa có)

```bash
git clone https://github.com/ggerganov/llama.cpp.git
```

## Bước 4: Convert sang GGUF

```bash
cd llama.cpp
python convert_hf_to_gguf.py \
  ../models/harmbench-mistral-7b \
  --outfile ../harmbench-mistral-7b-q4_0.gguf \
  --outtype q4_0
```

**Lưu ý**: 
- `q4_0` = 4-bit quantization (tiết kiệm VRAM, ~4GB)
- `f16` = full precision (chính xác hơn, ~14GB)

## Bước 5: Tạo Modelfile

Tạo file `Modelfile` trong thư mục chứa file `.gguf`:

```bash
cat > Modelfile << 'EOF'
FROM ./harmbench-mistral-7b-q4_0.gguf
TEMPLATE """[INST] <<SYS>>
{{ .System }}
<</SYS>>

{{ .Prompt }} [/INST]
"""
PARAMETER temperature 0.0
PARAMETER top_p 0.9
EOF
```

## Bước 6: Import vào Ollama

```bash
ollama create harmbench-mistral-7b -f Modelfile
```

## Bước 7: Test

```bash
ollama run harmbench-mistral-7b "Hello, are you a classifier?"
```

## Sử dụng trong eval.py

```bash
python eval.py \
  --use_harmbench_classifier \
  --use_ollama \
  --ollama_model_name "harmbench-mistral-7b" \
  --use_local_embedding \
  --hf_token "hf_..." \
  --epochs 150 \
  --split lifelong \
  --strategy_library_pkl ./logs/lifelong_strategy_library.pkl \
  --data ./data/harmful_behavior_requests.json \
  --save_json ./logs/eval_results_harmbench_ollama.json
```

## Troubleshooting

### Lỗi: ModuleNotFoundError: transformers
```bash
pip install transformers torch huggingface_hub
```

### Lỗi: Model quá lớn
- Sử dụng quantization: `--outtype q4_0` hoặc `q8_0`
- Hoặc `q4_1`, `q5_0`, `q5_1` cho các mức quantization khác

### Lỗi: Conversion failed
- Kiểm tra model đã download đầy đủ
- Đảm bảo có đủ disk space (cần ~20GB cho quá trình convert)

### Lỗi: Ollama import failed
- Kiểm tra đường dẫn đến file .gguf trong Modelfile
- Đảm bảo Ollama đang chạy: `ollama serve`
