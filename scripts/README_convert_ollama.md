# Convert HarmBench Model to Ollama Format

Hướng dẫn convert model `cais/HarmBench-Mistral-7b-val-cls` từ HuggingFace sang Ollama format.

## Phương pháp 1: Sử dụng llama.cpp (Khuyến nghị)

### Bước 1: Cài đặt llama.cpp và dependencies

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Cài đặt Python dependencies cho conversion script
pip install transformers torch huggingface_hub

# Build với CMake (yêu cầu CMake >= 3.13)
mkdir build
cd build
cmake ..
cmake --build . --config Release

# Hoặc đơn giản hơn: chỉ cần Python dependencies để convert
# (không cần build nếu chỉ dùng để convert)
```

### Bước 2: Download model từ HuggingFace

```bash
# Sử dụng huggingface-cli
pip install huggingface_hub
huggingface-cli download cais/HarmBench-Mistral-7b-val-cls --local-dir ./harmbench-model
```

### Bước 3: Convert sang GGUF

```bash
cd llama.cpp
# Sử dụng script convert_hf_to_gguf.py (có sẵn trong llama.cpp)
python convert_hf_to_gguf.py ../harmbench-model --outfile ../harmbench-mistral-7b.gguf --outtype f16

# Hoặc với quantization (tiết kiệm VRAM):
python convert_hf_to_gguf.py ../harmbench-model --outfile ../harmbench-mistral-7b-q4_0.gguf --outtype q4_0
```

### Bước 4: Import vào Ollama

Tạo file `Modelfile`:

```dockerfile
FROM ./harmbench-mistral-7b.gguf
TEMPLATE """[INST] <<SYS>>
{{ .System }}
<</SYS>>

{{ .Prompt }} [/INST]
"""
PARAMETER temperature 0.0
PARAMETER top_p 0.9
```

Sau đó import:

```bash
ollama create harmbench-mistral-7b -f Modelfile
```

## Phương pháp 2: Sử dụng script tự động

```bash
python scripts/convert_harmbench_to_ollama.py \
  --model_name cais/HarmBench-Mistral-7b-val-cls \
  --ollama_model_name harmbench-mistral-7b \
  --hf_token <your_hf_token>
```

**Lưu ý**: Script này cần llama.cpp hoặc llama-cpp-python đã được cài đặt.

## Phương pháp 3: Sử dụng Ollama trực tiếp (nếu hỗ trợ)

Một số phiên bản Ollama mới có thể import trực tiếp từ HuggingFace:

```bash
# Thử import trực tiếp (có thể không hoạt động với fine-tuned models)
ollama import cais/HarmBench-Mistral-7b-val-cls harmbench-mistral-7b
```

## Kiểm tra model sau khi import

```bash
# Test model
ollama run harmbench-mistral-7b "Hello, are you a classifier?"

# List models
ollama list
```

## Sử dụng trong eval.py

Sau khi import thành công, bạn có thể sử dụng:

```bash
python eval.py \
  --use_harmbench_classifier \
  --use_ollama \
  --ollama_model_name "harmbench-mistral-7b" \
  # ... other args
```

## Troubleshooting

### Lỗi: Model quá lớn
- Sử dụng quantization khi convert: `--outtype q4_0` hoặc `--outtype q8_0`
- Ví dụ: `python convert.py ... --outtype q4_0`

### Lỗi: Tokenizer không tương thích
- Model HarmBench sử dụng Mistral tokenizer, nên cần đảm bảo template đúng
- Kiểm tra chat template của Mistral trong Ollama

### Lỗi: Conversion failed
- Đảm bảo model đã download đầy đủ
- Kiểm tra phiên bản llama.cpp
- Thử convert với các options khác nhau

## Tài liệu tham khảo

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Ollama Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
