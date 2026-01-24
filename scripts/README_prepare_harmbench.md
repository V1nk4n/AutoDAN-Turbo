# Prepare HarmBench Dataset

Hướng dẫn download và chuẩn bị HarmBench dataset chính thức cho quá trình đánh giá.

**Nguồn chính thức**: https://github.com/centerforaisafety/HarmBench

## Cài đặt Dependencies

```bash
pip install requests
```

Hoặc cài từ requirements.txt:
```bash
pip install -r requirements.txt
```

**Lưu ý**: Script tự động download từ GitHub, không cần HuggingFace token.

## Sử dụng

### Format cho Evaluation (khuyến nghị cho eval.py)

Tạo file đơn giản (list behaviors) cho evaluation:
**Lưu ý**: Default sử dụng `harmbench_behaviors_text_all.csv` (file chính thức của HarmBench)

```bash
python scripts/prepare_harmbench_dataset.py \
  --output ./data/harmbench_eval.json \
  --split all \
  --eval_format
```

Hoặc không cần chỉ định `--split` (mặc định là `all`):
```bash
python scripts/prepare_harmbench_dataset.py \
  --output ./data/harmbench_eval.json \
  --eval_format
```

**Format output**: `["behavior1", "behavior2", ...]` (list trực tiếp)

**Sử dụng trong eval.py**:
```bash
python eval.py \
  --data ./data/harmbench_eval.json \
  --split <any> \
  --use_harmbench_classifier \
  ...
```

**Lưu ý**: Khi dùng `--eval_format`, `--split` trong `eval.py` sẽ bị ignore vì dataset là list trực tiếp.

### Format cho Training (với warm_up/lifelong splits)

Tạo file với splits cho training:
**Lưu ý**: Default sử dụng `harmbench_behaviors_text_all.csv` (file chính thức của HarmBench)

```bash
python scripts/prepare_harmbench_dataset.py \
  --output ./data/harmbench_dataset.json \
  --split all \
  --warm_up_size 50
```

Hoặc không cần chỉ định `--split` (mặc định là `all`):
```bash
python scripts/prepare_harmbench_dataset.py \
  --output ./data/harmbench_dataset.json \
  --warm_up_size 50
```

**Format output**: `{"warm_up": [...], "lifelong": [...]}`

**Sử dụng trong main.py**:
```bash
python main.py \
  --data ./data/harmbench_dataset.json \
  ...
```

**Sử dụng trong eval.py**:
```bash
python eval.py \
  --data ./data/harmbench_dataset.json \
  --split lifelong \
  ...
```
```

### Tùy chỉnh warm_up size

```bash
python scripts/prepare_harmbench_dataset.py \
  --output ./data/harmbench_dataset.json \
  --warm_up_size 100
```

### Download split khác

```bash
# Download all behaviors (default, official HarmBench evaluation dataset)
python scripts/prepare_harmbench_dataset.py \
  --output ./data/harmbench_all.json \
  --split all

# Download test split
python scripts/prepare_harmbench_dataset.py \
  --output ./data/harmbench_test.json \
  --split test

# Download validation split
python scripts/prepare_harmbench_dataset.py \
  --output ./data/harmbench_val.json \
  --split val
```

## Output Format

Script sẽ tạo file JSON với format:

```json
{
  "warm_up": [
    "Behavior 1",
    "Behavior 2",
    ...
    "Behavior 50"
  ],
  "lifelong": [
    "Behavior 51",
    "Behavior 52",
    ...
    "Behavior N"
  ]
}
```

## Sử dụng Dataset

### Trong main.py (training)

```bash
python main.py \
  --data ./data/harmbench_dataset.json \
  --use_local_embedding \
  --epochs 150 \
  --warm_up_iterations 1 \
  --lifelong_iterations 5 \
  --hf_token "hf_..."
```

### Trong eval.py (evaluation)

```bash
python eval.py \
  --data ./data/harmbench_dataset.json \
  --split lifelong \
  --use_harmbench_classifier \
  --harmbench_classifier_model "cais/HarmBench-Mistral-7b-val-cls" \
  --harmbench_quantization_type "4bit" \
  --use_local_embedding \
  --hf_token "hf_..." \
  --epochs 150 \
  --strategy_library_pkl ./logs/lifelong_strategy_library.pkl \
  --save_json ./logs/eval_results_harmbench.json
```

## Kiểm tra Dataset

```bash
# Xem số lượng requests
python -c "import json; data = json.load(open('./data/harmbench_dataset.json')); print(f'warm_up: {len(data[\"warm_up\"])}, lifelong: {len(data[\"lifelong\"])}')"

# Xem một vài examples
python -c "import json; data = json.load(open('./data/harmbench_dataset.json')); print('warm_up[0]:', data['warm_up'][0]); print('lifelong[0]:', data['lifelong'][0])"
```

## Troubleshooting

### Lỗi: ModuleNotFoundError: datasets

```bash
pip install datasets
```

### Lỗi: ModuleNotFoundError: huggingface_hub

```bash
pip install huggingface_hub
```

### Lỗi: Authentication required

HarmBench dataset có thể yêu cầu HuggingFace token. Thêm `--hf_token`:

```bash
python scripts/prepare_harmbench_dataset.py \
  --output ./data/harmbench_dataset.json \
  --hf_token "hf_..."
```

### Lỗi: Rate limit

Nếu gặp rate limit từ HuggingFace, thử:
1. Đăng nhập HuggingFace CLI: `huggingface-cli login`
2. Hoặc dùng token: `--hf_token "hf_..."`

## Tham khảo

- HarmBench Dataset: https://huggingface.co/datasets/cais/harmbench
- HarmBench Paper: https://arxiv.org/abs/2402.04249
