# Hướng dẫn chạy từng phần riêng biệt

Tài liệu này hướng dẫn cách chạy warm_up và từng lifelong iteration riêng biệt.

## Tổng quan

Bạn có thể chạy:
1. **Warm-up** riêng (1 lần)
2. **Lifelong iterations** riêng từng lần (5 lần)

Mỗi lần chạy sẽ lưu kết quả vào `logs/` và có thể dừng/tiếp tục bất cứ lúc nào.

---

## Bước 1: Chạy Warm-up

Chạy warm-up một lần và lưu kết quả:

```bash
python main.py \
  --only_warm_up \
  --use_local_embedding \
  --epochs 150 \
  --warm_up_iterations 1 \
  --data ./data/harmful_behavior_requests.json \
  --hf_token "your_hf_token" \
  --log_every 10
```

**Kết quả sẽ được lưu vào:**
- `logs/warm_up_strategy_library.json`
- `logs/warm_up_strategy_library.pkl`
- `logs/warm_up_attack_log.json`
- `logs/warm_up_summarizer_log.json`

---

## Bước 2: Chạy từng Lifelong Iteration

Sau khi warm-up hoàn thành, chạy từng lifelong iteration một:

### Lifelong Iteration 1

```bash
python main.py \
  --only_lifelong 1 \
  --use_local_embedding \
  --epochs 150 \
  --data ./data/harmful_behavior_requests.json \
  --hf_token "your_hf_token" \
  --log_every 10
```

**Kết quả sẽ được lưu vào:**
- `logs/lifelong_strategy_library.json`
- `logs/lifelong_strategy_library.pkl`
- `logs/lifelong_attack_log.json`
- `logs/lifelong_summarizer_log.json`

### Lifelong Iteration 2

```bash
python main.py \
  --only_lifelong 2 \
  --use_local_embedding \
  --epochs 150 \
  --data ./data/harmful_behavior_requests.json \
  --hf_token "your_hf_token" \
  --log_every 10
```

### Lifelong Iteration 3

```bash
python main.py \
  --only_lifelong 3 \
  --use_local_embedding \
  --epochs 150 \
  --data ./data/harmful_behavior_requests.json \
  --hf_token "your_hf_token" \
  --log_every 10
```

### Lifelong Iteration 4

```bash
python main.py \
  --only_lifelong 4 \
  --use_local_embedding \
  --epochs 150 \
  --data ./data/harmful_behavior_requests.json \
  --hf_token "your_hf_token" \
  --log_every 10
```

### Lifelong Iteration 5

```bash
python main.py \
  --only_lifelong 5 \
  --use_local_embedding \
  --epochs 150 \
  --data ./data/harmful_behavior_requests.json \
  --hf_token "your_hf_token" \
  --log_every 10
```

---

## Lưu ý

1. **Phải chạy theo thứ tự**: Warm-up → Iteration 1 → Iteration 2 → ... → Iteration 5
2. **Mỗi iteration sẽ load từ iteration trước**: Iteration 1 load từ warm-up, Iteration 2+ load từ iteration trước đó
3. **Nếu thiếu file**: Script sẽ báo lỗi và hướng dẫn bạn chạy phần nào trước
4. **Log files**: Mỗi lần chạy sẽ tạo log file riêng với timestamp (nếu đã sửa logging code)

---

## Chạy tất cả cùng lúc (backward compatible)

Nếu muốn chạy tất cả trong một lần (như trước), không dùng flags:

```bash
python main.py \
  --use_local_embedding \
  --epochs 150 \
  --warm_up_iterations 1 \
  --lifelong_iterations 5 \
  --data ./data/harmful_behavior_requests.json \
  --hf_token "your_hf_token" \
  --log_every 10
```

---

## Script tự động (tùy chọn)

Bạn có thể tạo script shell để chạy tự động:

```bash
#!/bin/bash
# run_all_phases.sh

HF_TOKEN="your_hf_token"
DATA_FILE="./data/harmful_behavior_requests.json"

# Warm-up
echo "Running warm-up..."
python main.py --only_warm_up --use_local_embedding --epochs 150 --data "$DATA_FILE" --hf_token "$HF_TOKEN" --log_every 10

# Lifelong iterations
for i in {1..5}; do
    echo "Running lifelong iteration $i..."
    python main.py --only_lifelong $i --use_local_embedding --epochs 150 --data "$DATA_FILE" --hf_token "$HF_TOKEN" --log_every 10
done

echo "All phases completed!"
```
