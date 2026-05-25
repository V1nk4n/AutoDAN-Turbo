# PRO run config (production)

Chỉnh toàn bộ tham số PRO trong **`scripts/run_pro_production.sh`**.

## Chạy từng giai đoạn (khuyến nghị)

Mỗi lệnh chạy **một giai đoạn** rồi **thoát**. Lifelong **không** chạy lại warm-up (`main.py` bỏ qua warm-up khi có `--only_lifelong N`).

```bash
cd ~/Working/AutoDAN-Turbo

# 1) Warm-up (50 request × EPOCHS repeat) → lưu logs/warm_up_*
bash scripts/run_pro_warmup.sh

# 2) Lifelong chunk 1 (toàn bộ lifelong dataset, 1 pass) → lưu logs/lifelong_*
bash scripts/run_pro_lifelong.sh 1

# 3) Lifelong chunk 2 (load chunk 1, pass tiếp — không warm-up)
bash scripts/run_pro_lifelong.sh 2

# 4) Chunk 3, 4 … (mặc định 4 chunk, khớp --lifelong_iterations trong main.py)
bash scripts/run_pro_lifelong.sh 3
bash scripts/run_pro_lifelong.sh 4
```

Tương đương qua biến `MODE`:

```bash
bash scripts/run_pro_production.sh                              # MODE=warm_up
MODE=lifelong LIFELONG_CHUNK=2 bash scripts/run_pro_production.sh
```

### State giữ giữa các lần chạy

| Sau bước | File quan trọng |
|----------|-----------------|
| warm-up | `logs/warm_up_strategy_library.pkl`, `warm_up_attack_log.json`, `pattern_library.json` |
| lifelong N | `logs/lifelong_strategy_library.pkl`, `lifelong_attack_log.json`, `logs/epoch_refine_memory.json` |

Chunk 1 đọc warm-up; chunk 2+ đọc lifelong trước đó.

### Smoke (test)

```bash
MODE=smoke bash scripts/run_pro_production.sh
# EPOCHS=3, pattern_library_smoke.json, vẫn chỉ warm_up
```

## Chạy nhanh (alias)

| Script | Việc làm |
|--------|----------|
| `run_pro_warmup.sh` | `MODE=warm_up` → `--only_warm_up` |
| `run_pro_lifelong.sh N` | `MODE=lifelong` → `--only_lifelong N` |
| `run_lifelong_fast.sh N` | Delegate → `run_pro_lifelong.sh` |

## Biến quan trọng

| Biến | Mặc định | Ghi chú |
|------|----------|---------|
| `MODE` | `warm_up` | `lifelong` / `smoke` |
| `LIFELONG_CHUNK` | `1` | Chỉ khi `MODE=lifelong` |
| `LIFELONG_TOTAL_CHUNKS` | `4` | Ghi chú; khớp `main.py` |
| `EPOCHS` | `50` (warm_up), `3` (smoke) | Repeat / request |
| `NLL_MIN` / `NLL_MAX` | 2 / 5 | Map NLL → score_loss |
| `PRO_PATTERN_RANK_W_*` | 0.3 / 0.3 / 0.4 | Hoạt động với `four_tier` + embedding |
| `PRO_EARLY_STOP_MIN_DELTA` | 0.015 | ×10 nếu CLI &lt; 1 → effective 0.15 |
| `PRO_USE_FAST_PROFILE` | 0 | 1 = ghi đè nhiều CONFIG |

## Sau mỗi run

```bash
python scripts/pro_smoke_gate.py --latest
python scripts/analyze_pro_thresholds.py --latest --out threshold_report_prod.md
```

## Calib (tune threshold, chậm)

```bash
bash scripts/run_warmup_threshold_calib.sh
```
