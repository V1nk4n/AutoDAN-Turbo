# PRO warm-up — hướng dẫn từng bước

Làm **đúng thứ tự**. Không tune `pro_goal_similarity_floor` / `embed_min_sim` trước khi **Gate 0 PASS**.

---

## Bước 0 — Chuẩn bị

```bash
cd /path/to/AutoDAN-Turbo
# Đảm bảo đang ở branch có fix parser XML + prune + embed 0.28
git status
```

Cần:

- Embedding local (nếu dùng `--use_local_embedding`)
- Target + attacker model đã cấu hình trong `main.py` / env

---

## Bước 1 — Smoke test (3 repeat, 1 request)

Chạy **rất ngắn** để kiểm tra generator:

```bash
python main.py \
  --pro_enabled \
  --only_warm_up \
  --epochs 3 \
  --log_every 1 \
  --use_local_embedding \
  --pattern_filepath ./logs/pattern_library_smoke.json \
  --pro_four_tier_eval \
  --pro_dynamic_pattern_select \
  --pro_per_candidate_strategy_bundles \
  --pro_explore_n_candidates 4 \
  --pro_explore_top_k 2 \
  --pro_explore_max_new_tokens 512 \
  --pro_eval_batch_size 4
```

### Kiểm tra Gate 0

```bash
python scripts/pro_smoke_gate.py --latest
```

**PASS khi:**

- `parser_goal_fallback` = 0 (hoặc <10%)
- `valid_n median` ≥ 2
- `prune n_selected median` ≥ 2 (với `pro_explore_top_k=2`; four_tier dùng floor tuyệt đối 0.15 rồi top-k)

Hoặc đọc `logs/logs_per_run/<run>/running.log`:

```
valid_n=3 hoặc 4
extraction_stage_counts có xml / xml_truncated (không chỉ no_json_braces)
KHÔNG có parser_goal_fallback
```

**Nếu FAIL:** attacker vẫn không ra `<Thought>/<Strategy>/<Response>`. Xem `reject_reason_counts` trong log; cân nhắc model attacker mạnh hơn trước khi tune threshold.

---

## Bước 2 — Warm-up calibration (15 epochs / request)

Chỉ chạy khi Bước 1 PASS.

```bash
bash scripts/run_warmup_threshold_calib.sh
```

Hoặc lệnh đầy đủ (tương đương script):

```bash
PATTERN_LIB=./logs/pattern_library_calib_v4.json EPOCHS=15 \
bash scripts/run_warmup_threshold_calib.sh
```

**Lưu ý config:**

| Flag | Ý nghĩa |
|------|---------|
| `pro_explore_n_candidates` / `pro_exploit_n_candidates` | Số probe **mỗi repeat theo phase** |
| `pro_n_candidates` | Chỉ dùng cho bundle pattern; **không** thay explore/exploit |
| `pro_phase_split 0.30` | Exploit từ ~repeat 5/15 |
| `pro_early_stop_patience 8` | Tránh dừng sớm trước exploit |
| `pro_early_stop_min_delta 0.05` | Effective **0.5** trên thang score_loss 0–10 |

---

## Bước 3 — Phân tích báo cáo

```bash
python scripts/analyze_pro_thresholds.py --latest --out threshold_report_calib.md
```

Mở report, đọc **theo thứ tự:**

1. **Calibration gates** — cả 3 Gate PASS → mới tin threshold section
2. **Health checks** — tier1 `regex_refusal` vs `short_response`
3. **Strategy embedding** — chọn `embed_min_sim` sao cho `n_match==1` cao
4. **Goal similarity prune** — dùng `goal_response_sim`, không `goal_prompt_sim`

```bash
# Gate nhanh sau warm-up
python scripts/pro_smoke_gate.py --latest
```

---

## Bước 4 — Freeze config → lifelong

Khi Gate 0–2 PASS và đã chọn threshold từ report:

1. Lưu `pattern_library` từ warm-up
2. Ghi lại CLI đã chọn vào file notes / script
3. Chạy lifelong từng chunk:

```bash
python main.py --pro_enabled --only_lifelong 1 \
  --pattern_filepath ./logs/pattern_library_calib_v4.json \
  ... # cùng các flag PRO đã freeze
```

---

## Config map (tránh nhầm)

```
CLI ProPipelineConfig
    → pipeline __init__ (baseline_cli trong telemetry)
    → mỗi repeat: explore HOẶC exploit override n_candidates, top_k, max_new_tokens
    → generate (attacker XML) → semantic_prune → four_tier eval (target)
```

---

## Sửa lỗi đã merge (branch hiện tại)

1. **Parser:** JSON fail (`no_json_braces`) vẫn parse XML `<Response>` (bug cũ chặn 100% repeat).
2. **Fallback:** Không inject raw goal; repeat skip nếu không có structured output.
3. **Prune:** top-k + floor tương đối; bỏ candidate trùng verbatim goal.
4. **Attribution:** embed 0.28 + margin top-1.
5. **Telemetry:** `baseline_cli`, `generation_summary`, gates trong analyzer.

---

## Khi nào dừng và debug

| Triệu chứng | Hành động |
|-------------|-----------|
| `valid_n=1`, `parser_goal_fallback` | Gate 0 FAIL — sửa attacker, không tune threshold |
| `goal_sim=1.0` toàn run | Cùng nguyên nhân fallback / prompt=goal |
| 100% `phase=explore` | Giảm `pro_phase_split` hoặc tăng `patience` |
| 90% tier1 `regex_refusal` | Target an toàn — đổi target hoặc chỉnh regex refusal |
| `n_match>=2` attribution cao | Tăng `embed_min_sim` / margin (đã default 0.28/0.05) |
