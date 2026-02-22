# CosyVoice2 标点缓冲灰度与回滚策略

## 灰度开关

- `PUNCT_BUFFER_ENABLED`：是否启用标点边界缓冲（默认 `false`）
- `PUNCT_MIN_CHARS`：软边界最小字符阈值（默认 `12`）
- `PUNCT_MAX_CHARS`：强制切分最大字符阈值（默认 `80`）
- `PUNCT_MAX_WAIT_MS`：未来流式输入场景的超时阈值（默认 `400`，当前整句输入用于配置对齐）
- `OUTPUT_JITTER_BUFFER_MS`：出站平滑缓存（默认 `0`）

## 灰度流程

1. **Phase A（5%）**
   - 将 `PUNCT_BUFFER_ENABLED=true` 仅放到 5% 流量副本。
   - 运行 `python test_cosyvoice2_trt.py`，观察 `p95-gap` 与 `max-gap`。
2. **Phase B（20%）**
   - 若连续 24 小时未出现丢字，提升到 20%。
3. **Phase C（50%）**
   - 继续观察 24 小时，重点检查长句与多标点文本。
4. **Phase D（100%）**
   - 无回归后全量启用。

## 回滚策略

- 立即回滚（推荐）：`PUNCT_BUFFER_ENABLED=false`
- 若仍有异常，额外恢复：
  - `OUTPUT_JITTER_BUFFER_MS=0`
  - `docker compose up -d --force-recreate cosyvoice2-bridge`

## 判定门禁

- 不得出现“丢字/截词/重复片段”。
- `p95-gap <= 700ms`
- `max-gap <= 1200ms`
