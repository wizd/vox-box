# Vox Box API 客户端调用指南

Vox Box 提供兼容 OpenAI API 的语音服务，包含**语音合成 (TTS)** 和**语音识别 (STT)** 多个独立服务。

## 服务地址

| 服务 | 地址 | 模型 | 功能 |
|------|------|------|------|
| 语音识别 (STT) | `http://localhost:8080` | faster-whisper-large-v3 | 语音转文字 |
| 语音合成 (TTS) - CosyVoice | `http://localhost:8082` | CosyVoice-300M-SFT | 文字转语音（支持流式） |
| 语音合成 (TTS) - Qwen3-TTS | `http://localhost:8083` | Qwen3-TTS-12Hz-1.7B-CustomVoice | 文字转语音（高质量，支持风格控制） |
| 语音合成 (TTS) - CosyVoice3 | `http://localhost:8188` | Fun-CosyVoice3-0.5B | 文字转语音（流式，声色克隆） |
| **语音合成 (TTS) - CosyVoice2 TRT** | **`http://localhost:9880`** | **CosyVoice2-0.5B (TensorRT-LLM)** | **文字转语音（推荐，极致低延迟流式）** |

---

## 一、语音合成 - CosyVoice (Text-to-Speech)

> 端口 `8082` | 支持流式 PCM/WAV | 适合实时对话场景

### `POST /v1/audio/speech`

将文本转换为语音音频文件。

### 请求参数 (JSON Body)

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | string | 是 | - | 模型名称，填 `"cosyvoice"` 即可 |
| `input` | string | 是 | - | 要合成的文本内容 |
| `voice` | string | 是 | - | 语音角色，见下方可用声色表 |
| `response_format` | string | 否 | `"mp3"` | 输出音频格式（`pcm`/`wav` 支持流式） |
| `speed` | float | 否 | `1.0` | 语速，范围 `0.25` ~ `2.0` |

### 可用声色 (voice)

| voice 值 | 说明 |
|----------|------|
| `Chinese Female` | 中文女声 |
| `Chinese Male` | 中文男声 |
| `English Female` | 英文女声 |
| `English Male` | 英文男声 |
| `Japanese Male` | 日语男声 |
| `Cantonese Female` | 粤语女声 |
| `Korean Female` | 韩语女声 |

### 支持的输出格式 (response_format)

| 格式 | MIME 类型 | 流式 |
|------|-----------|------|
| `mp3` | `audio/mpeg` | 否 |
| `wav` | `audio/wav` | 是 |
| `flac` | `audio/x-flac` | 否 |
| `opus` | `audio/ogg;codec=opus` | 否 |
| `aac` | `audio/aac` | 否 |
| `pcm` | `audio/pcm` | 是 |

### 示例

#### cURL

```bash
# 中文女声合成
curl http://localhost:8082/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice",
    "input": "你好，欢迎使用 Vox Box 语音合成服务。",
    "voice": "Chinese Female",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3

# 中文男声 + WAV 格式
curl http://localhost:8082/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice",
    "input": "这是一段测试语音。",
    "voice": "Chinese Male",
    "response_format": "wav"
  }' \
  --output speech.wav

# 英文女声 + 1.5 倍速
curl http://localhost:8082/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice",
    "input": "Hello, welcome to Vox Box text-to-speech service.",
    "voice": "English Female",
    "speed": 1.5
  }' \
  --output speech_fast.mp3
```

#### Python

```python
import requests

response = requests.post(
    "http://localhost:8082/v1/audio/speech",
    json={
        "model": "cosyvoice",
        "input": "你好，这是中文语音合成测试。",
        "voice": "Chinese Female",
        "response_format": "mp3",
        "speed": 1.0,
    },
)

with open("output.mp3", "wb") as f:
    f.write(response.content)
```

#### Python (OpenAI SDK 兼容)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8082/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="cosyvoice",
    input="你好，这是通过 OpenAI SDK 调用的语音合成。",
    voice="Chinese Female",
    response_format="mp3",
    speed=1.0,
)

response.stream_to_file("output.mp3")
```

#### JavaScript / Node.js

```javascript
const fs = require("fs");

async function textToSpeech(text, voice = "Chinese Female") {
  const response = await fetch("http://localhost:8082/v1/audio/speech", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "cosyvoice",
      input: text,
      voice: voice,
      response_format: "mp3",
      speed: 1.0,
    }),
  });

  const buffer = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync("output.mp3", buffer);
}

textToSpeech("你好，世界！");
```

---

## 一-B、语音合成 - Qwen3-TTS (Text-to-Speech)

> 端口 `8083` | 基于 vLLM-Omni | 高质量多语言语音合成，支持风格/情感控制

### `POST /v1/audio/speech`

将文本转换为语音音频文件。API 兼容 OpenAI 格式。

### 请求参数 (JSON Body)

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `input` | string | 是 | - | 要合成的文本内容 |
| `voice` | string | 否 | `"vivian"` | 语音角色，见下方可用声色表 |
| `language` | string | 否 | `"Auto"` | 语言：`Auto`、`Chinese`、`English`、`Japanese`、`Korean` |
| `response_format` | string | 否 | `"wav"` | 输出格式：`wav`、`mp3`、`flac`、`pcm`、`aac`、`opus` |
| `speed` | float | 否 | `1.0` | 语速，范围 `0.25` ~ `4.0` |
| `instructions` | string | 否 | `""` | 风格/情感指令，如 `"用开心的语气说"`、`"Speak with great enthusiasm"` |
| `model` | string | 否 | 自动 | 模型名称（单模型部署时可省略） |

### 可用声色 (voice)

通过 `GET /v1/audio/voices` 获取完整列表。

| voice 值 | 说明 |
|----------|------|
| `vivian` | 女声（默认） |
| `serena` | 女声 |
| `ono_anna` | 女声 |
| `sohee` | 女声 |
| `ryan` | 男声 |
| `aiden` | 男声 |
| `dylan` | 男声 |
| `eric` | 男声 |
| `uncle_fu` | 男声 |

### 支持语言

| language 值 | 语言 |
|-------------|------|
| `Auto` | 自动检测（默认） |
| `Chinese` | 中文 |
| `English` | 英文 |
| `Japanese` | 日语 |
| `Korean` | 韩语 |

### 示例

#### cURL

```bash
# 基础中文女声合成
curl -X POST http://localhost:8083/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好，欢迎使用 Qwen3 语音合成服务。",
    "voice": "vivian",
    "language": "Chinese"
  }' --output qwen3_speech.wav

# 带风格控制 — 开心的语气
curl -X POST http://localhost:8083/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "哇，今天的天气也太好了吧！",
    "voice": "vivian",
    "language": "Chinese",
    "instructions": "用开心的语气说"
  }' --output qwen3_happy.wav

# 英文男声
curl -X POST http://localhost:8083/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a voice synthesis test.",
    "voice": "ryan",
    "language": "English"
  }' --output qwen3_en.wav

# MP3 格式输出
curl -X POST http://localhost:8083/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "这是 MP3 格式的输出测试。",
    "voice": "vivian",
    "response_format": "mp3"
  }' --output qwen3_speech.mp3

# 查询可用声色
curl http://localhost:8083/v1/audio/voices
```

#### Python

```python
import requests

response = requests.post(
    "http://localhost:8083/v1/audio/speech",
    json={
        "input": "你好，这是 Qwen3-TTS 中文语音合成测试。",
        "voice": "vivian",
        "language": "Chinese",
        "response_format": "wav",
    },
    timeout=300,
)

with open("qwen3_output.wav", "wb") as f:
    f.write(response.content)
```

#### Python (OpenAI SDK 兼容)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8083/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    input="你好，这是通过 OpenAI SDK 调用的 Qwen3 语音合成。",
    voice="vivian",
)

response.stream_to_file("qwen3_output.wav")
```

#### JavaScript / Node.js

```javascript
const fs = require("fs");

async function qwen3TTS(text, voice = "vivian") {
  const response = await fetch("http://localhost:8083/v1/audio/speech", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input: text,
      voice: voice,
      language: "Chinese",
      response_format: "wav",
    }),
  });

  const buffer = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync("qwen3_output.wav", buffer);
}

qwen3TTS("你好，世界！");
```

### CosyVoice vs Qwen3-TTS 对比

| 特性 | CosyVoice (8082) | Qwen3-TTS (8083) |
|------|-------------------|-------------------|
| 模型大小 | 300M | 1.7B |
| 流式支持 | 是（PCM/WAV） | 否（全量生成后返回） |
| 风格/情感控制 | 否 | 是（`instructions` 参数） |
| 支持语言 | 中/英/日/粤/韩 | 中/英/日/韩 |
| 语速控制 | 0.25 ~ 2.0 | 0.25 ~ 4.0 |
| 声色数量 | 7 | 9 |
| 适用场景 | 实时对话（低 TTFB） | 离线生成、高质量场景 |

---

## 一-C、语音合成 - CosyVoice3 (Text-to-Speech) [推荐]

> 端口 `8188` | Fun-CosyVoice3-0.5B | 流式 PCM | TTFB ~2.5s | 声色克隆

CosyVoice3 基于阿里巴巴最新的 Fun-CosyVoice3-0.5B 模型，支持流式输出和声色克隆。

### `POST /v1/audio/speech`

将文本转换为语音音频。API 兼容 OpenAI 格式。

### 请求参数 (JSON Body)

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `input` | string | 是 | - | 要合成的文本内容 |
| `voice` | string | 是 | - | 声色 ID（通过 `/v1/voices/create` 创建） |
| `response_format` | string | 否 | `"wav"` | 输出格式：`wav`、`pcm`（流式推荐 `pcm`） |

### 声色管理

CosyVoice3 使用**声色克隆**机制：上传一段参考音频即可创建自定义声色。

```bash
# 创建声色（上传参考音频 + 对应文本）
curl -X POST http://localhost:8188/v1/voices/create \
  -F "audio=@reference_voice.wav" \
  -F "name=MyVoice" \
  -F "text=参考音频的文本内容"

# 自动识别参考音频文本（内置 ASR）
curl -X POST http://localhost:8188/v1/voices/create \
  -F "audio=@reference_voice.wav" \
  -F "name=MyVoice"

# 返回: {"voice_id": "abc123", ...}

# 查询已创建的声色
curl http://localhost:8188/v1/voices/custom

# 删除声色
curl -X DELETE http://localhost:8188/v1/voices/abc123
```

### 支持语言

中文、英文、日语、韩语、德语、西班牙语、法语、意大利语、俄语 + 18 种中国方言（粤语、四川话、东北话、上海话、闽南语等）

### 示例

#### cURL

```bash
# WAV 格式
curl http://localhost:8188/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好，欢迎使用 CosyVoice3 语音合成。", "voice": "YOUR_VOICE_ID"}' \
  -o output.wav

# PCM 流式（最低延迟，推荐 LiveKit 等实时场景）
curl http://localhost:8188/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "voice": "YOUR_VOICE_ID", "response_format": "pcm"}' \
  -o output.pcm

# PCM 转 WAV
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

#### Python

```python
import requests

# 创建声色
with open("reference.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8188/v1/voices/create",
        files={"audio": f},
        data={"name": "MyVoice"}
    )
voice_id = resp.json()["voice_id"]

# 合成语音
resp = requests.post(
    "http://localhost:8188/v1/audio/speech",
    json={"input": "你好，这是 CosyVoice3 测试。", "voice": voice_id}
)
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

#### Python (流式接收)

```python
import requests

resp = requests.post(
    "http://localhost:8188/v1/audio/speech",
    json={
        "input": "你好，这是流式语音合成测试。",
        "voice": "YOUR_VOICE_ID",
        "response_format": "pcm"
    },
    stream=True
)

with open("output.pcm", "wb") as f:
    for chunk in resp.iter_content(chunk_size=4096):
        if chunk:
            f.write(chunk)
            # 实时场景可在此处播放 chunk
```

#### Python (OpenAI SDK 兼容)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8188/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="cosyvoice-v3",
    input="你好，这是通过 OpenAI SDK 调用的 CosyVoice3 语音合成。",
    voice="YOUR_VOICE_ID",
)

response.stream_to_file("output.wav")
```

### 性能基准（RTX 5090, 流式 PCM）

| 文本长度 | TTFB | 总耗时 | 音频时长 | RTF |
|----------|------|--------|----------|-----|
| 短句 4 字 | 4.5s* | 5.2s | 1.3s | 4.1x |
| 短句 10 字 | 2.5s | 11.1s | 6.5s | 1.7x |
| 中句 30 字 | 2.5s | 11.6s | 6.7s | 1.7x |
| 长句 50 字 | 2.4s | 25.5s | 15.7s | 1.6x |

> *首次请求因模型预热较慢，后续请求稳定在 ~2.5s TTFB

### 全方案 TTS 对比总览

| 特性 | CosyVoice v1 (8082) | Qwen3-TTS (8083) | CosyVoice3 (8188) | CosyVoice2 TRT (9880) |
|------|---------------------|-------------------|---------------------|------------------------|
| 模型 | CosyVoice-300M-SFT | Qwen3-TTS-1.7B | Fun-CosyVoice3-0.5B | CosyVoice2-0.5B + TRT-LLM |
| TTFB (流式) | ~5s | 不支持流式 | ~2.5s | **~300-400ms** |
| RTF | ~1.5x | N/A | ~1.7x | **~0.5-0.9x** |
| 流式支持 | 是 | 否 | 是 | **是（gRPC + HTTP）** |
| 声色克隆 | 否 | 否 | 是 | 是（需 gRPC） |
| 风格控制 | 否 | 是 | 否 | 否 |
| 语言 | 中/英/日/粤/韩 | 中/英/日/韩 | 中/英/日/韩 + 18 方言 | 中/英 |
| VRAM | ~4-6 GB | ~16 GB | ~3.2 GB | **~5 GB** |
| 采样率 | 22050 Hz | 24000 Hz | 24000 Hz | 24000 Hz |
| 推荐场景 | 备用 | 离线高质量 | 声色克隆场景 | **实时对话（推荐）** |

---

## 一-D、语音合成 - CosyVoice2 TRT-LLM (Text-to-Speech) [推荐]

> 端口 `9880` | CosyVoice2-0.5B + TensorRT-LLM | 流式 WAV/PCM | TTFB ~300ms | RTF < 1.0

CosyVoice2 TRT-LLM 是目前性能最强的中文 TTS 方案，基于 NVIDIA TensorRT-LLM 加速的 CosyVoice2-0.5B 模型，
通过 Triton Inference Server 提供 gRPC 流式推理，外加 OpenAI 兼容 HTTP Bridge 层。

### 架构

```
Client → HTTP Bridge (9880) → Triton gRPC (8001) → TRT-LLM Engine → 流式音频
```

### `POST /v1/audio/speech`

将文本转换为流式语音音频。API 兼容 OpenAI 格式。

### 请求参数 (JSON Body)

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `input` | string | 是 | - | 要合成的文本内容 |
| `model` | string | 否 | `"cosyvoice2-0.5b"` | 模型名称（可省略） |
| `voice` | string | 否 | `"default"` | 语音角色（当前使用默认说话人） |
| `response_format` | string | 否 | `"wav"` | 输出格式：`wav` 或 `pcm` |

### 示例

#### cURL

```bash
# WAV 格式
curl -X POST http://localhost:9880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好，欢迎使用 CosyVoice2 极速语音合成。"}' \
  -o output.wav

# PCM 格式（最低延迟，适合 LiveKit 等实时场景）
curl -X POST http://localhost:9880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "response_format": "pcm"}' \
  -o output.pcm
```

#### Python

```python
import requests

response = requests.post(
    "http://localhost:9880/v1/audio/speech",
    json={
        "input": "你好，这是 CosyVoice2 TRT-LLM 极速语音合成测试。",
        "response_format": "wav",
    },
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### Python (流式接收)

```python
import requests

resp = requests.post(
    "http://localhost:9880/v1/audio/speech",
    json={
        "input": "你好，这是流式极速语音合成测试。",
        "response_format": "pcm"
    },
    stream=True
)

with open("output.pcm", "wb") as f:
    for chunk in resp.iter_content(chunk_size=4096):
        if chunk:
            f.write(chunk)
            # 实时场景可在此处播放 chunk
```

#### Python (OpenAI SDK 兼容)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9880/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="cosyvoice2-0.5b",
    input="你好，这是通过 OpenAI SDK 调用的 CosyVoice2 极速语音合成。",
    voice="default",
)

response.stream_to_file("output.wav")
```

### 性能基准（RTX 5090, 流式）

| 文本长度 | TTFB | 总耗时 | 音频时长 | RTF |
|----------|------|--------|----------|-----|
| 短句 10 字 | 324ms | 3.9s | 3.9s | 0.98 |
| 中句 16 字 | 315ms | 3.9s | 4.8s | 0.81 |
| 长句 24 字 | 307ms | 4.2s | 7.5s | 0.56 |

> 首次请求因模型预热 TTFB 约 350ms，后续请求稳定在 ~300ms

### 辅助端点

```bash
# 健康检查
curl http://localhost:9880/health

# 查询模型
curl http://localhost:9880/v1/models
```

---

## 二、语音识别 (Speech-to-Text)

### `POST /v1/audio/transcriptions`

将音频文件转录为文本。

### 请求参数 (multipart/form-data)

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | file | 是 | - | 音频文件 |
| `model` | string | 否 | - | 模型名称，填 `"whisper-large-v3"` 即可 |
| `language` | string | 否 | `auto` | 音频语言代码，见下方语言表 |
| `prompt` | string | 否 | - | 提示文本，可引导模型识别特定术语 |
| `temperature` | float | 否 | `0` | 采样温度，范围 `0` ~ `1`，越低越确定 |
| `response_format` | string | 否 | `"json"` | 返回格式 |
| `timestamp_granularities` | string[] | 否 | - | 时间戳粒度 |

### 支持的音频输入格式

`flac`、`mp3`、`mp4`、`mpeg`、`mpga`、`m4a`、`ogg`、`wav`、`webm`

### 支持的返回格式 (response_format)

| 格式 | 说明 |
|------|------|
| `json` | JSON 对象，包含 `text` 字段 |
| `text` | 纯文本 |
| `srt` | SRT 字幕格式 |
| `vtt` | WebVTT 字幕格式 |
| `verbose_json` | 详细 JSON，包含时间戳等信息 |

### 常用语言代码 (language)

| 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|
| `auto` | 自动检测 | `zh` | 中文 |
| `en` | 英语 | `ja` | 日语 |
| `ko` | 韩语 | `fr` | 法语 |
| `de` | 德语 | `es` | 西班牙语 |
| `ru` | 俄语 | `pt` | 葡萄牙语 |
| `it` | 意大利语 | `nl` | 荷兰语 |
| `ar` | 阿拉伯语 | `th` | 泰语 |
| `vi` | 越南语 | `hi` | 印地语 |
| `yue` | 粤语 | `bo` | 藏语 |
| `mn` | 蒙古语 | `my` | 缅甸语 |

> 完整列表支持 99 种语言，可通过 `GET /v1/languages` 端点获取。

### 示例

#### cURL

```bash
# 基础用法 — 自动检测语言
curl http://localhost:8080/v1/audio/transcriptions \
  -F file="@audio.mp3" \
  -F model="whisper-large-v3"

# 指定中文 + JSON 返回
curl http://localhost:8080/v1/audio/transcriptions \
  -F file="@chinese_audio.wav" \
  -F model="whisper-large-v3" \
  -F language="zh" \
  -F response_format="json"

# 生成 SRT 字幕文件
curl http://localhost:8080/v1/audio/transcriptions \
  -F file="@video_audio.mp3" \
  -F model="whisper-large-v3" \
  -F language="zh" \
  -F response_format="srt" \
  --output subtitles.srt

# 使用 prompt 引导识别专业术语
curl http://localhost:8080/v1/audio/transcriptions \
  -F file="@tech_talk.mp3" \
  -F model="whisper-large-v3" \
  -F language="zh" \
  -F prompt="GPUStack, Kubernetes, CosyVoice"
```

#### Python

```python
import requests

with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8080/v1/audio/transcriptions",
        files={"file": ("audio.mp3", f, "audio/mpeg")},
        data={
            "model": "whisper-large-v3",
            "language": "zh",
            "response_format": "json",
        },
    )

print(response.json()["text"])
```

#### Python (OpenAI SDK 兼容)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=f,
        language="zh",
        response_format="text",
    )

print(transcript)
```

#### JavaScript / Node.js

```javascript
const fs = require("fs");
const FormData = require("form-data");

async function transcribe(filePath, language = "zh") {
  const form = new FormData();
  form.append("file", fs.createReadStream(filePath));
  form.append("model", "whisper-large-v3");
  form.append("language", language);
  form.append("response_format", "json");

  const response = await fetch("http://localhost:8080/v1/audio/transcriptions", {
    method: "POST",
    body: form,
  });

  const data = await response.json();
  console.log(data.text);
}

transcribe("audio.mp3");
```

---

## 三、辅助端点

### 健康检查

```bash
# STT 服务
curl http://localhost:8080/health
# TTS - CosyVoice
curl http://localhost:8082/health
# TTS - Qwen3-TTS
curl http://localhost:8083/health
# TTS - CosyVoice3
curl http://localhost:8188/health
# TTS - CosyVoice2 TRT (推荐)
curl http://localhost:9880/health
```

**返回**: `{"status": "ok"}` 或 HTTP 503 (模型加载中)

### 查询模型信息

```bash
# STT 模型列表
curl http://localhost:8080/v1/models
# TTS - CosyVoice 模型列表
curl http://localhost:8082/v1/models
# TTS - Qwen3-TTS 模型列表
curl http://localhost:8083/v1/models
# TTS - CosyVoice3 模型列表
curl http://localhost:8188/v1/models
```

### 查询可用语言 (STT)

```bash
curl http://localhost:8080/v1/languages
```

### 查询可用声色 (TTS)

```bash
# CosyVoice 声色
curl http://localhost:8082/v1/voices
# Qwen3-TTS 声色
curl http://localhost:8083/v1/audio/voices
# CosyVoice3 自定义声色
curl http://localhost:8188/v1/voices/custom
```

---

## 四、完整语言代码参考表

<details>
<summary>点击展开 99 种语言完整列表</summary>

| 代码 | 语言 | 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|------|------|
| `auto` | 自动检测 | `en` | English | `zh` | 中文 |
| `de` | German | `es` | Spanish | `ru` | Russian |
| `ko` | Korean | `fr` | French | `ja` | Japanese |
| `pt` | Portuguese | `pl` | Polish | `ca` | Catalan |
| `nl` | Dutch | `it` | Italian | `th` | Thai |
| `tr` | Turkish | `ar` | Arabic | `sv` | Swedish |
| `id` | Indonesian | `hi` | Hindi | `fi` | Finnish |
| `vi` | Vietnamese | `he` | Hebrew | `uk` | Ukrainian |
| `el` | Greek | `ms` | Malay | `cs` | Czech |
| `ro` | Romanian | `da` | Danish | `hu` | Hungarian |
| `ta` | Tamil | `no` | Norwegian | `ur` | Urdu |
| `hr` | Croatian | `bg` | Bulgarian | `lt` | Lithuanian |
| `la` | Latin | `mi` | Maori | `ml` | Malayalam |
| `cy` | Welsh | `sk` | Slovak | `te` | Telugu |
| `fa` | Persian | `lv` | Latvian | `bn` | Bengali |
| `sr` | Serbian | `az` | Azerbaijani | `sl` | Slovenian |
| `kn` | Kannada | `et` | Estonian | `mk` | Macedonian |
| `br` | Breton | `eu` | Basque | `is` | Icelandic |
| `hy` | Armenian | `ne` | Nepali | `mn` | Mongolian |
| `bs` | Bosnian | `kk` | Kazakh | `sq` | Albanian |
| `sw` | Swahili | `gl` | Galician | `mr` | Marathi |
| `pa` | Punjabi | `si` | Sinhala | `km` | Khmer |
| `sn` | Shona | `yo` | Yoruba | `so` | Somali |
| `af` | Afrikaans | `oc` | Occitan | `ka` | Georgian |
| `be` | Belarusian | `tg` | Tajik | `sd` | Sindhi |
| `gu` | Gujarati | `am` | Amharic | `yi` | Yiddish |
| `lo` | Lao | `uz` | Uzbek | `fo` | Faroese |
| `ht` | Haitian Creole | `ps` | Pashto | `tk` | Turkmen |
| `nn` | Nynorsk | `mt` | Maltese | `sa` | Sanskrit |
| `lb` | Luxembourgish | `my` | Myanmar | `bo` | Tibetan |
| `tl` | Tagalog | `mg` | Malagasy | `as` | Assamese |
| `tt` | Tatar | `haw` | Hawaiian | `ln` | Lingala |
| `ha` | Hausa | `ba` | Bashkir | `jw` | Javanese |
| `su` | Sundanese | `yue` | Cantonese | | |

</details>

---

## 五、错误处理

所有端点在出错时返回 HTTP 错误码和 JSON 详情：

| HTTP 状态码 | 说明 |
|-------------|------|
| `200` | 成功 |
| `400` | 请求参数错误（格式不支持、voice 不存在、speed 越界等） |
| `500` | 服务端内部错误 |
| `503` | 模型尚在加载中 |

**错误返回示例**:

```json
{
  "detail": "Unsupported audio format: xyz"
}
```

---

## 六、注意事项

1. **API Key**: 本地部署无需认证，`api_key` 字段可填任意值（如 `"not-needed"`）
2. **模型加载**: 首次启动需从 HuggingFace 下载模型，请等待 `/health` 返回 `ok` 后再调用
3. **并发**: CosyVoice 使用线程池处理；Qwen3-TTS 基于 vLLM 引擎，当前 `max_num_seqs=1`（顺序处理）
4. **语速**: CosyVoice `speed` 范围 `0.25` ~ `2.0`；Qwen3-TTS `speed` 范围 `0.25` ~ `4.0`
5. **中文识别建议**: STT 可设置 `language=zh` 以提高中文识别准确率，或使用 `auto` 自动检测
6. **Qwen3-TTS 特殊说明**:
   - 首次请求较慢（模型预热），后续请求速度更稳定
   - `instructions` 参数可控制风格/情感（仅 CustomVoice 任务支持）
   - 当前不支持流式输出，音频全量生成后一次性返回
   - `timeout` 建议设置 300 秒以上

## 七、LiveKit Agent 建议配置

### CosyVoice（推荐实时对话场景）

```
response_format = "pcm"  # 流式 PCM，最低 TTFB
sample_rate = 22050       # 见响应头 X-Audio-Sample-Rate
channels = 1              # 单声道
bits_per_sample = 16      # 16-bit signed little-endian
```

### CosyVoice2 TRT（推荐实时对话，最低延迟）

```
base_url = "http://localhost:9880/v1"
response_format = "pcm"   # 流式 PCM，最低 TTFB
sample_rate = 24000       # CosyVoice2 采样率
channels = 1              # 单声道
bits_per_sample = 16      # 16-bit signed little-endian
# TTFB ~300ms, RTF < 1.0
```

### CosyVoice3（声色克隆场景）

```
response_format = "pcm"   # 流式 PCM
sample_rate = 24000       # CosyVoice3 采样率
channels = 1              # 单声道
bits_per_sample = 16      # 16-bit signed little-endian
```

### Qwen3-TTS（推荐高质量离线生成）

```
response_format = "wav"   # WAV 格式
sample_rate = 24000       # Qwen3-TTS 默认采样率
channels = 1              # 单声道
bits_per_sample = 16      # 16-bit signed little-endian
```