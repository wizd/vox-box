# Vox Box API 客户端调用指南

Vox Box 提供兼容 OpenAI API 的语音服务，包含**语音合成 (TTS)** 和**语音识别 (STT)** 两个独立服务。

## 服务地址

| 服务 | 地址 | 模型 | 功能 |
|------|------|------|------|
| 语音识别 (STT) | `http://localhost:8080` | faster-whisper-large-v3 | 语音转文字 |
| 语音合成 (TTS) | `http://localhost:8082` | CosyVoice-300M-SFT | 文字转语音 |

---

## 一、语音合成 (Text-to-Speech)

### `POST /v1/audio/speech`

将文本转换为语音音频文件。

### 请求参数 (JSON Body)

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | string | 是 | - | 模型名称，填 `"cosyvoice"` 即可 |
| `input` | string | 是 | - | 要合成的文本内容 |
| `voice` | string | 是 | - | 语音角色，见下方可用声色表 |
| `response_format` | string | 否 | `"mp3"` | 输出音频格式 |
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

| 格式 | MIME 类型 |
|------|-----------|
| `mp3` | `audio/mpeg` |
| `wav` | `audio/wav` |
| `flac` | `audio/x-flac` |
| `opus` | `audio/ogg;codec=opus` |
| `aac` | `audio/aac` |
| `pcm` | `audio/pcm` |

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
# TTS 服务
curl http://localhost:8082/health
```

**返回**: `{"status": "ok"}` 或 HTTP 503 (模型加载中)

### 查询模型信息

```bash
# STT 模型列表
curl http://localhost:8080/v1/models

# TTS 模型列表
curl http://localhost:8082/v1/models
```

### 查询可用语言 (STT)

```bash
curl http://localhost:8080/v1/languages
```

### 查询可用声色 (TTS)

```bash
curl http://localhost:8082/v1/voices
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
3. **并发**: 服务使用线程池处理请求，支持并发调用
4. **语速**: TTS `speed` 参数范围为 `0.25`（慢速）到 `2.0`（快速），默认 `1.0`
5. **中文识别建议**: STT 可设置 `language=zh` 以提高中文识别准确率，或使用 `auto` 自动检测


# LiveKit Agent 建议配置
response_format = "pcm"  # 流式 PCM，最低 TTFB
sample_rate = 22050       # 见响应头 X-Audio-Sample-Rate
channels = 1              # 单声道
bits_per_sample = 16      # 16-bit signed little-endian