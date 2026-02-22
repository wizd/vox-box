#!/usr/bin/env python3
"""
CosyVoice2 TRT-LLM OpenAI-Compatible HTTP Bridge

将 OpenAI /v1/audio/speech API 请求转发到 Triton CosyVoice2 gRPC 流式推理，
返回流式音频 (PCM 16-bit, 24kHz)。

用法:
    python cosyvoice2_bridge.py --triton-url localhost:8001 --port 9880
"""

import argparse
import asyncio
import io
import os
import queue
import re
import struct
import time
import logging
from uuid import uuid4

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cosyvoice2-bridge")

app = FastAPI(title="CosyVoice2 TRT-LLM OpenAI Bridge")

TRITON_URL = "localhost:8001"
MODEL_NAME = "cosyvoice2"
SAMPLE_RATE = 24000
BITS_PER_SAMPLE = 16
NUM_CHANNELS = 1


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Invalid int env {name}={raw!r}, using default={default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


PUNCT_BUFFER_ENABLED = _env_bool("PUNCT_BUFFER_ENABLED", False)
PUNCT_MIN_CHARS = _env_int("PUNCT_MIN_CHARS", 12)
PUNCT_MAX_CHARS = _env_int("PUNCT_MAX_CHARS", 80)
PUNCT_MAX_WAIT_MS = _env_int("PUNCT_MAX_WAIT_MS", 400)
OUTPUT_JITTER_BUFFER_MS = _env_int("OUTPUT_JITTER_BUFFER_MS", 0)
SESSION_SERIALIZE_ENABLED = _env_bool("SESSION_SERIALIZE_ENABLED", True)
SESSION_LOCK_TIMEOUT_S = _env_int("SESSION_LOCK_TIMEOUT_S", 120)

HARD_BOUNDARY_RE = re.compile(r"[。！？；.!?;\n]")
SOFT_BOUNDARY_RE = re.compile(r"[，,]")

_session_locks: dict[str, asyncio.Lock] = {}
_session_locks_guard = asyncio.Lock()


def _resolve_session_id(request: Request) -> str:
    header_sid = (
        request.headers.get("x-session-id")
        or request.headers.get("x-client-session-id")
        or request.headers.get("x-request-session-id")
    )
    if header_sid:
        return header_sid.strip()
    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")
    # Fallback to connection identity so requests from same client serialize.
    return f"fallback:{client_host}:{user_agent}"


async def _get_or_create_session_lock(session_id: str) -> asyncio.Lock:
    async with _session_locks_guard:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            _session_locks[session_id] = lock
        return lock


def create_wav_header(sample_rate=24000, bits_per_sample=16, num_channels=1, data_size=0):
    """Create a WAV file header."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    # Use 0xFFFFFFFF for streaming (unknown size)
    if data_size == 0:
        data_size = 0xFFFFFFFF - 36
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        data_size + 36,  # file size - 8
        b'WAVE',
        b'fmt ',
        16,  # fmt chunk size
        1,   # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size,
    )
    return header


class StreamCollector:
    """Collects streaming responses from Triton."""
    def __init__(self):
        self.queue = queue.Queue()
        self.start_time = None
        self.first_chunk_time = None
        self.error = None

    def callback(self, result, error):
        if error:
            self.error = error
            self.queue.put(None)
            return
        now = time.time()
        try:
            output = result.as_numpy("waveform")
        except Exception:
            # Final empty response has no waveform tensor
            self.queue.put(None)
            return
        if output is not None and len(output) > 0:
            if self.first_chunk_time is None:
                self.first_chunk_time = now
            self.queue.put(output.flatten())
        else:
            # Empty response = final response from decoupled model
            self.queue.put(None)


FIRST_CHUNK_TIMEOUT = _env_int("FIRST_CHUNK_TIMEOUT_S", 30)   # first chunk may be slow (model warm-up)
NEXT_CHUNK_TIMEOUT = _env_int("NEXT_CHUNK_TIMEOUT_S", 3)      # keep tail wait bounded to reduce long pauses


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    low = int(k)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def _normalize_input_text(raw_input: object) -> list[str]:
    if isinstance(raw_input, str):
        return [raw_input]
    if isinstance(raw_input, list):
        chunks = [str(item) for item in raw_input if str(item)]
        return chunks if chunks else [""]
    return [str(raw_input) if raw_input is not None else ""]


def _segment_text_by_punctuation(text_fragments: list[str]) -> list[str]:
    merged = "".join(text_fragments).strip()
    if not merged:
        return []
    if not PUNCT_BUFFER_ENABLED:
        return [merged]

    segments: list[str] = []
    buf = ""

    def _emit(segment: str) -> None:
        seg = segment.strip()
        if seg:
            segments.append(seg)

    for frag in text_fragments:
        if not frag:
            continue
        buf += frag
        while True:
            hard = HARD_BOUNDARY_RE.search(buf)
            if hard:
                # Hard boundary should always flush for semantic completeness.
                _emit(buf[: hard.end()])
                buf = buf[hard.end():]
                continue

            soft = SOFT_BOUNDARY_RE.search(buf)
            if soft and len(buf) >= PUNCT_MIN_CHARS:
                _emit(buf[: soft.end()])
                buf = buf[soft.end():]
                continue

            if len(buf) >= PUNCT_MAX_CHARS:
                split_pos = -1
                for idx, ch in enumerate(buf[:PUNCT_MAX_CHARS]):
                    if HARD_BOUNDARY_RE.match(ch) or SOFT_BOUNDARY_RE.match(ch):
                        split_pos = idx + 1
                if split_pos <= 0:
                    split_pos = PUNCT_MAX_CHARS
                _emit(buf[:split_pos])
                buf = buf[split_pos:]
                continue
            break

    _emit(buf)
    return segments


async def stream_tts(
    text: str,
    response_format: str = "wav",
    request: Request | None = None,
    request_id: str | None = None,
    segment_index: int = 0,
    segment_total: int = 1,
    emit_wav_header: bool = True,
):
    """Async generator: stream TTS audio from Triton CosyVoice2."""
    collector = StreamCollector()
    collector.start_time = time.time()
    req_id = request_id or str(uuid4())

    client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)

    # Prepare input - only target_text (uses default speaker)
    target_text = np.array([[text]], dtype=object)
    inputs = [grpcclient.InferInput("target_text", [1, 1], "BYTES")]
    inputs[0].set_data_from_numpy(target_text)
    outputs = [grpcclient.InferRequestedOutput("waveform")]

    # Start streaming
    client.start_stream(callback=collector.callback, stream_timeout=60)
    client.async_stream_infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )

    loop = asyncio.get_event_loop()
    sent_header = False
    total_samples = 0
    first_chunk = True
    buffered_chunks: list[tuple[bytes, float]] = []
    buffered_seconds = 0.0
    jitter_started = False
    target_jitter_seconds = OUTPUT_JITTER_BUFFER_MS / 1000.0
    chunk_count = 0
    chunk_intervals_ms: list[float] = []
    last_chunk_emit_monotonic = 0.0

    async def flush_buffered() -> None:
        nonlocal sent_header
        if not buffered_chunks:
            return
        if emit_wav_header and not sent_header and response_format == "wav":
            yield_header = create_wav_header(SAMPLE_RATE, BITS_PER_SAMPLE, NUM_CHANNELS)
            sent_header = True
            yield yield_header
        for payload, _payload_seconds in buffered_chunks:
            yield payload
        buffered_chunks.clear()

    try:
        logger.info(
            f"[{req_id}] start segment {segment_index + 1}/{segment_total}, "
            f"text_len={len(text)}, punct_buffer={PUNCT_BUFFER_ENABLED}, jitter_ms={OUTPUT_JITTER_BUFFER_MS}"
        )
        while True:
            if request is not None and await request.is_disconnected():
                logger.info(f"[{req_id}] client disconnected, cancelling stream")
                break

            timeout = FIRST_CHUNK_TIMEOUT if first_chunk else NEXT_CHUNK_TIMEOUT
            try:
                chunk = await loop.run_in_executor(
                    None, lambda t=timeout: collector.queue.get(timeout=t)
                )
            except queue.Empty:
                if first_chunk:
                    logger.warning(f"Triton timeout waiting for first chunk ({FIRST_CHUNK_TIMEOUT}s)")
                async for payload in flush_buffered():
                    yield payload
                break

            if chunk is None:
                if collector.error:
                    logger.error(f"[{req_id}] Triton error: {collector.error}")
                async for payload in flush_buffered():
                    yield payload
                break

            first_chunk = False

            # Convert float32 to int16 PCM
            audio_int16 = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
            total_samples += len(audio_int16)
            chunk_bytes = audio_int16.tobytes()
            chunk_seconds = len(audio_int16) / SAMPLE_RATE

            # Zero jitter means pure passthrough mode: no pacing and no buffering.
            if target_jitter_seconds <= 0:
                if emit_wav_header and not sent_header and response_format == "wav":
                    sent_header = True
                    yield create_wav_header(SAMPLE_RATE, BITS_PER_SAMPLE, NUM_CHANNELS)
                emit_ts = time.monotonic()
                if last_chunk_emit_monotonic > 0:
                    chunk_intervals_ms.append((emit_ts - last_chunk_emit_monotonic) * 1000)
                last_chunk_emit_monotonic = emit_ts
                chunk_count += 1
                yield chunk_bytes
                continue

            if not jitter_started:
                buffered_chunks.append((chunk_bytes, chunk_seconds))
                buffered_seconds += chunk_seconds

                if buffered_seconds < target_jitter_seconds:
                    continue

                jitter_started = True

                async for payload in flush_buffered():
                    now = time.monotonic()
                    if last_chunk_emit_monotonic > 0:
                        chunk_intervals_ms.append((now - last_chunk_emit_monotonic) * 1000)
                    last_chunk_emit_monotonic = now
                    chunk_count += 1
                    yield payload
                continue

            emit_ts = time.monotonic()
            if last_chunk_emit_monotonic > 0:
                chunk_intervals_ms.append((emit_ts - last_chunk_emit_monotonic) * 1000)
            last_chunk_emit_monotonic = emit_ts
            chunk_count += 1
            yield chunk_bytes

    finally:
        client.stop_stream(cancel_requests=True)
        elapsed = time.time() - collector.start_time
        audio_duration = total_samples / SAMPLE_RATE
        ttfb = (collector.first_chunk_time - collector.start_time) * 1000 if collector.first_chunk_time else 0
        p95_gap = _percentile(chunk_intervals_ms, 0.95)
        max_gap = max(chunk_intervals_ms) if chunk_intervals_ms else 0.0
        if audio_duration > 0:
            logger.info(
                f"[{req_id}] segment done {segment_index + 1}/{segment_total}: "
                f"text_len={len(text)} chunks={chunk_count} ttfb={ttfb:.0f}ms "
                f"total={elapsed*1000:.0f}ms audio={audio_duration:.2f}s "
                f"rtf={elapsed/audio_duration:.3f} p95_gap={p95_gap:.1f}ms max_gap={max_gap:.1f}ms"
            )
        else:
            logger.warning(
                f"[{req_id}] no audio produced for segment {segment_index + 1}/{segment_total}: "
                f"text_len={len(text)} elapsed={elapsed*1000:.0f}ms"
            )


async def stream_tts_with_punctuation_buffer(
    text_fragments: list[str],
    response_format: str,
    request: Request | None = None,
):
    req_id = str(uuid4())
    full_text = "".join(text_fragments)
    segments = _segment_text_by_punctuation(text_fragments)
    if not segments:
        logger.warning(f"[{req_id}] empty text after segmentation")
        return

    logger.info(
        f"[{req_id}] request accepted: fragments={len(text_fragments)} segments={len(segments)} "
        f"input_chars={sum(len(f) for f in text_fragments)} "
        f"buffer_enabled={PUNCT_BUFFER_ENABLED} min={PUNCT_MIN_CHARS} max={PUNCT_MAX_CHARS} "
        f"max_wait_ms={PUNCT_MAX_WAIT_MS}"
    )
    logger.info(f"[{req_id}] input_text={full_text!r}")

    for idx, segment in enumerate(segments):
        logger.info(f"[{req_id}] segment_text {idx + 1}/{len(segments)}: {segment!r}")
        async for payload in stream_tts(
            text=segment,
            response_format=response_format,
            request=request,
            request_id=req_id,
            segment_index=idx,
            segment_total=len(segments),
            emit_wav_header=(idx == 0),
        ):
            yield payload


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)
        if client.is_server_live():
            return {"status": "ok", "triton": "live"}
        return JSONResponse({"status": "error", "triton": "not live"}, status_code=503)
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=503)


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "cosyvoice2-0.5b",
                "object": "model",
                "created": 1700000000,
                "owned_by": "FunAudioLLM",
            }
        ]
    }


@app.post("/v1/audio/speech")
async def create_speech(request: Request):
    """OpenAI-compatible TTS endpoint.

    Request body:
        {
            "model": "cosyvoice2-0.5b",  // ignored, only one model
            "input": "要合成的文本",
            "voice": "default",           // currently only default speaker
            "response_format": "wav",     // wav or pcm
            "speed": 1.0                  // currently ignored
        }
    """
    body = await request.json()
    text_fragments = _normalize_input_text(body.get("input", ""))
    if not any(text_fragments):
        return JSONResponse({"error": {"message": "input is required"}}, status_code=400)

    response_format = body.get("response_format", "wav")
    if response_format not in ("wav", "pcm"):
        response_format = "wav"

    content_type = "audio/wav" if response_format == "wav" else "audio/pcm"

    session_id = _resolve_session_id(request)
    lock: asyncio.Lock | None = None
    queued_wait_ms = 0.0

    if SESSION_SERIALIZE_ENABLED:
        lock = await _get_or_create_session_lock(session_id)
        wait_start = time.monotonic()
        try:
            await asyncio.wait_for(lock.acquire(), timeout=SESSION_LOCK_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                f"session lock timeout: session_id={session_id!r}, timeout_s={SESSION_LOCK_TIMEOUT_S}"
            )
            return JSONResponse(
                {"error": {"message": "session queue timeout, please retry"}},
                status_code=429,
            )
        queued_wait_ms = (time.monotonic() - wait_start) * 1000
        logger.info(
            f"session lock acquired: session_id={session_id!r}, queued_wait_ms={queued_wait_ms:.1f}"
        )

    async def guarded_stream():
        try:
            async for payload in stream_tts_with_punctuation_buffer(
                text_fragments, response_format, request=request
            ):
                yield payload
        finally:
            if lock is not None and lock.locked():
                lock.release()
                logger.info(
                    f"session lock released: session_id={session_id!r}, queued_wait_ms={queued_wait_ms:.1f}"
                )

    return StreamingResponse(
        guarded_stream(),
        media_type=content_type,
        headers={
            "Content-Type": content_type,
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Bits-Per-Sample": str(BITS_PER_SAMPLE),
            "Transfer-Encoding": "chunked",
            "X-Session-Id": session_id,
            "X-Session-Queue-Wait-Ms": f"{queued_wait_ms:.1f}",
        }
    )


def main():
    parser = argparse.ArgumentParser(description="CosyVoice2 TRT-LLM OpenAI Bridge")
    parser.add_argument("--triton-url", default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=9880, help="Listen port")
    args = parser.parse_args()

    global TRITON_URL
    TRITON_URL = args.triton_url

    logger.info(f"Starting CosyVoice2 Bridge on {args.host}:{args.port}")
    logger.info(f"Triton backend: {TRITON_URL}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
