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
import queue
import struct
import time
import logging

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


FIRST_CHUNK_TIMEOUT = 30   # seconds – first chunk may be slow (model warm-up)
NEXT_CHUNK_TIMEOUT = 10    # seconds – subsequent chunks should arrive faster


async def stream_tts(text: str, response_format: str = "wav"):
    """Async generator: stream TTS audio from Triton CosyVoice2."""
    collector = StreamCollector()
    collector.start_time = time.time()

    client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)

    # Prepare input - only target_text (uses default speaker)
    target_text = np.array([[text]], dtype=object)
    inputs = [grpcclient.InferInput("target_text", [1, 1], "BYTES")]
    inputs[0].set_data_from_numpy(target_text)
    outputs = [grpcclient.InferRequestedOutput("waveform")]

    # Start streaming
    client.start_stream(callback=collector.callback)
    client.async_stream_infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )

    loop = asyncio.get_event_loop()
    sent_header = False
    total_samples = 0
    first_chunk = True
    try:
        while True:
            timeout = FIRST_CHUNK_TIMEOUT if first_chunk else NEXT_CHUNK_TIMEOUT
            try:
                chunk = await loop.run_in_executor(
                    None, lambda t=timeout: collector.queue.get(timeout=t)
                )
            except queue.Empty:
                if first_chunk:
                    logger.warning(f"Triton timeout waiting for first chunk ({FIRST_CHUNK_TIMEOUT}s)")
                break

            if chunk is None:
                if collector.error:
                    logger.error(f"Triton error: {collector.error}")
                break

            first_chunk = False

            # Convert float32 to int16 PCM
            audio_int16 = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
            total_samples += len(audio_int16)

            if not sent_header and response_format == "wav":
                yield create_wav_header(SAMPLE_RATE, BITS_PER_SAMPLE, NUM_CHANNELS)
                sent_header = True

            yield audio_int16.tobytes()

    finally:
        client.stop_stream()
        elapsed = time.time() - collector.start_time
        audio_duration = total_samples / SAMPLE_RATE
        ttfb = (collector.first_chunk_time - collector.start_time) * 1000 if collector.first_chunk_time else 0
        if audio_duration > 0:
            logger.info(f"TTS completed: text='{text[:30]}...' ttfb={ttfb:.0f}ms "
                        f"total={elapsed*1000:.0f}ms audio={audio_duration:.2f}s "
                        f"rtf={elapsed/audio_duration:.3f}")
        else:
            logger.warning(f"TTS produced no audio: text='{text[:30]}...' elapsed={elapsed*1000:.0f}ms")


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
    text = body.get("input", "")
    if not text:
        return JSONResponse({"error": {"message": "input is required"}}, status_code=400)

    response_format = body.get("response_format", "wav")
    if response_format not in ("wav", "pcm"):
        response_format = "wav"

    content_type = "audio/wav" if response_format == "wav" else "audio/pcm"

    return StreamingResponse(
        stream_tts(text, response_format),
        media_type=content_type,
        headers={
            "Content-Type": content_type,
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Bits-Per-Sample": str(BITS_PER_SAMPLE),
            "Transfer-Encoding": "chunked",
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
