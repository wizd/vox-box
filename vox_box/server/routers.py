import asyncio
import functools
import logging
import mimetypes
import queue
import struct
import threading
from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse

from vox_box.backends.stt.base import STTBackend
from vox_box.backends.tts.base import TTSBackend
from vox_box.server.model import get_model_instance
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

router = APIRouter()

executor = ThreadPoolExecutor()

ALLOWED_SPEECH_OUTPUT_AUDIO_TYPES = {
    "mp3",
    "opus",
    "aac",
    "flac",
    "wav",
    "pcm",
}

# Formats that support true HTTP-level streaming (no container finalization needed)
STREAMABLE_FORMATS = {"pcm", "wav"}


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: str = "mp3"
    speed: float = 1.0


def _create_wav_header(sample_rate: int, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Create a WAV/RIFF header for streaming (data size set to max placeholder)."""
    data_size = 0x7FFFFFFF  # Unknown size — streaming
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        data_size + 36,
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header


@router.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    try:
        if (
            request.response_format
            and request.response_format not in ALLOWED_SPEECH_OUTPUT_AUDIO_TYPES
        ):
            return HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {request.response_format}",
            )

        if request.speed < 0.25 or request.speed > 2:
            return HTTPException(
                status_code=400, detail="Speed must be between 0.25 and 2"
            )

        model_instance: TTSBackend = get_model_instance()
        if not isinstance(model_instance, TTSBackend):
            return HTTPException(
                status_code=400, detail="Model instance does not support speech API"
            )

        # Determine if we can use streaming path:
        # 1. Format must be streamable (pcm or wav)
        # 2. Speed must be 1.0 (CosyVoice streaming doesn't support speed changes)
        # 3. Backend must support speech_stream()
        can_stream = (
            request.response_format in STREAMABLE_FORMATS
            and request.speed == 1.0
            and hasattr(model_instance, "speech_stream")
        )

        if can_stream:
            try:
                # Verify speech_stream is actually implemented (not just base class stub)
                model_instance.speech_stream.__func__
                if model_instance.speech_stream.__func__ is TTSBackend.speech_stream:
                    can_stream = False
            except AttributeError:
                pass

        if can_stream:
            return _streaming_speech_response(model_instance, request)
        else:
            return await _file_speech_response(model_instance, request)

    except Exception as e:
        return HTTPException(status_code=500, detail=f"Failed to generate speech, {e}")


def _streaming_speech_response(model_instance: TTSBackend, request: SpeechRequest):
    """Return a StreamingResponse that yields audio chunks in real-time."""
    sample_rate = getattr(model_instance, "stream_sample_rate", 22050)
    media_type = get_media_type(request.response_format)

    async def audio_stream_generator():
        chunk_queue = queue.Queue(maxsize=20)
        error_holder = [None]

        def producer():
            try:
                for pcm_chunk in model_instance.speech_stream(
                    request.input,
                    request.voice,
                    request.speed,
                ):
                    chunk_queue.put(pcm_chunk)
            except Exception as e:
                error_holder[0] = e
            finally:
                chunk_queue.put(None)  # Sentinel: end of stream

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        # For WAV format, emit RIFF header before PCM data
        if request.response_format == "wav":
            yield _create_wav_header(sample_rate, channels=1, bits_per_sample=16)

        loop = asyncio.get_event_loop()
        while True:
            chunk = await loop.run_in_executor(None, chunk_queue.get)
            if chunk is None:
                break
            yield chunk

        thread.join(timeout=5)
        if error_holder[0]:
            logger.error(f"Streaming TTS error: {error_holder[0]}")

    headers = {
        "Transfer-Encoding": "chunked",
        "X-Audio-Sample-Rate": str(sample_rate),
        "X-Audio-Channels": "1",
        "X-Audio-Bits-Per-Sample": "16",
        "X-Audio-Codec": "pcm_s16le",
    }

    return StreamingResponse(
        audio_stream_generator(),
        media_type=media_type,
        headers=headers,
    )


async def _file_speech_response(model_instance: TTSBackend, request: SpeechRequest):
    """Return a FileResponse with the complete audio file (non-streaming fallback)."""
    func = functools.partial(
        model_instance.speech,
        request.input,
        request.voice,
        request.speed,
        request.response_format,
    )

    loop = asyncio.get_event_loop()
    audio_file = await loop.run_in_executor(
        executor,
        func,
    )

    media_type = get_media_type(request.response_format)
    return FileResponse(audio_file, media_type=media_type)


# ref: https://github.com/LMS-Community/slimserver/blob/public/10.0/types.conf
ALLOWED_TRANSCRIPTIONS_INPUT_AUDIO_FORMATS = {
    # flac
    "audio/flac",
    "audio/x-flac",
    # mp3
    "audio/mpeg",
    "audio/x-mpeg",
    "audio/mp3",
    "audio/mp3s",
    "audio/mpeg3",
    "audio/mpg",
    # mp4
    "audio/m4a",
    "audio/x-m4a",
    "audio/mp4",
    # mpeg
    "audio/mpga",
    # ogg
    "audio/ogg",
    "audio/x-ogg",
    # wav
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    # webm
    "video/webm",
    "audio/webm",
    # file
    "application/octet-stream",
}

ALLOWED_TRANSCRIPTIONS_OUTPUT_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}


@router.post("/v1/audio/transcriptions")
async def transcribe(request: Request):
    try:
        form = await request.form()
        keys = form.keys()
        if "file" not in keys:
            return HTTPException(status_code=400, detail="Field file is required")

        file: UploadFile = form[
            "file"
        ]  # flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm
        file_content_type = file.content_type or mimetypes.guess_type(file.filename)[0]
        if file_content_type not in ALLOWED_TRANSCRIPTIONS_INPUT_AUDIO_FORMATS:
            return HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_content_type}",
            )

        audio_bytes = await file.read()
        language = form.get("language")
        prompt = form.get("prompt")
        temperature = float(form.get("temperature", 0))
        if not (0 <= temperature <= 1):
            return HTTPException(
                status_code=400, detail="Temperature must be between 0 and 1"
            )

        timestamp_granularities = form.getlist("timestamp_granularities")
        response_format = form.get("response_format", "json")
        if response_format not in ALLOWED_TRANSCRIPTIONS_OUTPUT_FORMATS:
            return HTTPException(
                status_code=400, detail="Unsupported response_format: {response_format}"
            )

        model_instance: STTBackend = get_model_instance()
        if not isinstance(model_instance, STTBackend):
            return HTTPException(
                status_code=400,
                detail="Model instance does not support transcriptions API",
            )

        kwargs = {
            "content_type": file_content_type,
        }
        func = functools.partial(
            model_instance.transcribe,
            audio_bytes,
            language,
            prompt,
            temperature,
            timestamp_granularities,
            response_format,
            **kwargs,
        )

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            executor,
            func,
        )

        if response_format == "json":
            return {"text": data}
        elif response_format == "text":
            return data
        else:
            return data
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Failed to transcribe audio, {e}")


@router.get("/health")
async def health():
    model_instance = get_model_instance()
    if model_instance is None or (not model_instance.is_load()):
        return HTTPException(status_code=503, detail="Loading model")
    return {"status": "ok"}


@router.get("/v1/models")
async def get_model_list():
    model_instance = get_model_instance()
    if model_instance is None:
        return []
    return {"object": "list", "data": [model_instance.model_info()]}


@router.get("/v1/models/{model_id}")
async def get_model_info(model_id: str):
    model_instance = get_model_instance()
    if model_instance is None:
        return {}
    return model_instance.model_info()


@router.get("/v1/languages")
async def get_languages():
    model_instance = get_model_instance()
    if model_instance is None:
        return {}
    return {
        "languages": model_instance.model_info().get("languages", []),
    }


@router.get("/v1/voices")
async def get_voice():
    model_instance = get_model_instance()
    if model_instance is None:
        return {}
    return {
        "voices": model_instance.model_info().get("voices", []),
    }


def get_media_type(response_format) -> str:
    if response_format == "mp3":
        media_type = "audio/mpeg"
    elif response_format == "opus":
        media_type = "audio/ogg;codec=opus"
    elif response_format == "aac":
        media_type = "audio/aac"
    elif response_format == "flac":
        media_type = "audio/x-flac"
    elif response_format == "wav":
        media_type = "audio/wav"
    elif response_format == "pcm":
        media_type = "audio/pcm"
    else:
        raise Exception(
            f"Invalid response_format: '{response_format}'", param="response_format"
        )

    return media_type
