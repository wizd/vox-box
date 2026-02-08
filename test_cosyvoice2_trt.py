#!/usr/bin/env python3
"""CosyVoice2 TRT-LLM Triton 流式性能测试"""
import time
import queue
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

SERVER = "localhost:8001"
MODEL = "cosyvoice2"
SAMPLE_RATE = 24000

TEST_TEXTS = [
    "你好，请问你叫什么名字？",
    "今天天气真不错，我们一起出去走走吧。",
    "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
]


class StreamData:
    def __init__(self):
        self.queue = queue.Queue()
        self.start_time = None
        self.first_chunk_time = None
        self.chunks = []

    def record_start(self):
        self.start_time = time.time()


def callback(stream_data, result, error):
    if error:
        stream_data.queue.put(error)
        return
    now = time.time()
    try:
        output = result.as_numpy("waveform")
    except Exception:
        # Final empty response has no waveform tensor
        stream_data.queue.put(None)
        return
    if output is not None and len(output) > 0:
        if stream_data.first_chunk_time is None:
            stream_data.first_chunk_time = now
        stream_data.chunks.append(output.flatten())
        stream_data.queue.put(result)
    else:
        # Empty response = final from decoupled model
        stream_data.queue.put(None)


def test_streaming(text, server=SERVER, model=MODEL):
    """测试流式推理，返回 (ttfb_ms, total_ms, audio_duration_s, num_chunks)"""
    stream_data = StreamData()
    client = grpcclient.InferenceServerClient(url=server, verbose=False)

    # 准备输入
    target_text = np.array([[text]], dtype=object)
    inputs = [
        grpcclient.InferInput("target_text", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(target_text)

    outputs = [grpcclient.InferRequestedOutput("waveform")]

    # 开始流式推理
    stream_data.record_start()
    client.start_stream(callback=lambda result, error: callback(stream_data, result, error))
    client.async_stream_infer(
        model_name=model,
        inputs=inputs,
        outputs=outputs,
    )

    # 收集结果 - 使用短超时检测流结束
    final_error = None
    last_data_time = time.time()
    while True:
        try:
            item = stream_data.queue.get(timeout=3)
            if item is None:
                break  # final response
            if isinstance(item, Exception):
                final_error = item
                break
            last_data_time = time.time()
        except queue.Empty:
            # 3秒无新数据，认为流结束
            break

    client.stop_stream()

    end_time = time.time()
    total_ms = (end_time - stream_data.start_time) * 1000

    if final_error:
        print(f"  Error: {final_error}")
        return None

    ttfb_ms = None
    if stream_data.first_chunk_time:
        ttfb_ms = (stream_data.first_chunk_time - stream_data.start_time) * 1000

    # 计算音频时长
    total_samples = sum(len(c) for c in stream_data.chunks)
    audio_duration = total_samples / SAMPLE_RATE

    return ttfb_ms, total_ms, audio_duration, len(stream_data.chunks)


def test_offline(text, server=SERVER, model=MODEL):
    """测试离线推理（单次请求），返回 (latency_ms, audio_duration_s)"""
    client = grpcclient.InferenceServerClient(url=server, verbose=False)

    target_text = np.array([[text]], dtype=object)
    inputs = [
        grpcclient.InferInput("target_text", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(target_text)

    outputs = [grpcclient.InferRequestedOutput("waveform")]

    start = time.time()
    result = client.infer(model_name=model, inputs=inputs, outputs=outputs)
    latency = (time.time() - start) * 1000

    audio = result.as_numpy("waveform").flatten()
    audio_duration = len(audio) / SAMPLE_RATE

    return latency, audio_duration


if __name__ == "__main__":
    print("=" * 60)
    print("CosyVoice2 TRT-LLM Triton 性能测试")
    print(f"Server: {SERVER}, Model: {MODEL}")
    print("=" * 60)

    # Warmup
    print("\n--- Warmup (首次推理较慢) ---")
    try:
        r = test_offline("测试", server=SERVER)
        if r:
            print(f"  Warmup: {r[0]:.0f}ms, audio={r[1]:.2f}s")
    except Exception as e:
        print(f"  Warmup offline failed: {e}")
        print("  Trying streaming warmup...")
        try:
            r = test_streaming("测试", server=SERVER)
            if r:
                print(f"  Warmup streaming: TTFB={r[0]:.0f}ms, total={r[1]:.0f}ms")
        except Exception as e2:
            print(f"  Streaming warmup also failed: {e2}")

    # Streaming tests
    print("\n--- 流式推理测试 (Streaming) ---")
    for text in TEST_TEXTS:
        try:
            r = test_streaming(text, server=SERVER)
            if r:
                ttfb, total, dur, chunks = r
                rtf = (total / 1000) / dur if dur > 0 else float('inf')
                print(f"  Text: {text[:20]}...")
                print(f"    TTFB: {ttfb:.0f}ms | Total: {total:.0f}ms | "
                      f"Audio: {dur:.2f}s | Chunks: {chunks} | RTF: {rtf:.3f}")
        except Exception as e:
            print(f"  Text: {text[:20]}... Error: {e}")

    # Offline tests
    print("\n--- 离线推理测试 (Offline) ---")
    for text in TEST_TEXTS:
        try:
            r = test_offline(text, server=SERVER)
            if r:
                latency, dur = r
                rtf = (latency / 1000) / dur if dur > 0 else float('inf')
                print(f"  Text: {text[:20]}...")
                print(f"    Latency: {latency:.0f}ms | Audio: {dur:.2f}s | RTF: {rtf:.3f}")
        except Exception as e:
            print(f"  Text: {text[:20]}... Error: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
