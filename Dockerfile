FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install vox-box (torch/torchaudio already in base image)
RUN pip install --no-cache-dir vox-box

# Patch: disable JIT loading for CosyVoice (JIT .zip files are not in HuggingFace repos)
RUN COSYVOICE_CLI=$(python -c "import vox_box; import os; print(os.path.join(os.path.dirname(vox_box.__file__), 'third_party/CosyVoice/cosyvoice/cli/cosyvoice.py'))") && \
    sed -i 's/def __init__(self, model_dir, load_jit=True, load_onnx=False, fp16=True):/def __init__(self, model_dir, load_jit=False, load_onnx=False, fp16=True):/' "$COSYVOICE_CLI" && \
    echo "Patched CosyVoice: load_jit default changed to False"

# Create data directory
RUN mkdir -p /data

EXPOSE 80

ENTRYPOINT ["vox-box", "start"]
CMD ["--host", "0.0.0.0", "--port", "80", "--data-dir", "/data"]
