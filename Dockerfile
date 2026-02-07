FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install vox-box dependencies from PyPI (stable, resolved versions)
RUN pip install --no-cache-dir vox-box

# Copy local source with streaming TTS improvements
COPY vox_box /opt/vox_box_src

# Overwrite installed package with our modified source
RUN SITE_PKG=$(python -c "import vox_box; import os; print(os.path.dirname(vox_box.__file__))") && \
    cp -r /opt/vox_box_src/* "$SITE_PKG/" && \
    rm -rf /opt/vox_box_src && \
    echo "Patched vox_box with streaming TTS support"

# Create data directory
RUN mkdir -p /data

EXPOSE 80

ENTRYPOINT ["vox-box", "start"]
CMD ["--host", "0.0.0.0", "--port", "80", "--data-dir", "/data"]
