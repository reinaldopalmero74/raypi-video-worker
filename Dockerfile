# Imagen pre-construida para el worker de videos RAY PI 3.14.
# Basada en runpod/pytorch (probada: pip installs funcionan).
# Incluye: ffmpeg, repos Wan2.2 + LatentSync, todas las dependencias Python, worker.sh.
# Los modelos (~12GB) se descargan en el primer arranque (no van en la imagen).
FROM runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update -qq && apt-get install -y -qq ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -q "huggingface_hub[cli]" hf_transfer

# Clonar repos de los modelos
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2 /opt/Wan2.2 \
    && git clone --depth 1 https://github.com/bytedance/LatentSync /opt/LatentSync

# Dependencias Python (con fix: mediapipe 0.10.11 ya no existe en PyPI)
RUN grep -v -E "^(torch|torchvision)" /opt/LatentSync/requirements.txt \
      | sed 's/mediapipe==0.10.11/mediapipe==0.10.13/' > /tmp/ls-req.txt \
    && pip install --no-cache-dir -r /tmp/ls-req.txt \
    && grep -v -E "^(torch|torchvision|torchaudio)" /opt/Wan2.2/requirements.txt > /tmp/wan-req.txt \
    && pip install --no-cache-dir -r /tmp/wan-req.txt \
    && pip install --no-cache-dir peft \
    && curl -sL -o /tmp/flash_attn.whl "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
    && pip install --no-cache-dir /tmp/flash_attn.whl \
    && rm /tmp/flash_attn.whl \
    && rm -rf /root/.cache/pip

# Worker
COPY worker.sh /worker.sh
RUN chmod +x /worker.sh && bash -n /worker.sh

# Env vars requeridas (vienen del pod): SONG_URL, IMAGE_URL, SCENES_JSON, etc.
# Ver worker.sh para la lista completa.
CMD ["/worker.sh"]
