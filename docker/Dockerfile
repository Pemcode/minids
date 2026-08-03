# Image du pod miniDS : VGGT-Ω + SAM 3 + gsplat (2DGS) + Open3D + API HTTP.
#
# Base *devel* et non *runtime* : gsplat compile ses noyaux CUDA à l'installation.
# TORCH_CUDA_ARCH_LIST couvre A100 (8.0), RTX 30xx (8.6), L4 / RTX 4090 (8.9) et
# H100 (9.0) — sans cette liste, la compilation ne cible que le GPU du builder.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    HF_HOME=/workspace/minids/cache/hf \
    MINIDS_DATA_DIR=/workspace/minids \
    MINIDS_PORT=8000

# ffmpeg est requis même côté pod : il sert quand on envoie la vidéo brute
# (--send-video). libzimg fournit zscale, donc le tonemap HDR→SDR.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git build-essential ninja-build \
        libgl1 libglib2.0-0 libgomp1 libzimg2 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

COPY docker/requirements.txt /opt/requirements.txt
RUN pip install --upgrade pip && pip install -r /opt/requirements.txt

# Optionnel : sans wheel disponible, on continue sans ajustement de faisceaux
# plutôt que de perdre tout le build (le pipeline gère déjà son absence).
RUN pip install pycolmap==3.11.1 \
    || echo "AVERTISSEMENT: pycolmap indisponible — option --bundle-adjustment inactive"

# Précompile les noyaux CUDA de gsplat pendant le build. Sans cela, ils se
# compilent au premier scan sur le pod : plusieurs minutes de GPU facturées, à
# chaque démarrage de pod.
RUN python -c "import gsplat.cuda._backend as b; print('noyaux gsplat compilés')" \
    || echo "AVERTISSEMENT: précompilation gsplat impossible — compilation JIT au premier scan"

# VGGT-Ω : dépôt source, les poids sont téléchargés au démarrage (accès HF restreint).
ARG VGGT_OMEGA_REF=main
RUN git clone --depth 1 --branch ${VGGT_OMEGA_REF} https://github.com/facebookresearch/vggt-omega.git /opt/vggt-omega \
    && pip install -e /opt/vggt-omega

WORKDIR /app
COPY common /app/common
COPY server /app/server
COPY client /app/client
COPY bench /app/bench
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
