#!/usr/bin/env bash
# Démarrage du serveur miniDS sur le pod.
#
# Si MINIDS_TOKEN n'est pas fourni, on en génère un et on l'affiche dans les logs
# du pod : lisibles depuis l'interface RunPod, donc toujours pas besoin d'ouvrir
# un terminal web.
set -euo pipefail
umask 077

DATA_DIR="${MINIDS_DATA_DIR:-/workspace/minids}"
PORT="${MINIDS_PORT:-8000}"
mkdir -p "$DATA_DIR/cache" "$DATA_DIR/jobs"

if [ -z "${MINIDS_TOKEN:-}" ]; then
    TOKEN_FILE="$DATA_DIR/token.txt"
    MINIDS_TOKEN=""
    if [ -s "$TOKEN_FILE" ]; then
        MINIDS_TOKEN="$(<"$TOKEN_FILE")"
    fi
    if [ -z "$MINIDS_TOKEN" ]; then
        MINIDS_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
        printf '%s' "$MINIDS_TOKEN" > "$TOKEN_FILE"
    fi
    chmod 600 "$TOKEN_FILE"
    export MINIDS_TOKEN
    echo "=============================================================="
    echo " MINIDS_TOKEN généré : $MINIDS_TOKEN"
    echo " (à passer au client via --token ou la variable MINIDS_TOKEN)"
    echo "=============================================================="
fi

echo "miniDS — données: $DATA_DIR | port: $PORT | fake_gpu: ${MINIDS_FAKE_GPU:-0}"
if [ "${MINIDS_FAKE_GPU:-0}" != "1" ]; then
    if [ -z "${MINIDS_CKPT:-}" ]; then
        echo "ATTENTION: MINIDS_CKPT vide — renseigner un chemin local ou 'repo_id:fichier' Hugging Face."
    fi
    # Les poids VGGT-Ω sont sous accès restreint : sans jeton, le job ne meurt
    # qu'à l'étape vggt, après cinq étapes déjà facturées.
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "ATTENTION: HF_TOKEN vide — le téléchargement des poids VGGT-Ω échouera (401 « Please log in »)."
    fi
fi

python - <<'PY' || true
import torch
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} Go)", flush=True)
PY

exec uvicorn server.app:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --timeout-keep-alive 75 \
    --log-level "$(printf '%s' "${MINIDS_LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')" \
    "$@"
