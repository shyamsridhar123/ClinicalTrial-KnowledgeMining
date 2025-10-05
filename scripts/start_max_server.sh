#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${DOCINTEL_LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
LOG_FILE="${LOG_DIR}/max-serve-${TIMESTAMP}.log"

export MAX_LOG_LEVEL="${MAX_LOG_LEVEL:-info}"
export MAX_LOG_FORMAT="${MAX_LOG_FORMAT:-text}"
export MODULAR_CACHE_DIR="${MODULAR_CACHE_DIR:-${PROJECT_ROOT}/models}"
export HUGGINGFACE_HUB_CACHE="${PROJECT_ROOT}/models"
export HF_HOME="${PROJECT_ROOT}/models"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

MODEL_NAME="${DOCLING_MODEL_NAME:-ibm-granite/granite-docling-258M}"
PORT="${DOCLING_MAX_PORT:-8000}"
DEVICE_UTIL="${DOCLING_DEVICE_MEM_UTIL:-0.9}"
QUANT_ENCODING="${DOCLING_QUANT_ENCODING:-bfloat16}"

CMD=(
  max serve
  --model "${MODEL_NAME}"
  --port "${PORT}"
  --device-memory-utilization "${DEVICE_UTIL}"
  --quantization-encoding "${QUANT_ENCODING}"
)

STRUCTURED_FLAG="${DOCLING_ENABLE_STRUCTURED_OUTPUT:-${DOCINTEL_DOCLING_STRUCTURED_OUTPUT_ENABLED:-1}}"
case "${STRUCTURED_FLAG,,}" in
  1|true|yes)
    CMD+=(--enable-structured-output)
    ;;
  *)
    ;;
esac

{
  echo "[$(date --iso-8601=seconds)] Starting Modular MAX server"
  echo "[$(date --iso-8601=seconds)] Logs: ${LOG_FILE}"
  echo "[$(date --iso-8601=seconds)] MODULAR_CACHE_DIR: ${MODULAR_CACHE_DIR}"
  echo "[$(date --iso-8601=seconds)] HUGGINGFACE_HUB_CACHE: ${HUGGINGFACE_HUB_CACHE}"
  echo "[$(date --iso-8601=seconds)] Model directory contents:"
  ls -la "${PROJECT_ROOT}/models/"
  echo "[$(date --iso-8601=seconds)] Command: ${CMD[*]}"
  "${CMD[@]}"
} 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

echo "[$(date --iso-8601=seconds)] Modular MAX server exited with code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
exit "${EXIT_CODE}"
