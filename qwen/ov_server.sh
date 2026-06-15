#!/usr/bin/env bash
# Start the LLaVA-OneVision-2-8B-Instruct video reward server (drop-in for server.sh).
#
# Run this INSIDE the OneVision-2 env (transformers>=5.7.0), on its own GPU.
#
#   export CUDA_VISIBLE_DEVICES=0
#   export OV_MODEL_PATH="lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"   # or a local dir
#   bash GRPO/qwen/ov_server.sh
#
# Optional environment variables:
#   HOST                 default: 0.0.0.0
#   PORT                 default: 18080   (use a DIFFERENT port if Qwen server is up)
#   DEVICE               default: cuda:0
#   DTYPE                default: bfloat16
#   PROMPT_DIR           default: /aigc/posttrain/siyuanfu/GRPO/prompts
#   VIDEO_BACKEND        default: frames  (paper/model-card default for short clips; codec = long-video mode)
#   VIDEO_NFRAMES        default: 16      (matches the OneVision-2-8B model-card video demo)
#   VIDEO_MAX_PIXELS     default: 200704  (=448x448, the model-card video example value)
#   ATTN_IMPLEMENTATION  default: auto
#   DEVICE_MAP_MODE      default: single
#   MAX_NEW_TOKENS       default: 1024
#   CACHE_SIZE           default: 512
#
# Extra arguments are forwarded to ov_server.py.

set -euo pipefail

MODEL_PATH="${OV_MODEL_PATH:-/aigc/posttrain/siyuanfu/models/OneVision}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
PROMPT_DIR="${PROMPT_DIR:-/aigc/posttrain/siyuanfu/GRPO/prompts}"
VIDEO_BACKEND="${VIDEO_BACKEND:-frames}"
VIDEO_NFRAMES="${VIDEO_NFRAMES:-16}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-200704}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
DEVICE_MAP_MODE="${DEVICE_MAP_MODE:-single}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
CACHE_SIZE="${CACHE_SIZE:-512}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXTRA_ARGS=()
if [[ -n "${VIDEO_MAX_PIXELS:-}" ]]; then
  EXTRA_ARGS+=(--video-max-pixels "${VIDEO_MAX_PIXELS}")
fi

exec python "${SCRIPT_DIR}/ov_server.py" \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --prompt-dir "${PROMPT_DIR}" \
  --video-backend "${VIDEO_BACKEND}" \
  --video-nframes "${VIDEO_NFRAMES}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --device-map-mode "${DEVICE_MAP_MODE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --cache-size "${CACHE_SIZE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
