#!/usr/bin/env bash
# Start the Qwen3-VL-72B video reward server.
#
# Fill MODEL_PATH before running, for example:
#   export MODEL_PATH="/path/to/Qwen3-VL-72B"
#   export CUDA_VISIBLE_DEVICES=0
#   ./scripts/start_qwen3vl_reward_server.sh
#
# Optional environment variables:
#   HOST                 default: 0.0.0.0
#   PORT                 default: 18080
#   DEVICE               default: cuda:0
#   DTYPE                default: bfloat16
#   PROMPT_DIR           default: /Users/siyuan.fu/fsy/posttrain/prompts
#   ATTN_IMPLEMENTATION  default: auto; uses flash_attention_2 if available, otherwise sdpa
#   MAX_NEW_TOKENS       default: 1024
#   CACHE_SIZE           default: 512
#   VIDEO_FPS            default: 1.0 for judge frame sampling, independent of generated mp4 fps
#
# Extra arguments are forwarded to qwen3vl_reward_server.py.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/aigc/opensourcemodel/Qwen3-VL-32B-Instruct-FP8}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is empty. Set it to your Qwen3-VL-72B path before running." >&2
  exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
PROMPT_DIR="${PROMPT_DIR:-/aigc/posttrain/siyuanfu/prompts}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
CACHE_SIZE="${CACHE_SIZE:-512}"
VIDEO_FPS="${VIDEO_FPS:-1.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python "${SCRIPT_DIR}/qwen3vl_reward_server.py" \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --prompt-dir "${PROMPT_DIR}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --cache-size "${CACHE_SIZE}" \
  --video-fps "${VIDEO_FPS}" \
  "$@"
