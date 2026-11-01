#!/bin/bash
# Full experiment on one GPU: a plain vLLM sweep, then one with the EAGLE-3 draft.
# Needs root, Docker with GPU support and uv. Run from the repo root.
#   OUT=out scripts/run.sh [sj-sweep args...]
set -euo pipefail

GPU=${GPU:-0}
MODEL=${MODEL:-Qwen/Qwen3-8B}
DRAFT=${DRAFT:-RedHatAI/Qwen3-8B-speculator.eagle3}
DRAFT_TOKENS=${DRAFT_TOKENS:-3}
IMAGE=${IMAGE:-vllm/vllm-openai:v0.32.0}
OUT=${OUT:-out}
HF=${HF:-/opt/hf}
mkdir -p "$OUT" "$HF"

stop() { docker rm -f sj-vllm >/dev/null 2>&1 || true; }
restore() {
  stop
  nvidia-smi -i "$GPU" -rgc >/dev/null 2>&1 || true
  nvidia-smi -i "$GPU" -q -d CLOCK,PERFORMANCE >"$OUT/clocks-after.txt" 2>&1 || true
}
trap restore EXIT

nvidia-smi -i "$GPU" -q -d CLOCK,PERFORMANCE,POWER >"$OUT/clocks-before.txt"
max=$(nvidia-smi -i "$GPU" --query-gpu=clocks.max.sm --format=csv,noheader,nounits)
if ! nvidia-smi -i "$GPU" -lgc "$max,$max" || ! nvidia-smi -i "$GPU" -rgc; then
  echo "sj: CLOCK LOCK DENIED"
  exit 3
fi
echo "sj: clock lock works"

docker pull -q "$IMAGE"
for m in "$MODEL" "$DRAFT"; do
  docker run --rm -v "$HF:/root/.cache/huggingface" --entrypoint python3 "$IMAGE" -c \
    "from huggingface_hub import snapshot_download as s; s('$m', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.py'])"
done
echo "sj: models cached"
uv sync --frozen -q

serve() {
  stop
  docker run -d --name sj-vllm --gpus "device=$GPU" --ipc=host -p 8000:8000 \
    -e HF_HUB_OFFLINE=1 -v "$HF:/root/.cache/huggingface" "$IMAGE" \
    --model "$MODEL" --no-enable-prefix-caching --max-num-seqs 128 --max-model-len 4096 "$@"
  until curl -sf localhost:8000/health >/dev/null; do
    sleep 5
    if [ -z "$(docker ps -q -f name=^sj-vllm$)" ]; then docker logs sj-vllm | tail -80; exit 4; fi
  done
}

spec_config=$(printf '{"model":"%s","num_speculative_tokens":%s,"method":"eagle3"}' "$DRAFT" "$DRAFT_TOKENS")
for arm in plain spec; do
  if [ "$arm" = spec ]; then serve --speculative-config "$spec_config"; else serve; fi
  echo "sj: $arm server up at $(date -u +%FT%TZ)"
  uv run sj-sweep --arm "$arm" --gpu "$GPU" --out "$OUT/$arm.jsonl" "$@"
  docker logs sj-vllm >"$OUT/vllm-$arm.log" 2>&1
done
stop
echo "sj: done at $(date -u +%FT%TZ)"
