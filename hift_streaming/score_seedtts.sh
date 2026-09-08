#!/bin/bash
# Score the wavs eval_seedtts.py wrote with the flow_grpo eval harness
# (CER via Paraformer, SS via ERes2Net, DNSMOS P.835 OVRL).
#
#   bash hift_streaming/score_seedtts.sh <out_root> [gpu] [systems...]
set -u
cd "$(dirname "$0")/.."
ROOT=$(cd "${1:-hift_streaming/exp/seedtts_zh}" && pwd)
GPU=${2:-0}
shift 2 2>/dev/null || true
SYSTEMS=${@:-"full recompute incremental graph"}
M=${MODEL_DIR:-/opt/dlami/nvme/leolxliu/vocoder/data/pretrained/CosyVoice3-0.5B}
PY=${PY:-/opt/conda_envs/flow_tts/bin/python}
export PYTHONPATH=$PWD:$PWD/third_party/Matcha-TTS
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/opt/dlami/nvme/leolxliu/.cache/modelscope}
export CUDA_VISIBLE_DEVICES=$GPU

for sys in $SYSTEMS; do
  echo "=== scoring $sys ==="
  (cd flow_grpo && $PY evaluate.py --model_dir "$M" --test_data "$ROOT/testset.jsonl" \
      --output_dir "$ROOT/$sys" --skip_synthesis)
done
echo "=== summaries ==="
for sys in $SYSTEMS; do
  echo -n "$sys: "; cat "$ROOT/$sys/summary.json" | tr -d '\n '; echo
done
