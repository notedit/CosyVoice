#!/bin/bash
# CER (Qwen3-ASR) / SS (CAM++) / DNSMOS on the three vocoder systems, paired vs full.
set -e
cd "$(dirname "$0")/.."
ROOT=${ROOT:-hift_streaming/exp/bench}
SYSTEMS=${SYSTEMS:-full recompute incremental}
EV=hidden_conditioning/eval30k
CV3=/opt/dlami/nvme/leolxliu/vocoder/data/pretrained/CosyVoice3-0.5B
PY=/opt/conda_envs/flow_tts/bin/python
ASR_PY=venv_qwen3_asr/bin/python
export PYTHONPATH=$PWD:$PWD/third_party/Matcha-TTS
export CUDA_VISIBLE_DEVICES=${1:-4}
for sys in $SYSTEMS; do
  echo "=== transcribe $sys ==="
  $ASR_PY $EV/transcribe_qwen3.py --wav_dir $ROOT/$sys/wavs --out $ROOT/$sys/hyps.jsonl 2>&1 | tail -1
  echo "=== metrics $sys ==="
  $PY $EV/metrics.py --wav_dir $ROOT/$sys/wavs --testset examples/libritts/cosyvoice3/exp/hidden_poc/eval30k/testset.jsonl \
    --campplus_onnx $CV3/campplus.onnx --hyps $ROOT/$sys/hyps.jsonl 2>&1 | tail -1
done
$PY $EV/paired_report.py --eval_root $ROOT --systems $SYSTEMS --baseline full --out $ROOT/REPORT.md
echo done
