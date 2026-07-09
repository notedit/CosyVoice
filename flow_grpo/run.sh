#!/usr/bin/env bash
# FlowTTS-GRPO recipe for the CosyVoice3 flow-matching module (arXiv:2606.23190).
set -eou pipefail

stage=-1
stop_stage=5

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

export PYTHONPATH=..:../third_party/Matcha-TTS:$PYTHONPATH

model_dir=../pretrained_models/Fun-CosyVoice3-0.5B
exp_dir=exp/fm_grpo
num_gpus=8

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "stage -1: download Fun-CosyVoice3-0.5B and the DNSMOS onnx model"
  python3 -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$model_dir')"
  mkdir -p models
  wget -O models/sig_bak_ovr.onnx \
    https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "stage 0: prepare prompt list from your jsonl metadata"
  # Provide data/raw.jsonl with lines: {"text": ..., "prompt_text": ..., "prompt_wav": ...}
  # The paper uses ~40k prompts from WenetSpeech4TTS Premium (zh) + LibriTTS-960 (en).
  python3 prepare_data.py --input data/raw.jsonl \
    --output data/train.jsonl --val_output data/val.jsonl --val_size 200
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "stage 1: synthesize hard-case (repetition) prompts"
  python3 make_hard_cases.py --input data/train.jsonl --output data/train_hard.jsonl --num_samples 20000
  cat data/train.jsonl data/train_hard.jsonl > data/train_all.jsonl
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "stage 2: GRPO training (only the DiT LoRA adapters are updated)"
  torchrun --nproc_per_node $num_gpus train_grpo.py \
    --config conf/grpo.yaml \
    --model_dir $model_dir \
    --train_data data/train_all.jsonl \
    --output_dir $exp_dir
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "stage 3: merge LoRA into a stock-format flow.pt"
  python3 export_merged.py --model_dir $model_dir \
    --lora_ckpt $exp_dir/lora_last.pt --output $exp_dir/flow_grpo.pt
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "stage 4: evaluate baseline vs GRPO on a Seed-TTS-Eval style jsonl"
  python3 evaluate.py --model_dir $model_dir \
    --test_data data/test_zh.jsonl --output_dir $exp_dir/eval_baseline
  python3 evaluate.py --model_dir $model_dir --flow_ckpt $exp_dir/flow_grpo.pt \
    --test_data data/test_zh.jsonl --output_dir $exp_dir/eval_grpo
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "stage 5: CPU unit tests for the RL math (no GPU / pretrained model needed)"
  python3 -m pytest tests -q
fi
