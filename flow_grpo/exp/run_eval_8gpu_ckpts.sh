#!/bin/bash
# Evaluates the two latest 8-GPU-run checkpoints (step5600, step6000) on the
# AISHELL-3 val split, comparable with exp/eval_aishell3_baseline.
set -e
cd /opt/dlami/nvme/leolxliu/CosyVoice/flow_grpo
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=..:../third_party/Matcha-TTS
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
PY=../venv_fm_grpo/bin/python

for step in 5600 6000; do
    echo "=== MERGE step${step} ==="
    $PY export_merged.py --model_dir ../pretrained_models/Fun-CosyVoice3-0.5B \
        --lora_ckpt exp/fm_grpo_8gpu/lora_step${step}.pt \
        --output exp/fm_grpo_8gpu/flow_step${step}.pt
    echo "=== EVAL step${step} start ==="
    $PY evaluate.py --model_dir ../pretrained_models/Fun-CosyVoice3-0.5B \
        --flow_ckpt exp/fm_grpo_8gpu/flow_step${step}.pt \
        --test_data data/aishell3_val.jsonl \
        --output_dir exp/eval_aishell3_8gpu_step${step}
    echo "=== EVAL step${step} done ==="
done
echo "ALL_EVALS_DONE"
