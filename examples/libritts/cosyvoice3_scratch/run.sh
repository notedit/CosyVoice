#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
#
# CosyVoice3 LM -- train FROM SCRATCH on your own data.
#
# What "from scratch" means here (confirmed design):
#   * The Qwen transformer body is WARM-STARTED from the pretrained text LLM
#     (CosyVoice-BlankEN = Qwen2.5-0.5B) inside Qwen2Encoder.from_pretrained(),
#   * speech_embedding + llm_decoder start from RANDOM init and are trained from zero,
#   * we DO NOT pass --checkpoint, so the released llm.pt is never loaded,
#   * we train ONLY the llm (flow / hift are reused from the released model at inference).
#
# This recipe assumes you have ALREADY extracted features offline, i.e. each
# data/<set> dir already contains:
#   wav.scp  text  utt2spk  spk2utt  utt2embedding.pt  spk2embedding.pt  utt2speech_token.pt
# so the default entry point is stage 3 (pack parquet). Stages 0-2 are kept as a
# reference for the raw-audio path only.
. ./path.sh || exit 1;

stage=3
stop_stage=6

# ===================== EDIT THESE =====================
# Released model dir: provides CosyVoice-BlankEN (Qwen backbone + tokenizer) and the *.onnx files.
pretrained_model_dir=../../../pretrained_models/Fun-CosyVoice3-0.5B
# CV3 REQUIRES a system/instruct prompt per utterance. Keep <|endofprompt|> at the end.
system_prompt="You are a helpful assistant.<|endofprompt|>"
# Space-separated subset dir names under data/ (your own splits).
train_sets="train"
dev_sets="dev"
# ======================================================

# ---------- Stages 0-2: raw-audio path only (SKIP if features are already extracted) ----------
# stage 0: build wav.scp/text/utt2spk/spk2utt (+ instruct). local/prepare_data.py is LibriTTS-specific;
#          for custom data, produce these files yourself (the instruct file is (re)generated in stage 3).
#   python local/prepare_data.py --src_dir <raw> --des_dir data/<set> --instruct "$system_prompt"
# stage 1: speaker embedding  -> utt2embedding.pt / spk2embedding.pt
#   ../../../tools/extract_embedding.py --dir data/<set> --onnx_path $pretrained_model_dir/campplus.onnx
# stage 2: discrete speech token -> utt2speech_token.pt
#   ../../../tools/extract_speech_token.py --dir data/<set> --onnx_path $pretrained_model_dir/speech_tokenizer_v3.onnx

# ---------- Stage 3: ensure CV3 instruct file exists, then pack parquet ----------
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: ensure instruct file + pack parquet (must include the 'instruct' column for CV3)"
  for x in $train_sets $dev_sets; do
    # CV3 needs a per-utt instruct/system prompt; (re)generate from the text file if missing.
    # make_parquet_list.py reads data/<set>/instruct and writes it into the 'instruct' parquet column.
    if [ ! -f data/$x/instruct ]; then
      echo "  generating data/$x/instruct : '$system_prompt'"
      awk -v p="$system_prompt" '{print $1" "p}' data/$x/text > data/$x/instruct
    fi
    mkdir -p data/$x/parquet
    ../../../tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 32 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi

# ---------- Stage 4: validate packed parquet BEFORE training (gate) ----------
# Catches the #1 from-scratch crash (missing/empty instruct) plus speech_token
# range / embedding-dim issues, so a bad pack fails here in seconds instead of
# minutes into the multi-GPU job. run.sh has no `set -e`, so we gate with `|| exit 1`.
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: validate parquet (instruct present + speech_token in [0,6561) + embedding dim 192)"
  cat $(for x in $train_sets; do echo data/$x/parquet/data.list; done) > data/train.data.list
  cat $(for x in $dev_sets;   do echo data/$x/parquet/data.list; done) > data/dev.data.list
  # samples the first 20 shards of each list (issues are systemic); use --num_parquet 0 for all.
  python validate_parquet.py --data_list data/train.data.list --num_parquet 20 || exit 1
  python validate_parquet.py --data_list data/dev.data.list   --num_parquet 20 || exit 1
fi

# ---------- Stage 5: train the LM from scratch ----------
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=8
prefetch=200
train_engine=torch_ddp   # torch_ddp is enough for 0.5B on 8xH200; switch to deepspeed for ZeRO-2
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: train CosyVoice3 LM FROM SCRATCH on $num_gpus GPU(s)"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cat $(for x in $train_sets; do echo data/$x/parquet/data.list; done) > data/train.data.list
  cat $(for x in $dev_sets;   do echo data/$x/parquet/data.list; done) > data/dev.data.list
  # KEY DIFFERENCES vs the released SFT recipe:
  #   * only --model llm (no flow/hifigan loop)
  #   * NO --checkpoint  -> from-scratch speech head; Qwen body warm-starts via --qwen_pretrain_path
  #   * conf/cosyvoice3.yaml uses the from-scratch train_conf (lr 1e-4 / warmuplr / longer warmup)
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
    ../../../cosyvoice/bin/train.py \
    --train_engine $train_engine \
    --config conf/cosyvoice3.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
    --onnx_path $pretrained_model_dir \
    --model llm \
    --model_dir `pwd`/exp/cosyvoice3_scratch/llm/$train_engine \
    --tensorboard_dir `pwd`/tensorboard/cosyvoice3_scratch/llm/$train_engine \
    --ddp.dist_backend $dist_backend \
    --num_workers ${num_workers} \
    --prefetch ${prefetch} \
    --pin_memory \
    --use_amp \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer
fi

# ---------- Stage 6: average the best checkpoints ----------
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  decode_checkpoint=`pwd`/exp/cosyvoice3_scratch/llm/$train_engine/llm.pt
  echo "Stage 6: averaging $average_num best-by-cv checkpoints -> $decode_checkpoint"
  python ../../../cosyvoice/bin/average_model.py \
    --dst_model $decode_checkpoint \
    --src_path `pwd`/exp/cosyvoice3_scratch/llm/$train_engine \
    --num ${average_num} \
    --val_best
fi

# ---------- Stage 7 (optional): LM-only sanity check ----------
# NOT run by default (stop_stage=6). Bump stop_stage to 7, or just call the script
# directly. Needs no onnx and no vocoder -> finishes in seconds. It loads only the
# llm, runs one autoregressive decode and checks the model emits speech tokens in
# [0,6561) and stops at a stop token. Works on any mid-training epoch_*_step_*.pt too.
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Stage 7: LM-only sanity check (autoregressive token generation + stop)"
  python sanity_check_lm.py \
    --config conf/cosyvoice3.yaml \
    --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
    --llm_pt `pwd`/exp/cosyvoice3_scratch/llm/$train_engine/llm.pt \
    --prompt_text "$system_prompt" \
    --text "今天天气真不错，我们出去走走吧。" || exit 1
fi
