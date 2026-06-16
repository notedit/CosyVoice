# CosyVoice3 LM — Train From Scratch

A self-contained recipe for training the **CosyVoice3 LM** (text + instruct → speech-token
autoregressive model) **from scratch** on your own data. It does **not** modify the official
`examples/libritts/cosyvoice3/` recipe.

## What "from scratch" means here

- The Qwen transformer body is **warm-started** from the pretrained text LLM
  (`CosyVoice-BlankEN` = Qwen2.5-0.5B) inside `Qwen2Encoder.from_pretrained()`
  (`cosyvoice/llm/llm.py:226`). This is the recommended setup and needs **no code change**.
- `speech_embedding` and `llm_decoder` (the speech-token head, vocab **6761**) start from
  **random init** and are trained from zero (`cosyvoice/llm/llm.py:687,696`).
- We **do not** pass `--checkpoint`, so the released `llm.pt` is never loaded
  (`cosyvoice/bin/train.py:135`).
- We train **only the `llm`**; `flow.pt` / `hift.pt` are reused from the released model at inference.

## Prerequisites

1. Download `Fun-CosyVoice3-0.5B` into `../../../pretrained_models/Fun-CosyVoice3-0.5B/`.
   Used here: `CosyVoice-BlankEN/` (Qwen backbone **and** the `cosyvoice3` tokenizer),
   `campplus.onnx`, `speech_tokenizer_v3.onnx` (+ `*.batch.onnx`). `flow.pt`/`hift.pt` are for the
   inference check only.
2. Features already extracted offline. Each `data/<set>/` should contain:
   `wav.scp text utt2spk spk2utt utt2embedding.pt spk2embedding.pt utt2speech_token.pt`.
   Speech tokens **must** come from `speech_tokenizer_v3` (token ids in `0..6560`).

## How it differs from the released SFT recipe (`conf/cosyvoice3.yaml`)

Only `train_conf` + batching changed; the **model architecture section is identical** (do not touch
`CosyVoice3LM`, `speech_token_size: 6561`, the 6761 vocab, `mix_ratio: [5,15]`, `version: cosyvoice3`).

| Field | Released (SFT) | From-scratch (here) | Why |
|---|---|---|---|
| `lr` | `1e-5` | `1.0e-4` | random speech head needs a larger lr |
| `scheduler` | `constantlr` | `warmuplr` | warmup→decay protects the warm-started backbone |
| `warmup_steps` | `2500` | `10000` | large data + 8 GPUs |
| `max_epoch` | `200` | `6` | 100k h → convergence is step-driven |
| `save_per_step` | `-1` | `5000` | checkpoint + cv periodically, not per-epoch |
| `max_frames_in_batch` | `2000` | `15000` | H200 141GB headroom; tune by OOM |
| `shuffle_size`/`sort_size` | `1000`/`500` | `5000`/`1000` | better global shuffling over many shards |

`--use_amp` selects **bf16** automatically (`cosyvoice/utils/train_utils.py:74`), ideal on H200.

## Run

```bash
# edit pretrained_model_dir / system_prompt / train_sets / dev_sets at the top of run.sh first
bash run.sh                 # default: stage 3 (pack) -> 4 (validate) -> 5 (train) -> 6 (average)
```

- **Stage 3** (re)generates `data/<set>/instruct` from `text` and packs parquet (with the `instruct` column).
- **Stage 4** validates the packed parquet (`validate_parquet.py`) and **aborts before training** if anything is wrong — see [Helper scripts](#helper-scripts).
- **Stage 5** trains the LM from scratch on 8 GPUs (`--model llm`, **no** `--checkpoint`).
- **Stage 6** averages the best-by-cv checkpoints into `exp/cosyvoice3_scratch/llm/<engine>/llm.pt`.
- **Stage 7** (opt-in, `stop_stage=7`) runs the LM-only sanity check (`sanity_check_lm.py`).

Smoke-test first: point `train_sets` at a ~100–500 h subset, run a few hundred steps, confirm
`loss` drops and `acc` rises (TensorBoard), then scale to the full corpus.

## Gotchas (in order of likelihood)

1. **`instruct` column is mandatory.** `padding` only emits `batch['instruct_token']` when **every**
   sample in the batch has it (`cosyvoice/dataset/processor.py:403-406`); otherwise
   `CosyVoice3LM.forward` raises `KeyError` at `cosyvoice/llm/llm.py:388`. Verify:
   ```python
   import pandas as pd; print(pd.read_parquet('data/<set>/parquet/xxx.parquet').columns.tolist())
   # expect 'instruct', 'text', 'speech_token', and 'utt_embedding'/'spk_embedding'
   ```
   If missing, re-run stage 3 — no need to re-extract features. Stage 4 (`validate_parquet.py`)
   checks this automatically and fails the run before any GPU time is spent.
2. **Speech tokenizer must be v3** (vocab/embedding size is locked to 6761). Do **not** load a CV2 `llm.pt`.
3. **Tokenizer version must be `cosyvoice3`** (already set in the yaml).
4. **`--onnx_path`** points at the released dir: because the env var gets set, `online_feature`
   becomes `True` (`cosyvoice/utils/onnx.py:50-54`) and the onnx extractors are *loaded* but, since
   your parquet already has `speech_token`/embedding, **never actually run**. Make sure
   `campplus.onnx` and `speech_tokenizer_v3.batch.onnx` exist in that dir.

## End-to-end verification

Assemble an inference model dir = your trained `llm.pt` + released `flow.pt`/`hift.pt`/onnx +
`CosyVoice-BlankEN/` + `cosyvoice3.yaml` (use the released model-structure yaml), then:

```python
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice3
m = CosyVoice3('my_cv3_model')
for i, out in enumerate(m.inference_zero_shot(
        '要合成的目标文本',
        'You are a helpful assistant.<|endofprompt|>提示音频对应的文本',
        prompt_16k_wav)):
    torchaudio.save(f'out_{i}.wav', out['tts_speech'], m.sample_rate)
```

## Helper scripts

Two standalone scripts (no changes to the training code) make the pipeline safer and give a
fast feedback loop:

### `validate_parquet.py` — pre-flight parquet check (stage 4)

Validates packed shards before you burn GPU time. Exits non-zero (aborting `run.sh`) if any of:
the `instruct` column is missing/empty or lacks `<|endofprompt|>`; `speech_token` ids fall outside
`[0, 6561)` or a row is empty; `utt_embedding`/`spk_embedding` dimension ≠ 192.

```bash
python validate_parquet.py --data_list data/train.data.list --num_parquet 20   # sample 20 shards
python validate_parquet.py --dir data/train/parquet --num_parquet 0            # check all shards
python validate_parquet.py --parquet data/train/parquet/parquet_000000000.tar  # one shard
# --no_require_instruct to skip the CV3 instruct checks (e.g. CV2-style data)
```

### `sanity_check_lm.py` — LM-only generation check (stage 7)

Loads **only** the LM (flow/hift overridden to `None`, no onnx needed) and runs one autoregressive
decode to confirm the model emits speech tokens in `[0, 6561)` and stops at a stop token. Seconds,
not minutes — works on any mid-training `epoch_*_step_*.pt` as well as the averaged `llm.pt`.

```bash
python sanity_check_lm.py \
    --config conf/cosyvoice3.yaml \
    --qwen_pretrain_path ../../../pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
    --llm_pt exp/cosyvoice3_scratch/llm/torch_ddp/llm.pt \
    --prompt_text "You are a helpful assistant.<|endofprompt|>" \
    --text "今天天气真不错，我们出去走走吧。"
```

A `PASS` means the LM is wired up and generating/terminating correctly; for audio quality use the
end-to-end check above.
