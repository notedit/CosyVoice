# FlowTTS-GRPO: RL Fine-Tuning of the CosyVoice3 Flow-Matching Module

Reproduction of **FlowTTS-GRPO: Online Reinforcement Learning with Multi-Objective
Reward Optimization for Flow-Matching Based Text-to-Speech**
([arXiv:2606.23190](https://arxiv.org/abs/2606.23190), Tongyi Lab) on CosyVoice3.

The paper fine-tunes **only the flow-matching (FM) module** of CosyVoice 3.0 with
online GRPO — the LLM, speech tokenizer and vocoder stay frozen — and reports on
Seed-TTS-Eval-zh: speaker similarity SS1 0.777 → **0.804** (surpassing closed-source
Seed-TTS), DNSMOS P.835 3.353 → **3.536**, with CER essentially unchanged.

This recipe is complementary to [`examples/grpo/cosyvoice2`](../examples/grpo/cosyvoice2), which
applies GRPO to the **LLM** (improving CER). The paper's ablations show the two
levels decouple: LM-level RL fixes intelligibility, FM-level RL improves timbre
similarity and perceptual quality, and their gains stack.

## How it works

### 1. ODE → SDE conversion (the key enabler)

CosyVoice3's FM decoder samples mel spectrograms with a deterministic Euler ODE
(`cosyvoice/flow/flow_matching.py`), which has no likelihood — so policy-gradient
RL cannot be applied directly. Following the paper (Eq. 6–7), the ODE is converted
into an SDE that keeps the same marginals but injects Gaussian exploration noise:

```
x_mean   = x_t + [ v_θ + σ_t²/(2(1−t)) · (−x_t + t·v_θ) ] · Δt
x_{t+Δt} = x_mean + σ_t·√Δt · ε,      ε ~ N(0, I)
σ_t      = a·√((1−t)/t)               (noise level a = 0.5)
```

Each stochastic transition is a diagonal Gaussian with closed-form log-density
(Eq. 11), which makes the GRPO likelihood ratio computable. Implementation:
[`flow_grpo/sde_solver.py`](flow_grpo/sde_solver.py). Two CosyVoice3-specific
details handled there:

* the cosine t-scheduler (`t_span = 1 − cos(u·π/2)`) gives non-uniform Δt, so all
  formulas use the actual per-step Δt;
* σ_t diverges at t = 0, so the stochastic window never includes step 0.

### 2. Windowed stochasticity + no CFG during rollout

Only a contiguous window of `ws = 2` of the 10 Euler steps is run as SDE and
optimized (window start `S_min` sampled uniformly from [1, 3] per rollout); the
remaining steps stay deterministic ODE. Classifier-free guidance is **disabled
during RL rollouts** (paper Sec. 4.4.4: more exploration, faster reward growth);
standard inference keeps its usual CFG (rate 0.7). Implementation:
[`flow_grpo/policy.py`](flow_grpo/policy.py), which mirrors
`CausalMaskedDiffWithDiT.inference` conditioning but uses fresh `N(0, I)` initial
noise per sample instead of the fixed `rand_noise` buffer.

### 3. GRPO with multi-objective rewards

For each prompt, the frozen LLM generates the semantic tokens **once**; the FM
policy then draws `G = 8` samples sharing those conditions, so within-group reward
differences are attributable to the FM actions alone. Rewards (paper Sec. 3.3):

| Component | Model | Score |
|---|---|---|
| `R_ss` speaker similarity | ERes2Net (3D-Speaker) | cosine vs. prompt audio |
| `R_asr` intelligibility | Paraformer (zh) / Whisper-large-v3 (en) | 1 − CER / 1 − WER |
| `R_mos` perceptual quality | DNSMOS P.835 | OVRL score (16 kHz) |

combined with within-group std normalization (weighted scheme, the paper's best):

```
R = λ_ss·R_ss/std_g(R_ss) + λ_asr·R_asr/std_g(R_asr) + λ_mos·R_mos/std_g(R_mos)
λ = (1.0, 1.0, 0.4)
```

Group-relative advantages `(R − mean_g)/std_g` (zero-std groups dropped), PPO-clip
surrogate, and a closed-form Gaussian KL to the reference policy (adapters
disabled). Implementation: [`flow_grpo/grpo_loss.py`](flow_grpo/grpo_loss.py),
[`rewards/`](rewards/).

### 4. LoRA on the DiT

Only LoRA adapters on the DiT estimator are trained: rank 32, alpha 64, targeting
every attention q/k/v/out projection and both feed-forward linears of all 22
blocks — **10.09 M trainable parameters, exactly matching the paper** (verified
against the real CosyVoice3 DiT). The self-contained implementation
([`flow_grpo/lora.py`](flow_grpo/lora.py)) supports adapter toggling (reference
policy = adapters off) and merging back into a stock-format `flow.pt`
([`export_merged.py`](export_merged.py)), so the RL-tuned model loads with the
unmodified `CosyVoice3` class, including CFG/streaming/TRT paths.

## Usage

Hardware: the paper uses 8 GPUs (batch 512 samples/iteration = 8 GPU × 8 prompts
× 8 samples, ~9.5k steps). A single GPU works for pipeline bring-up with
`prompts_per_iter` reduced accordingly. This recipe was developed without GPUs:
the RL math is CPU-unit-tested (`tests/`), but full training has **not** been run —
expect to tune throughput knobs.

```bash
pip install -r requirements.txt         # on top of the base CosyVoice requirements

bash run.sh -1 -1     # download Fun-CosyVoice3-0.5B + DNSMOS onnx
bash run.sh 0 1       # prompt list (data/raw.jsonl -> train/val) + hard cases
bash run.sh 2 2       # GRPO training (torchrun, LoRA only)
bash run.sh 3 3       # merge LoRA -> exp/fm_grpo/flow_grpo.pt
bash run.sh 4 4       # evaluate baseline vs GRPO
bash run.sh 5 5       # CPU unit tests
```

Training data: any jsonl with `{"text", "prompt_text", "prompt_wav"}` per line.
The paper uses ~40k prompts from WenetSpeech4TTS Premium (zh) + LibriTTS-960 (en),
plus 20k hard-case prompts synthesized by repetition augmentation
([`make_hard_cases.py`](make_hard_cases.py): LWR / SMR / GSR). A few hundred
prompts are enough to validate the pipeline.

All hyperparameters live in [`conf/grpo.yaml`](conf/grpo.yaml), each annotated
`[paper]` (value from the paper) or `[default]` (unspecified there — notably the
PPO clip ε, KL weight β, and the exact reward checkpoints).

## Evaluation

[`evaluate.py`](evaluate.py) reports CER (zh) / WER (en), ERes2Net speaker
similarity (the paper's SS2) and DNSMOS P.835 OVRL on a Seed-TTS-Eval style jsonl.
For paper-comparable SS1 (WavLM) numbers, run the official
[seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) toolkit on the
saved wavs. Paper reference results (CosyVoice 3.0, Seed-TTS-Eval-zh):

| Metric | Baseline | + FM-GRPO |
|---|---|---|
| SS1 (WavLM) | 0.777 | **0.804** |
| SS2 (ERes2Net) | 0.830 | **0.859** |
| CER (Paraformer) | 1.20 | 1.26 |
| DNSMOS P.835 OVRL | 3.353 | **3.536** |

## Files

```
flow_grpo/sde_solver.py   ODE→SDE conversion, Gaussian log-prob, window sampling
flow_grpo/policy.py       rollout with per-step transitions + log-prob recompute
flow_grpo/grpo_loss.py    group advantages, PPO-clip, closed-form Gaussian KL
flow_grpo/lora.py         self-contained LoRA (inject / toggle / merge / save)
flow_grpo/buffer.py       per-group rollout storage
rewards/                  ERes2Net + Paraformer/Whisper + DNSMOS + composer
rollout.py                frontend + frozen LLM + FM policy + HiFT vocoder
train_grpo.py             main trainer (torchrun DDP, LoRA-grad all-reduce)
prepare_data.py           jsonl prompt-list preparation and filtering
make_hard_cases.py        LWR/SMR/GSR repetition augmentation (paper Sec. 3.4)
export_merged.py          LoRA merge -> stock-format flow.pt
evaluate.py               CER/WER + speaker similarity + DNSMOS evaluation
tests/                    CPU unit tests for all RL math (no pretrained model)
```

## Deviations & open items

* **Log-prob reduction**: per-element mean (official Flow-GRPO convention) instead
  of the paper's literal sum, keeping PPO ratios in a sane range; `sum` is available
  via `sampling.logprob_reduction`.
* **KL**: closed-form Gaussian KL (both kernels share σ), weight β not given in the
  paper — default 1e-3, set 0 to disable.
* **Unspecified in the paper**: PPO clip ε (default 0.2), exact ERes2Net/DNSMOS
  checkpoints, per-GPU batch split. All configurable.
* Rewards run in-process on each rank; for higher throughput move them to a
  dedicated GPU/server (see `examples/grpo/cosyvoice2/token2wav_asr_server.py`
  for a Triton-based pattern).
* The paper additionally evaluates on F5-TTS and CV3-Eval (9 languages); this
  recipe covers the CosyVoice3 path only.
