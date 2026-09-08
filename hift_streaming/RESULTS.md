# Chunk-level streaming vocoder for CosyVoice3 (CausalHiFTGenerator)

Date: 2026-09-04 (CUDA graph section 2026-09-05). Branch `claude/hidden-injection-poc` (uncommitted). GPU: H100 (idle), torch 2.5.1+cu124,
model `/opt/dlami/nvme/leolxliu/vocoder/data/pretrained/CosyVoice3-0.5B`.

> **Note (2026-09-08): the CUDA graph path has been removed from the branch.** The sections below
> that describe and measure it are kept as a record of what was built and what it bought; the code
> is in `git show e0fb3fb -- cosyvoice/hifigan/streaming.py` and the reasoning for parking it is in
> "Rolling it out" further down. What ships is the incremental vocoder, `hift_mode=incremental`.

## Problem

`CosyVoice3Model.token2wav` streamed the vocoder by concatenating **all** mel frames so far and
re-running `CausalHiFTGenerator.inference(finalize=False)` on the whole prefix every chunk
(`hift_cache_dict[uuid]['mel']` + `speech_offset`). The vocoder is fully causal, so the result is
correct, but the cost of an N-chunk utterance is O(N * T) and the per-chunk latency grows with the
utterance. The Triton runtime (`runtime/triton_trtllm/token2wav_cosyvoice3.py`) does the same.

## What was done

`cosyvoice/hifigan/streaming.py` + `CausalHiFTGenerator.new_stream_state() / inference_chunk()`:
a true incremental path that carries, per utterance,

- the left context of every causal conv (`CausalConv1d.cache`, `CausalConv1dUpsample` in the
  upsampled domain, an input buffer for the strided `CausalConv1dDownSample`),
- the right look-ahead frames of the f0 predictor (3 mel frames) and `conv_pre` (4 frames),
- the NSF phase accumulator (float64, wrapped mod 1) and the offset into the fixed noise buffers,
- the source-STFT tail (reflect padding reproduced at the stream start/end),
- the iSTFT overlap-add carry (12 samples) and the `n_fft // 2` center trim.

Each chunk therefore costs only its own frames plus a fixed 7-frame look-ahead. Concatenating the
chunk outputs reproduces `inference(full_mel)` sample-for-sample (same length, see below).

Integration: `CosyVoice3Model(use_hift_cache=True)` (default) routes every streaming chunk and the
finalize call through `inference_chunk`; `use_hift_cache=False` keeps the old prefix-recompute path.
`CosyVoice3Model.load` now folds weight norm (`fold_weight_norm`) which saves one weight
recomputation per conv per call (~2 ms/chunk). `inference_chunk(finalize_pad_multiple=50)` pads the
last chunk to a fixed set of shapes (see cuDNN note).

Scripts (all under `hift_streaming/`): `test_equivalence.py` (chunked vs full, timing),
`make_mels.py` (official flow -> mel for the 60 eval30k utterances), `bench_schedule.py` (vocoder
timing with the exact `tts(stream=True)` chunk schedule, dumps wavs), `bench_long.py` (30/60/120 s),
`run_quality.sh` (CER / SS / DNSMOS via the eval30k pipeline), `test_model_e2e.py`
(`CosyVoice3Model.tts` end to end with cached tokens), `probe_numerics.py`, `probe_latency.py`,
`summarize.py`. Outputs in `hift_streaming/exp/` (git-ignored).

## Numerical equivalence

Two things in the *reference* limit how closely anything can match it:

1. cuDNN runs the fp32 convs in TF32 by default, and picks different kernels for different input
   lengths. `decode()` on the same input with a different length already differs by ~2e-2 max abs,
   SNR ~45 dB. With TF32 disabled the reference is bit-exact across lengths.
2. `SineGen2` (causal) accumulates the NSF phase as `2*pi*480*cumsum(rad)` in float32. The phase
   reaches ~1e6 rad within 10-20 s, where a float32 ulp is 0.1-0.25 rad, so the reference's own
   sines are rounding-limited on long inputs. The streaming path accumulates in float64 mod 1
   instead (exact for any length). The prefix-recompute path inherits the float32 behaviour.

| setting (60 real utterances, 50-frame chunks) | incremental vs full | prefix-recompute vs full |
|---|---|---|
| TF32 off, reference phase in float64 (`--no_tf32 --f64_phase_ref`) | SNR >= 100.1 dB on all 60, max abs 1e-5 | 103-117 dB |
| production defaults (TF32 on, stock reference) | SNR median 45 dB (min 31) | SNR median 53 dB (min 39) |

So the streaming logic is exact; the residual under production settings is TF32 kernel noise plus the
reference's float32 phase error. Output length always equals the full-inference length (also for an
empty final chunk and for 1-frame chunks; `test_model_e2e.py`).

## Quality (60 eval30k utterances, same mel for all systems, `run_quality.sh`)

| system | CER (Qwen3-ASR) | SS (CAM++) | DNSMOS OVRL |
|---|---|---|---|
| full (non-streaming) | 0.0022 | 0.8491 | 3.386 |
| prefix-recompute (old streaming) | 0.0022 | 0.8491 | 3.385 |
| incremental (this work) | 0.0022 | 0.8491 | 3.385 |

Paired deltas vs full: dCER +0.0000 [+0.0000, +0.0000], dSS +0.0000 [-0.0000, +0.0001],
dDNSMOS -0.0003 [-0.0009, +0.0003] (95% bootstrap CI). No measurable degradation.

## Performance

The vocoder is launch-bound, not FLOP-bound: a steady-state 50-frame chunk (1 s of audio) is ~13 ms
wall time with ~4 ms of GPU time, and the cost per chunk is nearly independent of the chunk size
(13 ms at 50 frames, 14.5 ms at 100, 12.7 ms at 200). Two overheads dominate everything else:

- **cuDNN plan building**: every never-seen input shape costs ~0.4 ms per conv, i.e. ~35-50 ms per
  vocoder call (85 convs). The prefix-recompute path sees a new shape on *every* chunk of every
  utterance; the incremental path only on the first chunk (prompt-dependent size, 25 variants) and
  the final chunk. `finalize_pad_multiple=50` makes the final chunk one of 5 shapes.
- process warm-up: the first vocoder call of a process costs 400-700 ms (cuDNN init + plans).

### Realistic schedule, short utterances (60 utterances, median 6.5 s, 3 chunks; `bench_schedule.py --fold --pad_multiple 50`)

cold = first pass over that utterance (production-like), warm = same utterance again (all plans cached).

| system | pass | total vocoder ms (median) | first chunk | later chunks (median) | last chunk | max chunk |
|---|---|---|---|---|---|---|
| prefix-recompute | cold | 109 | 16 | 52 | 53 | 54 |
| prefix-recompute | warm | 49 | 16 | 16 | 16 | 16 |
| incremental | cold | 49 | 16 | 15 | 15 | 16 |
| incremental | warm | 45 | 15 | 14 | 15 | 15 |
| full (1 call) | - | 16 | | | | |

Without weight-norm folding / final-chunk padding (`bench_v2`): recompute cold 165 ms, incremental
cold 89 ms (54 ms of it is the new-shape final chunk).

### Long inputs (real mels concatenated; `bench_long.py --fold`, incremental with pad 50)

| input | chunks | recompute total ms (cold/warm) | recompute chunk ms median / max (cold) | incremental total ms (cold/warm) | incremental chunk ms median / max (cold) | speedup (cold total) |
|---|---|---|---|---|---|---|
| 30 s | 9 | 486 / 179 | 55 / 65 | 248 / 121 | 13 / 53 | 2.0x |
| 60 s | 17 | 758 / 551 | 52 / 78 | 261 / 242 | 13 / 50 | 2.9x |
| 120 s | 32 | 2258 / 1736 | 89 / 128 | 434 / 425 | 13 / 16 | 5.2x |

The prefix-recompute per-chunk latency keeps growing with the utterance (128 ms at 2 min and rising),
the incremental one is flat at 13-16 ms. Per second of generated audio the incremental vocoder costs
~4 ms in steady state (200-frame chunks), i.e. RTF ~0.004.

## CUDA graph replay (2026-09-05, removed from the branch 2026-09-08)

The remaining ~13 ms per chunk was Python + kernel-launch overhead (~1100 launches for ~4 ms of
GPU work), so steady-state chunks are now replayed from a captured CUDA graph
(`ChunkGraph` in `streaming.py`, `inference_chunk(use_graph=True)`):

- After the first chunk every state tensor has a fixed shape for a given chunk length (3 pending
  f0 frames, 4 pending conv_pre frames, fixed conv caches, 1924-sample source tail, 12-sample
  OLA carry), so one graph per chunk length is captured lazily (~0.4 s each; 2 warm-up runs on a
  side stream, then capture). The first chunk and the finalize chunk stay eager.
- The state is copied into the graph's static buffers before replay and back out after, so one
  graph serves any number of concurrent utterances (replays serialized by a lock). The SineGen2
  noise slice, indexed by absolute position on the host, is a static input refilled from pinned
  memory before each replay (`sine_waves` gets pinned once, 260 MB host memory).
- The STFT window is cached on device (a host->device copy per call would break capture).
- The zero-padded finalize (`finalize_pad_multiple`) was made exact: the source of the padded
  frames is replaced by the reflect pad the full-sequence STFT would apply, and the iSTFT only
  overlap-adds the frames the unpadded finalize would have. All 60 utterances now match the
  full reference at >= 93.9 dB SNR (max abs 1.2e-4) with graph replay + pad 50 + weight-norm
  folding, under TF32-off / float64-phase reference. Inside `CosyVoice3Model` the graphed and
  eager incremental paths are bit-identical (230 dB, `test_model_e2e.py`).
- `torch.compile` was not used: the captured graph already removes the launch overhead and keeps
  the eager kernels (hence exactness), without compile time or recompiles per shape.

Integration: `CosyVoice3Model(use_hift_graph=True)` (default; `CosyVoice3(use_hift_graph=)`).
`load()` captures the graphs for the schedule's steady-state chunk lengths (100 and 200 mel
frames) via `warmup_hift_graph()`, and `token2wav` calls `inference_chunk(..., finalize_pad_multiple=50, use_graph=True)`.

### Per-chunk latency (idle GPU, `probe_latency.py`, steady state)

| chunk length | eager incremental | graph replay |
|---|---|---|
| 50 frames (1 s audio) | 13.0 ms | 5.6 ms |
| 100 frames (2 s) | 14.5 ms | 6.4 ms |
| 200 frames (4 s) | 12.7 ms | 7.7 ms |

### Realistic schedule, 60 short utterances (`bench_schedule.py --fold --pad_multiple 50 --graph`)

| system | pass | total vocoder ms (median) | first chunk | later chunks (median) | last chunk | max chunk |
|---|---|---|---|---|---|---|
| prefix-recompute | cold | 121 | 18 | 52 | 53 | 54 |
| prefix-recompute | warm | 53 | 15 | 15 | 15 | 16 |
| incremental + graph | cold | 42 | 14 | 7 | 14 | 15 |
| incremental + graph | warm | 37 | 14 | 7 | 14 | 14 |

(The box was shared with other jobs during this run; the first/last chunks are eager.)

### Long inputs (`bench_long.py --fold --graph`)

| input | chunks | recompute total ms (cold/warm) | recompute chunk ms median / max (cold) | incremental+graph total ms (cold/warm) | incremental+graph chunk ms median / max (warm) | speedup (warm total) |
|---|---|---|---|---|---|---|
| 30 s | 9 | 563 / 222 | 58 / 80 | 775 / 80 | 8 / 14 | 2.8x |
| 60 s | 17 | 804 / 527 | 50 / 99 | 178 / 153 | 8 / 27 | 3.4x |
| 120 s | 32 | 2000 / 1733 | 84 / 113 | 259 / 254 | 8 / 14 | 6.8x |

The 775 ms cold total at 30 s contains the one-off 427 ms capture of the 200-frame graph (it is
done in `load()` in the model). Steady-state cost is ~2 ms per second of audio (RTF ~0.002).

### Quality with graph replay (60 utterances, `bench_graph`, paired vs full)

| system | CER (Qwen3-ASR) | SS (CAM++) | DNSMOS OVRL |
|---|---|---|---|
| full (non-streaming) | 0.0022 | 0.8491 | 3.385 |
| incremental + graph | 0.0011 | 0.8491 | 3.385 |

dCER +0.0011 [+0.0000, +0.0033], dSS +0.0000 [-0.0000, +0.0001], dDNSMOS +0.0000 [-0.0004, +0.0005].
The CER difference is a single item where the ASR wrote a homophone (已被 vs 以备); the ASR
transcripts of the *same* `full` wavs also differ in punctuation between the two evaluation runs,
so this is ASR noise, not a vocoder effect. SS and DNSMOS are unchanged to four decimals.

## Notes / follow-ups

- The first and the finalize chunk of each utterance are still eager (~15 ms each, plus a cuDNN
  plan build when the prompt-dependent first-chunk length is new to the process).
- `runtime/triton_trtllm/token2wav_cosyvoice3.py` and `model_repo_cosyvoice3/vocoder` still use the
  prefix-recompute loop and could switch to `inference_chunk` the same way.
- Only `CausalHiFTGenerator` (CosyVoice3) is covered. CosyVoice/CosyVoice2 use the non-causal
  `HiFTGenerator` with an 8/20-frame overlap + cross-fade, which is already linear-cost.
- `generator.py`'s `__main__` sets `cudnn.deterministic=True`; that forces dilated convs onto the
  slow native path (~50 ms per call), so do not benchmark with it.

## Port verification (branch `vocoder-optimization`, 2026-09-07)

The vocoder work above was ported into this repository (the unrelated flow kv-cache /
prompt-prefix work that shared the source branch was **not** ported). Re-run here with the same
weights (`CosyVoice3-0.5B`, `/opt/conda_envs/flow_tts`, torch 2.5.1+cu124) on an H200 shared with
other jobs, so latencies are noisier and higher than the idle-GPU numbers above; the ratios
reproduce. `third_party/Matcha-TTS` is checked out (`git submodule update --init`), the 60 eval
mels and their LLM caches were copied to `hift_streaming/exp/{mels,llm_cache}` (gitignored), so
every command below runs from this repository alone.

### Correctness

| check | result |
| --- | --- |
| `test_equivalence.py --n 60 --no_tf32 --f64_phase_ref` (eager, 50-frame chunks) | 60/60 length-exact, SNR 92.6–114.8 dB |
| same `--graph` (graph + pad 50 + fold) | 60/60 length-exact, SNR 93.9–113.9 dB |
| same, production default (TF32 on, float32 reference phase) | 60/60 length-exact, SNR 31.7–59.2 dB |
| `test_model_e2e.py $M $E/llm_cache 5` | identical lengths for graph / incremental / recompute on all 5 utterances, graph-vs-eager 227–238 dB, incremental-vs-recompute 43–55 dB; empty finalize chunk and 1-frame chunks length-exact |
| `bench_schedule.py --pad_multiple 50 --fold --graph` (60 utts) | incremental vs full: SNR median 45.9 dB (min 33.1); recompute vs full: 51.8 dB (min 39.2) |

The eager SNR floor here (92.6 dB) is a few dB below the 100.1 dB reported on the source machine —
`streaming.py` and its dependencies are byte-identical, so this is cuDNN algorithm / summation
order variation between the two GPUs; both clear the >= 90 dB bar, and the graph configuration
reproduces the source floor exactly (93.9 dB).

### Latency

`probe_latency.py` steady-state chunks (`FOLD=1 [GRAPH=1] CHUNK=c`), median of the second stream:

| chunk | eager | graph |
| --- | --- | --- |
| 50 frames (1 s) | 12.9 ms | 5.2 ms |
| 100 frames (2 s) | 14.7 ms | 5.9 ms |
| 200 frames (4 s) | 18.2 ms | 9.0 ms |

`bench_fixed_len.py $M 10 20`: 8 utterances per length with different prompt lengths (so every
call sees a new cuDNN shape, as in a real service). Median of the per-utterance total over 4
repetitions of the whole benchmark on an otherwise idle GPU, range across repetitions in brackets;
`FOLD=0` re-runs everything with weight norm left in place, i.e. the vocoder as it was before this
work:

| path | 10 s cold | 10 s warm | 20 s cold | 20 s warm |
| --- | --- | --- | --- | --- |
| prefix-recompute, before (`FOLD=0`) | 222 ms [214–227] | 66 ms | 273 ms [269–288] | 131 ms |
| prefix-recompute, now | 212 ms [204–221] | 62 ms | 265 ms [264–278] | 121 ms |
| incremental eager | 60 ms [53–64] | 58 ms | 101 ms [89–102] | 96 ms |
| incremental + graph | 44 ms [41–45] | 44 ms | 68 ms [64–68] | 67 ms |
| **non-stream 1 call, before (`FOLD=0`)** | **55 ms [54–68]** | 17 ms | **56 ms [56–57]** | 22 ms |
| **non-stream 1 call, now** | **53 ms [51–53]** | 16 ms | **54 ms [53–55]** | 22 ms |
| non-stream via 200-frame chunks + graph | 38 ms [34–38] | 38 ms | 61 ms [57–62] | 53 ms |

The non-streaming path is `CosyVoice3Model.tts(stream=False)`: one `token2wav(finalize=True)` call,
which is still the untouched `hift.inference(mel, finalize=True)` (with `stream=False` the
utterance's `hift_cache_dict` entry is `None`, so `token2wav` falls through to the original code).
**Before vs after for that path is 55 -> 53 ms at 10 s and 56 -> 54 ms at 20 s** — the ~2 ms comes
from `load()` folding weight norm, and nothing else changed: a single call has no chunks, so there
is nothing for a CUDA graph to replay. The graph only matters where the same shape recurs.

Cold is the production number for the non-streaming path too — every utterance has its own mel
length, so cuDNN builds a new plan on every call. That build is most of the cost (the same call on
a cached shape takes 16 / 22 ms), and it is exactly what fixed-shape graph chunks stop paying after
`load()`. Two consequences:

- Streaming is no longer much more expensive than one shot: incremental + graph costs 44 ms at
  10 s (below the 53 ms single call, whose plan build it avoids) and 68 ms at 20 s (a little above
  the 54 ms single call), where the old streaming path cost 4–5x a single call.
- An offline request can also be pushed through the incremental path in fixed 200-frame chunks
  (last row): 38 ms at 10 s beats the 53 ms single call, but at 20 s the single call wins
  (54 vs 61 ms), because the per-chunk launch overhead adds up faster than the one plan build it
  saves. So there is no reason to change the non-streaming path.

The non-streaming rows are measured last and on 1 / 3 extra mel frames (<= 0.1 % of the audio) so
that they build their own cuDNN plan instead of reusing the one the prefix-recompute row just built
for the exact utterance length.

#### Why a CUDA graph cannot speed up the non-streaming call

A graph only removes kernel-launch overhead, and it only exists per captured shape. Profiling one
call of each shape (`self_device_time_total` vs wall, weight norm folded):

| call | wall | GPU busy | kernels | launch overhead |
| --- | --- | --- | --- | --- |
| non-stream, one call, 500 frames (10 s) | 15.4 ms | 10.3 ms (67 %) | 1271 | 5.0 ms |
| non-stream, one call, 1000 frames (20 s) | 25.4 ms | 16.9 ms (66 %) | 1243 | 8.5 ms |
| streaming chunk, eager, 100 frames | 17.3 ms | 5.1 ms (29 %) | 1282 | 12.2 ms |
| streaming chunk, eager, 200 frames | 17.4 ms | 6.8 ms (39 %) | 1240 | 10.6 ms |

The vocoder issues ~1250 kernels per call whatever the input length, so the launch cost is a
per-call constant. Streaming pays it once per chunk — 4 to 7 times per utterance, each time for a
few ms of actual GPU work — which is why replacing those launches with a graph replay halves the
chunk. One non-streaming call pays it once, amortised over the whole utterance, where the GPU is
already busy two thirds of the time; even a perfect graph could remove at most 5–8 ms, and only if
one existed for that length. It would not: a graph is captured per shape (~0.4 s), and every
non-streaming request has its own mel length, so each capture would serve exactly one request. The
streaming schedule is the opposite case — 100 and 200-frame chunks recur for every utterance, so
two graphs captured in `load()` serve the whole service. This is also why the first and the
finalize chunk stay eager in the streaming path, and a non-streaming call is exactly that: a first
chunk that is also the finalize chunk.

`bench_schedule.py` on the 60 real utterances (median 6.5 s): total vocoder time 246 ms cold /
74 ms warm for prefix-recompute vs 43 / 41 ms for incremental + graph, against 33 ms for the
single non-streaming call. `bench_long.py` at 30 / 60 / 120 s: prefix-recompute 1355 / 2543 /
4083 ms with per-chunk latency growing with position, the incremental path 82 / 175 / 402 ms warm
with a flat 7–11 ms per steady-state chunk.

Except for the 10 / 20 s table (idle GPU, 4 repetitions), these ran on a node whose other GPUs were
busy, so they are noisier and higher than idle-GPU numbers; graph replay is launch-bound and
therefore the most sensitive to it (warm graph chunks occasionally at 20–35 ms instead of 7–9 ms).

### fp16

`CosyVoice3Model.token2wav` runs the vocoder inside `autocast(self.fp16)`. Under autocast the
incremental eager path works (5 utterances, length-exact, 28–42 dB vs the fp32 whole-utterance
reference, i.e. the usual half-precision loss, and on par with the prefix-recompute path at
32–41 dB), but **CUDA graph replay under autocast produced NaN or an illegal memory access** from
the second utterance on: autocast caches the fp16 weight copies per autocast region, and a graph
captured inside one replays against buffers freed when that region exits. `inference_chunk` now
falls back to the eager incremental path whenever `torch.is_autocast_enabled()` (fp16 graph output
then matches fp16 eager exactly), and `CosyVoice3Model` no longer captures graphs when `fp16=True`.
With the default `fp16=False` (`autocast(False)`) graph replay is unaffected.

### Quality (CER / SS / DNSMOS)

`bench_schedule.py --pad_multiple 50 --fold --graph` writes the `full` / `recompute` /
`incremental` wavs; they were scored with the eval30k toolchain from the source repository
(`transcribe_qwen3.py` in `venv_qwen3_asr`, then `metrics.py` and `paired_report.py`), report in
`hift_streaming/exp/bench_graph/REPORT.md`:

| system | CER | SS (CAM++) | DNSMOS OVRL | paired delta vs full (95% CI) |
| --- | --- | --- | --- | --- |
| full (whole utterance) | 0.0022 | 0.849088 | 3.38524 | - |
| prefix-recompute | 0.0011 | 0.849102 | 3.38524 | dCER +0.0011 [+0.0000, +0.0033], dSS +0.0000 [-0.0000, +0.0000], dDNSMOS -0.0000 [-0.0003, +0.0004] |
| incremental + graph | 0.0011 | 0.849109 | 3.38533 | dCER +0.0011 [+0.0000, +0.0033], dSS +0.0000 [-0.0000, +0.0001], dDNSMOS +0.0001 [-0.0004, +0.0006] |

SS and DNSMOS agree with the source run to four decimals (0.8491 / 3.385) and are identical
between the three systems; the CER difference is the same single homophone the source run saw
(both streaming systems transcribe it the same way, the reference `full` audio differently), well
inside the ASR noise band. So the 31–59 dB production-default SNR gap against whole-utterance
inference is not audible to any of the three metrics.

### Rolling it out: one flag, and what "unchanged" means

`CosyVoice3Model(..., hift_mode=...)` (also `CosyVoice3(..., hift_mode=...)`, and the
`COSYVOICE_HIFT_MODE` environment variable, which overrides the argument so a canary replica can
be flipped without a code change):

| hift_mode | streaming path | non-streaming path | weight norm |
| --- | --- | --- | --- |
| `legacy` | prefix recompute, as before | as before | live |
| `incremental` (default) | per-chunk with carried state | one call, folded weights | folded |

`legacy` is bit-identical to the code before this work, not merely equivalent: running the same
mels through the pre-change tree and through this branch in legacy mode gives byte-for-byte equal
waveforms for all 5 utterances tested, in both the non-streaming call and the old streaming loop
(10 of 10 tensors compare equal with `torch.equal`). Switching a replica back mid-rollout needs no
other cleanup - the vocoder state is per utterance and the flag is read at construction.

The one thing that is *not* bit-identical in `incremental` is the non-streaming call,
because `load()` folds weight norm into the weights for the whole module. Measured on 5
utterances, folded vs live for the same non-streaming call:

| | folded vs live | the model's own noise floor (same call, one mel frame shorter) |
| --- | --- | --- |
| TF32 on (production default) | 37–54 dB | 28–152 dB |
| TF32 off | 52–77 dB | 28–152 dB |

That is, the difference folding makes is the same order as the difference the reference
implementation already has between two input lengths, because cuDNN picks a different algorithm
per shape. If a deployment needs the non-streaming output to stay exactly what it was while
streaming uses the new path, that is what `legacy` is for on the non-streaming replicas; there is
no measured quality reason for it (see the Seed-TTS-Eval table below).

**The CUDA graph path is not in this branch any more.** It worked and it was measured (the numbers
below are kept, and `git show e0fb3fb -- cosyvoice/hifigan/streaming.py` still has the 107-line
`ChunkGraph`), but it bought the last sixth of the win - 101 -> 68 ms on a 20 s utterance, against
273 -> 101 ms for the incremental path itself - in exchange for failure modes that are silent
rather than loud: a graph replays recorded pointers, so anything that invalidates them (autocast's
fp16 weight cache did exactly this) produces NaN or garbage audio instead of an exception. It also
cost ~260 MB allocated / ~908 MB reserved of GPU memory plus a 259 MB pinned host buffer for two
captured lengths, ~0.3 s of capture per new chunk length, and a lock that serialises replays. The
incremental path alone has none of that and keeps five sixths of the speed-up, so the graph work
is parked until there is a latency budget that actually needs it.

### Seed-TTS-Eval test-zh (500 utterances)

`eval_seedtts.py` draws the LLM tokens once per utterance and replays them with the same seed
through each vocoder path, so the three streaming systems differ only in how the vocoder was
driven (`full` is the non-streaming system end to end - its flow is not chunked either, so it is a
reference, not a vocoder-only comparison). Scored with `flow_grpo/evaluate.py`
(CER: Paraformer, SS: ERes2Net cosine against the prompt, MOS: DNSMOS P.835 OVRL):

| system | CER | SS | DNSMOS | paired delta vs prefix-recompute (95% CI, 10k bootstrap) |
| --- | --- | --- | --- | --- |
| prefix-recompute (old streaming) | 0.007888 | 0.873878 | 3.311684 | - |
| incremental eager | 0.007888 | 0.873888 | 3.311685 | dCER 0, dSS +1.0e-05 [-6.1e-06, +2.7e-05], dMOS +1.0e-06 [-3.1e-04, +2.9e-04] |
| incremental + CUDA graph | 0.007888 | 0.873888 | 3.311685 | identical to incremental eager, row by row |
| full (non-stream, reference) | 0.008148 | 0.873999 | 3.312220 | dCER +2.6e-04 [-5.7e-04, +1.2e-03], dSS +1.2e-04, dMOS +5.4e-04 |

Not one of the 500 utterances transcribes differently between the old and the new streaming path,
so the CER difference is exactly zero; SS and DNSMOS differ four to five orders of magnitude below
the metric values themselves, with confidence intervals spanning zero. The graph path reproduces
the eager path's per-utterance rows exactly (its wavs are bit-identical). Listening comparison for
12 of the utterances: `exp/seedtts_zh/listen.html`, built by `make_listening.py`.

Scoring needs `funasr` and a few modelscope dependencies plus `flow_grpo/models/sig_bak_ovr.onnx`
(see flow_grpo/USAGE.zh.md); the wavs and scores live under `hift_streaming/exp/seedtts_zh/`
(gitignored).

### Not run here

`make_mels.py` and `hift_streaming/run_quality.sh` still assume the eval30k toolchain
(`hidden_conditioning/eval30k`, `examples/.../hidden_poc/eval30k`, `venv_qwen3_asr`) sits inside
this repository; it was not ported, so the quality numbers above were produced by calling those
scripts at their source-repository paths. The mel / LLM caches they build are already copied into
`hift_streaming/exp/`, so every other script runs from here unchanged, and
`test_equivalence.py` needs no data at all (it falls back to random mels when `--mel_dir` is
omitted).
