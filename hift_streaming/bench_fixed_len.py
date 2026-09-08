"""Production-style vocoder benchmark on N-second utterances with varying prompt lengths
(so the old prefix-recompute path sees a new cuDNN shape on every chunk, as in a real
service). Usage: bench_fixed_len.py <model_dir> [seconds ...]  (default 10 20)."""
import glob
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_equivalence import load_hift  # noqa: E402
from bench_schedule import schedule, run_full, run_recompute, run_incremental  # noqa: E402

MODEL_DIR = sys.argv[1]
SECONDS = [int(x) for x in sys.argv[2:]] or [10, 20]
MEL_DIR = os.path.join(os.path.dirname(__file__), 'exp', 'mels')

FOLD = os.environ.get('FOLD', '1') == '1'  # FOLD=0 reproduces the vocoder as it was before this work
hift = load_hift(MODEL_DIR, 'cuda')
if FOLD:
    hift.fold_weight_norm()
big = torch.cat([torch.load(p, map_location='cuda') for p in sorted(glob.glob(os.path.join(MEL_DIR, '*.pt')))], dim=2)
# The non-streaming rows run last so that their full-length inference call does not warm the
# cuDNN plan the prefix-recompute path needs for its final chunk, and each of them gets its own
# mel length (+1 / +3 frames, <= 0.1 % of the audio): every row must build its own plan,
# otherwise a row would report a plan warmed by an earlier row as if it were cold.
#   full (non-stream, 1 call)    what tts(stream=False) does, untouched by this work; run it with
#                                FOLD=0 and FOLD=1 to see what load()'s weight-norm folding buys
#   non-stream via 200-frame     an offline request pushed through the incremental path in
#   chunks                       fixed 200-frame chunks instead of one call
def offline_chunks(T, P):
    return [200] * (T // 200) + ([T % 200] if T % 200 else [])


paths = [('prefix-recompute (old)', run_recompute, 0, schedule),
         ('incremental', lambda h, m, s: run_incremental(h, m, s, 50), 0, schedule),
         ('full (non-stream, 1 call)', run_full, 1, schedule),
         ('non-stream via 200f chunks', lambda h, m, s: run_incremental(h, m, s, 50), 3, offline_chunks)]
with torch.inference_mode():
    for sec in SECONDS:
        T0 = sec * 50
        cases = [(T0 + 2 * i, 5 + 3 * i) for i in range(8)]  # 8 utterances, different prompt lengths
        print(f'=== {sec} s per utterance; weight norm {"folded" if FOLD else "live (pre-work baseline)"}; chunk schedule e.g. {schedule(cases[0][0] // 2, cases[0][1])} mel frames (50 frames = 1 s) ===')
        for name, fn, shift, sizes_fn in paths:
            fn(hift, big[:, :, :T0 - 3], sizes_fn(T0 - 3 if sizes_fn is offline_chunks else (T0 - 3) // 2, 0))  # process warm-up
            totals, firsts, mids, lasts, maxs = [], [], [], [], []
            for i, (T, P) in enumerate(cases):
                sizes = sizes_fn(T + shift, P) if sizes_fn is offline_chunks else sizes_fn(T // 2, P)
                _, lat = fn(hift, big[:, :, i * 1100:i * 1100 + T + shift], sizes)
                lat = [x * 1e3 for x in lat]
                totals.append(sum(lat)); firsts.append(lat[0]); mids += lat[1:-1]; lasts.append(lat[-1]); maxs.append(max(lat))
            tot = statistics.median(totals)
            mid = f'{statistics.median(mids):4.0f}' if mids else '   -'  # the non-stream path has one chunk
            print(f'{name:26s} total {tot:5.0f} ms (min {min(totals):.0f} max {max(totals):.0f}) | first {statistics.median(firsts):4.0f} | '
                  f'middle median {mid} | last {statistics.median(lasts):4.0f} | max chunk {statistics.median(maxs):4.0f} ms | '
                  f'RTF {tot / (sec * 1000):.4f} | per 1 s audio {tot / sec:.1f} ms')
        for name, fn, _, sizes_fn in paths:  # warm reference: same shapes twice, cuDNN plans all cached
            sizes = sizes_fn(T0, 0) if sizes_fn is offline_chunks else sizes_fn(T0 // 2, 0)
            fn(hift, big[:, :, :T0], sizes)
            lat = [x * 1e3 for x in fn(hift, big[:, :, :T0], sizes)[1]]
            print(f'  warm (shapes cached) {name:26s} total {sum(lat):4.0f} ms | chunks: ' + ' '.join(f'{x:.0f}' for x in lat))
