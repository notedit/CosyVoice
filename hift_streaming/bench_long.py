"""Per-chunk vocoder latency on long inputs (real mels concatenated to N seconds),
with the CosyVoice3Model streaming schedule (50/100/200/200... mel frames).
Cold = first pass over that length (production-like: recompute shapes are all new),
warm = second pass."""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Matcha-TTS'))
from test_equivalence import load_hift, snr_db  # noqa: E402
from bench_schedule import schedule, run_recompute, run_incremental  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--mel_dir', required=True)
    ap.add_argument('--seconds', type=int, nargs='+', default=[30, 60, 120])
    ap.add_argument('--out', required=True)
    ap.add_argument('--fold', action='store_true', help='fold weight norm into the weights first')
    args = ap.parse_args()
    device = torch.device('cuda')
    hift = load_hift(args.model_dir, device)
    if args.fold:
        hift.fold_weight_norm()
    mels = [torch.load(p, map_location=device) for p in sorted(glob.glob(os.path.join(args.mel_dir, '*.pt')))]
    big = torch.cat(mels, dim=2)
    rows = []
    for sec in args.seconds:
        T = sec * 50
        mel = big[:, :, :T]
        sizes = schedule(T // 2, 0)
        row = {'sec': sec, 'frames': T, 'chunks': sizes}
        with torch.inference_mode():
            full, _ = hift.inference(mel, finalize=True)
        for name, fn in [('recompute', run_recompute), ('incremental', lambda h, m, s: run_incremental(h, m, s, 50))]:
            for pas in ['cold', 'warm']:
                wav, lat = fn(hift, mel, sizes)
                row[f'{name}_{pas}_ms'] = [x * 1e3 for x in lat]
            row[f'{name}_snr_db'] = snr_db(full, wav)
        rows.append(row)
        for name in ['recompute', 'incremental']:
            for pas in ['cold', 'warm']:
                lat = row[f'{name}_{pas}_ms']
                print(f'{sec:4d}s {name:11s} {pas:4s} total={sum(lat):7.0f}ms  chunk ms: first={lat[0]:.0f} '
                      f'median={sorted(lat)[len(lat) // 2]:.0f} max={max(lat):.0f} last={lat[-1]:.0f}  n={len(lat)}', flush=True)
    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
