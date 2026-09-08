"""Vocoder-only benchmark with the exact chunk schedule CosyVoice3Model.tts uses in
streaming mode, on the 60 eval30k utterances (mel from the official flow).

Systems written to <out_root>/<system>/wavs:
  full         hift.inference(mel, finalize=True)            (non-streaming reference)
  recompute    current token2wav: hift.inference on the growing mel prefix per chunk
  incremental  hift.inference_chunk with carried state (this work)
Per-utterance timings go to <out_root>/bench.jsonl (cold = first pass, warm = second pass
with cudnn plans for these shapes already cached).
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import torch
import torchaudio

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Matcha-TTS'))
from test_equivalence import load_hift, snr_db  # noqa: E402

TOKEN_HOP, TOKEN_MAX_HOP, SCALE, LOOKAHEAD, RATIO = 25, 100, 2, 3, 2


def schedule(n_tokens, n_prompt):
    """Mel-frame chunk sizes produced by CosyVoice3Model.tts(stream=True)."""
    pad = int(math.ceil(n_prompt / TOKEN_HOP) * TOKEN_HOP - n_prompt)
    hop, off, sizes = TOKEN_HOP, 0, []
    while True:
        this_hop = hop + pad if off == 0 else hop
        if n_tokens - off >= this_hop + LOOKAHEAD:
            sizes.append(this_hop * RATIO)
            off += this_hop
            hop = min(TOKEN_MAX_HOP, hop * SCALE)
        else:
            break
    sizes.append((n_tokens - off) * RATIO)  # finalize
    return sizes


def sync():
    torch.cuda.synchronize()


@torch.inference_mode()
def run_recompute(hift, mel, sizes):
    outs, lat, t, off = [], [], 0, 0
    for i, c in enumerate(sizes):
        t += c
        sync(); t0 = time.perf_counter()
        wav, _ = hift.inference(mel[:, :, :t], finalize=i == len(sizes) - 1)
        wav = wav[:, off:]
        off += wav.shape[1]
        sync(); lat.append(time.perf_counter() - t0)
        outs.append(wav)
    return torch.cat(outs, dim=1), lat


@torch.inference_mode()
def run_full(hift, mel, sizes=None):
    """Non-streaming path (CosyVoice3Model.token2wav with stream=False): one call on the
    whole mel. `sizes` is ignored, it only keeps the signature of the streaming runners."""
    sync(); t0 = time.perf_counter()
    wav, _ = hift.inference(mel, finalize=True)
    sync()
    return wav, [time.perf_counter() - t0]


@torch.inference_mode()
def run_incremental(hift, mel, sizes, pad_multiple=None):
    outs, lat, t = [], [], 0
    st = hift.new_stream_state()
    for i, c in enumerate(sizes):
        sync(); t0 = time.perf_counter()
        outs.append(hift.inference_chunk(mel[:, :, t:t + c], st, finalize=i == len(sizes) - 1,
                                         finalize_pad_multiple=pad_multiple))
        sync(); lat.append(time.perf_counter() - t0)
        t += c
    return torch.cat(outs, dim=1), lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--mel_dir', required=True)
    ap.add_argument('--cache_dir', required=True, help='eval30k llm_cache, for prompt token lengths')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--pad_multiple', type=int, default=None)
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--fold', action='store_true', help='fold weight norm into the weights first')
    args = ap.parse_args()
    device = torch.device('cuda')
    hift = load_hift(args.model_dir, device)
    if args.fold:
        hift.fold_weight_norm()
    for s in ['full', 'recompute', 'incremental']:
        os.makedirs(os.path.join(args.out_root, s, 'wavs'), exist_ok=True)
    rows = []
    paths = sorted(glob.glob(os.path.join(args.mel_dir, '*.pt')))[:args.n]
    for path in paths:
        utt = os.path.basename(path)[:-3]
        mel = torch.load(path, map_location=device)
        c = torch.load(os.path.join(args.cache_dir, utt + '.pt'), map_location='cpu', weights_only=True)
        n_tokens, n_prompt = c['speech_token'].shape[1], c['flow_prompt_speech_token'].shape[1]
        assert mel.shape[2] == n_tokens * RATIO, (mel.shape, n_tokens)
        sizes = schedule(n_tokens, n_prompt)
        row = {'utt': utt, 'frames': mel.shape[2], 'sec': mel.shape[2] * 480 / 24000, 'chunks': sizes}
        wavs = {}
        # streaming passes first: the full-length reference call would otherwise warm the
        # cuDNN plan for the recompute path's final (full-length) shape
        for name, fn in [('recompute', run_recompute), ('incremental', lambda h, m, s: run_incremental(h, m, s, args.pad_multiple))]:
            for pas in ['cold', 'warm']:
                wavs[name], lat = fn(hift, mel, sizes)
                row[f'{name}_{pas}_ms'] = [x * 1e3 for x in lat]
        with torch.inference_mode():
            sync(); t0 = time.perf_counter()
            full, _ = hift.inference(mel, finalize=True)
            sync(); row['full_ms'] = (time.perf_counter() - t0) * 1e3
        for name, wav in wavs.items():
            assert wav.shape[1] == full.shape[1], (name, wav.shape, full.shape)
            row[f'{name}_snr_db'] = snr_db(full, wav)
            row[f'{name}_maxabs'] = (full - wav).abs().max().item()
            torchaudio.save(os.path.join(args.out_root, name, 'wavs', utt + '.wav'), wav.cpu(), 24000)
        torchaudio.save(os.path.join(args.out_root, 'full', 'wavs', utt + '.wav'), full.cpu(), 24000)
        rows.append(row)
        print(f"{utt} {row['sec']:5.1f}s chunks={len(sizes)} full={row['full_ms']:.0f}ms "
              f"recompute cold/warm={sum(row['recompute_cold_ms']):.0f}/{sum(row['recompute_warm_ms']):.0f}ms "
              f"incr cold/warm={sum(row['incremental_cold_ms']):.0f}/{sum(row['incremental_warm_ms']):.0f}ms "
              f"snr rec/incr={row['recompute_snr_db']:.1f}/{row['incremental_snr_db']:.1f}", flush=True)
    with open(os.path.join(args.out_root, 'bench.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
