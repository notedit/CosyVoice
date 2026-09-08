"""Chunked incremental HiFT vs full-sequence inference: numerical equivalence + timing.

Usage: python hift_streaming/test_equivalence.py --model_dir <CosyVoice3 dir> [--mel_dir <dir of *.pt mels>]
"""
import argparse
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Matcha-TTS'))
from hyperpyyaml import load_hyperpyyaml  # noqa: E402


def load_hift(model_dir, device):
    with open(os.path.join(model_dir, 'cosyvoice3.yaml')) as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'flow': None})
    hift = configs['hift']
    sd = torch.load(os.path.join(model_dir, 'hift.pt'), map_location='cpu', weights_only=True)
    hift.load_state_dict({k.replace('generator.', ''): v for k, v in sd.items()}, strict=True)
    return hift.to(device).eval()


def patch_reference_phase(hift):
    import numpy as np
    import torch.nn.functional as F
    sg = hift.m_source.l_sin_gen
    scale = int(sg.upsample_scale)

    def _f02sine(f0_values):
        rad = (f0_values / sg.sampling_rate) % 1
        rad[:, 0, :] = rad[:, 0, :] + sg.rand_ini.to(rad)
        rad = F.interpolate(rad.transpose(1, 2), scale_factor=1 / scale, mode='linear').transpose(1, 2)
        cum = torch.cumsum(rad.to(torch.float64), dim=1)
        frac = torch.remainder(cum * scale, 1.0).to(rad.dtype) * (2 * np.pi)
        phase = F.interpolate(frac.transpose(1, 2), scale_factor=scale, mode='nearest').transpose(1, 2)
        return torch.sin(phase)
    sg._f02sine = _f02sine


def snr_db(ref, x):
    n = min(ref.shape[-1], x.shape[-1])
    ref, x = ref[..., :n], x[..., :n]
    return (10 * torch.log10(ref.pow(2).sum() / (ref - x).pow(2).sum().clamp_min(1e-20))).item()


@torch.inference_mode()
def run_incremental(hift, mel, chunk_sizes, pad=None):
    state = hift.new_stream_state()
    outs, t = [], 0
    lat = []
    for i, c in enumerate(chunk_sizes):
        final = i == len(chunk_sizes) - 1
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs.append(hift.inference_chunk(mel[:, :, t:t + c], state, finalize=final, finalize_pad_multiple=pad))
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        t += c
    return torch.cat(outs, dim=1), lat


@torch.inference_mode()
def run_recompute(hift, mel, chunk_sizes):
    """The current CosyVoice3Model.token2wav behaviour: full prefix every call."""
    outs, t, off = [], 0, 0
    lat = []
    for i, c in enumerate(chunk_sizes):
        final = i == len(chunk_sizes) - 1
        t += c
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        wav, _ = hift.inference(mel[:, :, :t], finalize=final)
        wav = wav[:, off:]
        off += wav.shape[1]
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        outs.append(wav)
    return torch.cat(outs, dim=1), lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--mel_dir', default=None)
    ap.add_argument('--chunk', type=int, default=50, help='mel frames per chunk (25 tokens * 2)')
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--no_tf32', action='store_true', help='disable TF32 convolutions (bit-exact conv stack)')
    ap.add_argument('--fold', action='store_true', help='fold weight norm + finalize pad 50 (the production config)')
    ap.add_argument('--f64_phase_ref', action='store_true',
                    help='make the full-sequence reference accumulate the NSF phase in float64 (mod 1), '
                         'as the streaming path does, so the residual diff isolates logic errors')
    args = ap.parse_args()
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False
    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    hift = load_hift(args.model_dir, device)
    if args.f64_phase_ref:
        patch_reference_phase(hift)
    if args.fold:
        hift.fold_weight_norm()

    mels = []
    if args.mel_dir:
        for p in sorted(glob.glob(os.path.join(args.mel_dir, '*.pt')))[:args.n]:
            mels.append((os.path.basename(p)[:-3], torch.load(p, map_location=device)))
    else:
        torch.manual_seed(0)
        for T in [37, 120, 300, 731, 1500]:
            mels.append((f'rand{T}', torch.rand(1, 80, T, device=device)))

    # warm-up
    hift.inference(mels[0][1][:, :, :100], finalize=True)
    run_incremental(hift, mels[0][1][:, :, :100], [50, 50])

    print(f'{"utt":>22} {"T":>5} {"len_ok":>6} {"maxabs":>9} {"snr_dB":>7} | {"full_ms":>8} {"recomp_ms":>10} {"incr_ms":>8} {"recomp_last":>11} {"incr_max":>9}')
    for name, mel in mels:
        T = mel.shape[2]
        sizes = [args.chunk] * (T // args.chunk)
        rem = T - sum(sizes)
        if rem or not sizes:
            sizes.append(rem)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ref, _ = hift.inference(mel, finalize=True)
        torch.cuda.synchronize(); t_full = time.perf_counter() - t0
        inc, lat_inc = run_incremental(hift, mel, sizes, 50 if args.fold else None)
        rec, lat_rec = run_recompute(hift, mel, sizes)
        ok = ref.shape[1] == inc.shape[1] == rec.shape[1]
        d_inc = (ref[:, :inc.shape[1]] - inc[:, :ref.shape[1]]).abs().max().item()
        d_rec = (ref[:, :rec.shape[1]] - rec[:, :ref.shape[1]]).abs().max().item()
        print(f'{name:>22} {T:>5} {str(ok):>6} {d_inc:9.2e} {snr_db(ref, inc):7.1f} | {t_full*1e3:8.1f} {sum(lat_rec)*1e3:10.1f} {sum(lat_inc)*1e3:8.1f} {lat_rec[-1]*1e3:11.1f} {max(lat_inc)*1e3:9.1f}'
              f'   (recompute-vs-full maxabs {d_rec:.2e}, snr {snr_db(ref, rec):.1f})')


if __name__ == '__main__':
    main()
