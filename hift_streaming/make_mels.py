"""Decode the eval30k LLM token cache with the official flow (non-streaming) and dump
one mel .pt per utterance, so vocoder variants can be compared on identical input."""
import argparse
import glob
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'third_party', 'Matcha-TTS'))
sys.path.insert(0, os.path.join(ROOT, 'hidden_conditioning', 'eval30k'))
from common import load_flow  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    device = torch.device('cuda')
    os.makedirs(args.out_dir, exist_ok=True)
    flow = load_flow(args.model_dir, os.path.join(args.model_dir, 'flow.pt'), device)
    for path in sorted(glob.glob(os.path.join(args.cache_dir, '*.pt'))):
        utt = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out_dir, utt + '.pt')
        if os.path.exists(out):
            continue
        c = torch.load(path, map_location='cpu', weights_only=True)
        token = c['speech_token'].to(device)
        pt = c['flow_prompt_speech_token'].to(device)
        pf = c['prompt_speech_feat'].to(device)
        mel, _ = flow.inference(token=token, token_len=torch.tensor([token.shape[1]], dtype=torch.int32, device=device),
                                prompt_token=pt, prompt_token_len=torch.tensor([pt.shape[1]], dtype=torch.int32, device=device),
                                prompt_feat=pf, prompt_feat_len=torch.tensor([pf.shape[1]], dtype=torch.int32, device=device),
                                embedding=c['flow_embedding'].to(device), streaming=False, finalize=True)
        torch.save(mel.cpu(), out)
        print(utt, mel.shape[2], flush=True)


if __name__ == '__main__':
    main()
