#!/usr/bin/env python3
# Copyright 2024 Alibaba Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LM-only sanity check for a (from-scratch) trained CosyVoice3 llm.pt.

Loads ONLY the language model (flow / hift are overridden to None), runs one
autoregressive ``inference(...)`` call and confirms the LM produces speech
tokens that stop at a stop token. No vocoder, no audio -> seconds, not minutes.
Use it on any mid-training checkpoint (epoch_*_step_*.pt) or the averaged
llm.pt to confirm the model is wired up and learning before the full
LM + flow + hift inference in run.sh stage 6 / the README.

This deliberately leaves the ``onnx_path`` env var UNSET so that
``cosyvoice.utils.onnx.online_feature`` stays False and ``CosyVoice3LM.__init__``
skips loading speech_tokenizer_v3.batch.onnx -- no onnx files are needed here.

Example
-------
    python sanity_check_lm.py \\
        --config conf/cosyvoice3.yaml \\
        --qwen_pretrain_path ../../../pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \\
        --llm_pt exp/cosyvoice3_scratch/llm/torch_ddp/llm.pt \\
        --text "今天天气真不错。" \\
        --prompt_text "You are a helpful assistant.<|endofprompt|>"
"""
import os
# MUST run before importing cosyvoice: keeps cosyvoice.utils.onnx.online_feature
# False so building CosyVoice3LM does not try to load the *.onnx extractors.
os.environ.pop('onnx_path', None)

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from hyperpyyaml import load_hyperpyyaml  # noqa: E402

from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer  # noqa: E402

ENDOFPROMPT = '<|endofprompt|>'
# CosyVoice3LM.inference hard-asserts this id appears in text/prompt_text
# (cosyvoice/llm/llm.py). The CV3 tokenizer maps <|endofprompt|> to it.
ENDOFPROMPT_ID = 151646


def load_llm(config, qwen_pretrain_path, llm_pt, device):
    # Build ONLY the llm, mirroring the overrides in cosyvoice/bin/train.py.
    with open(config) as f:
        configs = load_hyperpyyaml(f, overrides={
            'flow': None, 'hift': None, 'hifigan': None,
            'qwen_pretrain_path': qwen_pretrain_path,
        })
    lm = configs['llm']

    state = torch.load(llm_pt, map_location='cpu')
    # training/average checkpoints are bare state_dicts; unwrap if nested.
    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    missing, unexpected = lm.load_state_dict(state, strict=False)
    if missing:
        print('[warn] {} missing key(s) when loading {} (e.g. {})'.format(
            len(missing), llm_pt, list(missing)[:5]))
    if unexpected:
        print('[warn] {} unexpected key(s) in {} (e.g. {})'.format(
            len(unexpected), llm_pt, list(unexpected)[:5]))
    return lm.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=str, default='conf/cosyvoice3.yaml')
    parser.add_argument('--qwen_pretrain_path', type=str, required=True,
                        help='CosyVoice-BlankEN dir (Qwen backbone + CV3 tokenizer)')
    parser.add_argument('--llm_pt', type=str, required=True,
                        help='trained llm checkpoint (epoch_*_step_*.pt or averaged llm.pt)')
    parser.add_argument('--text', type=str, default='今天天气真不错，我们出去走走吧。',
                        help='target text to synthesize tokens for')
    parser.add_argument('--prompt_text', type=str,
                        default='You are a helpful assistant.<|endofprompt|>',
                        help='system/instruct prompt; MUST contain <|endofprompt|>')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sampling', type=int, default=25)
    parser.add_argument('--max_token_text_ratio', type=float, default=20)
    parser.add_argument('--min_token_text_ratio', type=float, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    lm = load_llm(args.config, args.qwen_pretrain_path, args.llm_pt, device)
    speech_token_size = lm.speech_token_size

    # tokenize (CV3 tokenizer; <|endofprompt|> -> ENDOFPROMPT_ID)
    tokenizer = get_qwen_tokenizer(token_path=args.qwen_pretrain_path,
                                   skip_special_tokens=True, version='cosyvoice3')
    text_ids = tokenizer.encode(args.text)
    prompt_ids = tokenizer.encode(args.prompt_text)
    if ENDOFPROMPT_ID not in (prompt_ids + text_ids):
        print("[error] '{}' (id {}) not found after tokenization -- "
              "CosyVoice3LM.inference will assert. Put '{}' in --prompt_text.".format(
                  ENDOFPROMPT, ENDOFPROMPT_ID, ENDOFPROMPT), file=sys.stderr)
        sys.exit(1)

    # mirror the dtypes used by cosyvoice/cli/frontend.py + model.py (int32)
    text = torch.tensor([text_ids], dtype=torch.int32, device=device)
    text_len = torch.tensor([len(text_ids)], dtype=torch.int32, device=device)
    prompt_text = torch.tensor([prompt_ids], dtype=torch.int32, device=device)
    prompt_text_len = torch.tensor([len(prompt_ids)], dtype=torch.int32, device=device)
    # empty prompt speech token -> inference takes the zero-embedding branch
    prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32, device=device)
    prompt_speech_token_len = torch.tensor([0], dtype=torch.int32, device=device)
    # embedding is accepted but unused by Qwen2LM/CosyVoice3LM.inference
    embedding = torch.zeros(1, 192, device=device)

    max_len = int(len(text_ids) * args.max_token_text_ratio)

    t0 = time.time()
    tokens = []
    for top_id in lm.inference(
            text=text, text_len=text_len,
            prompt_text=prompt_text, prompt_text_len=prompt_text_len,
            prompt_speech_token=prompt_speech_token,
            prompt_speech_token_len=prompt_speech_token_len,
            embedding=embedding,
            sampling=args.sampling,
            max_token_text_ratio=args.max_token_text_ratio,
            min_token_text_ratio=args.min_token_text_ratio,
            uuid=''):
        tokens.append(int(top_id))
    dt = time.time() - t0

    # checks
    ok = True
    if len(tokens) == 0:
        print('[FAIL] LM produced 0 tokens')
        ok = False
    out_of_range = [t for t in tokens if t < 0 or t >= speech_token_size]
    if out_of_range:
        print('[FAIL] {} yielded token(s) outside [0,{}) e.g. {}'.format(
            len(out_of_range), speech_token_size, out_of_range[:5]))
        ok = False
    stopped = len(tokens) < max_len
    if not stopped:
        print('[FAIL] hit max_len={} without emitting a stop token '
              '(model likely undertrained or stop handling is wrong)'.format(max_len))
        ok = False

    print('---')
    print('device             : {}'.format(device))
    print('text tokens        : {}'.format(len(text_ids)))
    print('generated tokens   : {}'.format(len(tokens)))
    if tokens:
        print('token id range     : [{}, {}]  (valid speech ids: [0, {}))'.format(
            min(tokens), max(tokens), speech_token_size))
    print('max_len            : {}'.format(max_len))
    print('stopped at stop tok: {}'.format(stopped))
    print('decode time        : {:.2f}s  ({:.1f} tok/s)'.format(
        dt, len(tokens) / dt if dt > 0 else 0.0))
    print('RESULT             : {}'.format('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
