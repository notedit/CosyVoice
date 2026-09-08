# Copyright (c) 2026 CosyVoice
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
"""Seed-TTS-Eval synthesis for the vocoder paths that this work compares.

Every system below shares one LLM sampling per utterance (the tokens are drawn
once and replayed), and the flow runs with the same seed, so the wavs differ
only in how the vocoder was driven:

  full         tts(stream=False): one hift.inference(mel, finalize=True) call
  recompute    tts(stream=True) with hift_mode=legacy - the old streaming path,
               re-running the vocoder on the whole mel prefix per chunk
  incremental  tts(stream=True) with the incremental vocoder

Writes <out_root>/<system>/wavs/{idx:06d}.wav plus <out_root>/testset.jsonl in
the {"text", "prompt_text", "prompt_wav"} format flow_grpo/evaluate.py reads, so
scoring is:

    python flow_grpo/evaluate.py --model_dir $M --test_data <out_root>/testset.jsonl \
        --output_dir <out_root>/<system> --skip_synthesis
"""

import argparse
import json
import os
import sys
import time

import torch
import torchaudio

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'third_party', 'Matcha-TTS'))
from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402

CV3_INSTRUCT_PREFIX = 'You are a helpful assistant.<|endofprompt|>'
SYSTEMS = {  # name -> (stream, use_hift_cache)
    'full': (False, True),
    'recompute': (True, False),
    'incremental': (True, True),
}


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--testset_root', required=True, help='seedtts_testset/zh (holds meta.lst and prompt-wavs/)')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--n', type=int, default=500, help='number of utterances (0 = all)')
    ap.add_argument('--start', type=int, default=0, help='skip the first N utterances (to split a run over GPUs)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--systems', nargs='+', default=list(SYSTEMS))
    ap.add_argument('--legacy', action='store_true',
                    help='build the model with hift_mode=legacy: the vocoder exactly as it was before this work')
    return ap.parse_args()


def read_meta(root, n):
    """seedtts meta.lst: utt|prompt_text|prompt_wav|text"""
    items = []
    with open(os.path.join(root, 'meta.lst')) as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) != 4:
                continue
            utt, prompt_text, prompt_wav, text = parts
            items.append({'utt': utt, 'text': text, 'prompt_text': prompt_text,
                          'prompt_wav': os.path.join(root, prompt_wav)})
            if n and len(items) >= n:
                break
    return items


def main():
    args = get_args()
    items = read_meta(args.testset_root, args.n)
    os.makedirs(args.out_root, exist_ok=True)
    if args.start == 0:  # a split run must not truncate the shared index
        with open(os.path.join(args.out_root, 'testset.jsonl'), 'w') as f:
            for it in items:
                f.write(json.dumps({k: it[k] for k in ('text', 'prompt_text', 'prompt_wav')}, ensure_ascii=False) + '\n')
    for name in args.systems:
        os.makedirs(os.path.join(args.out_root, name, 'wavs'), exist_ok=True)

    cosyvoice = CosyVoice3(args.model_dir, **({'hift_mode': 'legacy'} if args.legacy else {}))
    model, frontend, sr = cosyvoice.model, cosyvoice.frontend, cosyvoice.sample_rate
    real_llm_job = model.llm_job
    t0 = time.time()
    for idx, item in enumerate(items):
        if idx < args.start:
            continue
        paths = {s: os.path.join(args.out_root, s, 'wavs', '{:06d}.wav'.format(idx)) for s in args.systems}
        if all(os.path.exists(p) for p in paths.values()):
            continue
        text = frontend.text_normalize(item['text'], split=False)
        prompt_text = CV3_INSTRUCT_PREFIX + frontend.text_normalize(item['prompt_text'], split=False)
        model_input = frontend.frontend_zero_shot(text, prompt_text, item['prompt_wav'], sr, '')

        # one LLM sampling per utterance, shared by every system
        model.llm_job = real_llm_job
        uid = 'tokens-{}'.format(idx)
        model.tts_speech_token_dict[uid], model.llm_end_dict[uid] = [], False
        torch.manual_seed(args.seed)
        model.llm_job(model_input['text'], model_input['prompt_text'],
                      model_input['llm_prompt_speech_token'], model_input['llm_embedding'], uid)
        tokens = list(model.tts_speech_token_dict.pop(uid))
        model.llm_end_dict.pop(uid)

        def replay_llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid, _t=tokens):
            model.tts_speech_token_dict[uuid].extend(_t)
            model.llm_end_dict[uuid] = True
        model.llm_job = replay_llm_job

        for name in args.systems:
            stream, use_cache = SYSTEMS[name]
            model.use_hift_cache = use_cache and hasattr(model.hift, 'inference_chunk')
            model.token_hop_len = 25  # tts() grows it; reset per utterance
            torch.manual_seed(args.seed)
            chunks = [o['tts_speech'] for o in model.tts(**model_input, stream=stream)]
            torchaudio.save(paths[name], torch.concat(chunks, dim=1), sr)
        if (idx + 1) % 25 == 0:
            print('{}/{} utterances, {:.0f}s elapsed'.format(idx + 1, len(items), time.time() - t0), flush=True)
    print('done: {} utterances x {} systems in {:.0f}s'.format(len(items), len(args.systems), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
