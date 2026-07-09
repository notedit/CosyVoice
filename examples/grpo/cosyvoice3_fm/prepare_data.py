# Copyright (c) 2026 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Builds the GRPO prompt list from generic jsonl metadata.

Input: one or more jsonl files whose lines contain at least
    {"text": ..., "prompt_text": ..., "prompt_wav": ...}
(`prompt_wav` is the reference audio to clone; `text` is what to synthesize).
The paper draws ~40k such prompts from WenetSpeech4TTS Premium (zh) and
LibriTTS-960 (en). Any dataset in this format works; a few hundred prompts is
enough to verify the pipeline end to end.

Filters: prompt wav must exist, be mono-loadable and 3-30 s long (the speech
tokenizer rejects prompts > 30 s); text must be non-empty.
"""

import argparse
import json
import logging
import os
import random

import torchaudio


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True, help='input jsonl file(s)')
    parser.add_argument('--output', required=True, help='output jsonl')
    parser.add_argument('--val_output', default='', help='optional validation split jsonl')
    parser.add_argument('--val_size', type=int, default=200)
    parser.add_argument('--max_samples', type=int, default=0, help='cap total samples, 0 = no cap')
    parser.add_argument('--min_prompt_sec', type=float, default=3.0)
    parser.add_argument('--max_prompt_sec', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=1986)
    return parser.parse_args()


def valid_item(item, min_sec, max_sec):
    for key in ('text', 'prompt_text', 'prompt_wav'):
        if not item.get(key):
            return False
    if not os.path.exists(item['prompt_wav']):
        return False
    try:
        info = torchaudio.info(item['prompt_wav'])
        duration = info.num_frames / info.sample_rate
    except Exception:
        return False
    return min_sec <= duration <= max_sec


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    items, skipped = [], 0
    for path in args.input:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if valid_item(item, args.min_prompt_sec, args.max_prompt_sec):
                    items.append({'text': item['text'], 'prompt_text': item['prompt_text'],
                                  'prompt_wav': item['prompt_wav']})
                else:
                    skipped += 1
    logging.info(f'kept {len(items)} prompts, skipped {skipped}')
    random.seed(args.seed)
    random.shuffle(items)
    if args.max_samples > 0:
        items = items[:args.max_samples]
    val = []
    if args.val_output:
        val, items = items[:args.val_size], items[args.val_size:]
        with open(args.val_output, 'w') as f:
            f.writelines(json.dumps(i, ensure_ascii=False) + '\n' for i in val)
        logging.info(f'wrote {len(val)} validation prompts to {args.val_output}')
    with open(args.output, 'w') as f:
        f.writelines(json.dumps(i, ensure_ascii=False) + '\n' for i in items)
    logging.info(f'wrote {len(items)} training prompts to {args.output}')


if __name__ == '__main__':
    main()
