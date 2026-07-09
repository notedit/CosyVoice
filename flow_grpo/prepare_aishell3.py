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
"""Converts voxbox AISHELL-3 metadata into the recipe's prompt-list jsonl.

Input: the SparkAudio/voxbox metadata jsonl (fields: index, split, text,
duration, wav_path, ...) plus the extracted audio directory. Each kept
utterance becomes one prompt: its own audio + transcript are the cloning
reference (prompt_wav / prompt_text) and the transcript of a *different*
random utterance is the text to synthesize - GRPO is online RL, no target
audio is needed. Output feeds prepare_data.py:

    wget https://huggingface.co/datasets/SparkAudio/voxbox/resolve/main/metadata/aishell-3.jsonl
    tar -xzf aishell-3_0000.tar.gz -C data/aishell3_wavs
    python prepare_aishell3.py --metadata data/aishell-3.jsonl \
        --wav_dir data/aishell3_wavs --output data/aishell3_raw.jsonl --num_samples 4000
"""

import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True, help='voxbox aishell-3.jsonl')
    parser.add_argument('--wav_dir', required=True, help='directory the audio tar was extracted into')
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='train', help='voxbox split to use (train/test)')
    parser.add_argument('--num_samples', type=int, default=4000, help='0 = keep all')
    parser.add_argument('--min_sec', type=float, default=3.0)
    parser.add_argument('--max_sec', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=1986)
    args = parser.parse_args()

    with open(args.metadata) as f:
        meta = [json.loads(line) for line in f if line.strip()]
    kept = [m for m in meta
            if m.get('split') == args.split and m.get('text')
            and args.min_sec <= m.get('duration', 0) <= args.max_sec]
    print(f'{len(meta)} metadata lines -> {len(kept)} usable {args.split} utterances '
          f'({args.min_sec}-{args.max_sec}s)')

    random.seed(args.seed)
    random.shuffle(kept)
    if args.num_samples > 0:
        kept = kept[:args.num_samples]
    # texts to synthesize are drawn from the whole pool, shifted by one so an
    # utterance never gets its own transcript (avoids degenerate prompt==text)
    texts = [m['text'] for m in kept]
    with open(args.output, 'w') as f:
        for i, m in enumerate(kept):
            wav = os.path.abspath(os.path.join(args.wav_dir, m['wav_path']))
            if not os.path.exists(wav):  # the audio tar extracts flat, without the metadata's dir prefix
                wav = os.path.abspath(os.path.join(args.wav_dir, os.path.basename(m['wav_path'])))
            f.write(json.dumps({
                'text': texts[(i + 1) % len(texts)],
                'prompt_text': m['text'],
                'prompt_wav': wav,
            }, ensure_ascii=False) + '\n')
    print(f'wrote {len(kept)} prompts to {args.output}')


if __name__ == '__main__':
    main()
