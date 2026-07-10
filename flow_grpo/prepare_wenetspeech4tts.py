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
"""Converts extracted WenetSpeech4TTS (Premium) into the recipe's prompt jsonl.

Layout after extracting the HF tars (Wenetspeech4TTS/WenetSpeech4TTS):

    extracted/WenetSpeech4TTS_Premium_N/wavs/<id>.wav
    extracted/WenetSpeech4TTS_Premium_N/txts/<id>.txt   # line 1: "<id>\t<transcript>"
                                                        # line 2: char timestamps (ms)

The paper draws ~40k zh prompts from Premium. Each kept utterance clones its
own audio (prompt_wav/prompt_text) and synthesizes a different utterance's
transcript. Duration is pre-filtered from the timestamp line (no audio reads);
prepare_data.py re-validates via torchaudio afterwards:

    python prepare_wenetspeech4tts.py --extracted_dir data/wenetspeech4tts/extracted \
        --output data/ws4tts_raw.jsonl --num_samples 40000
    python prepare_data.py --input data/ws4tts_raw.jsonl \
        --output data/ws4tts_train.jsonl --val_output data/ws4tts_val.jsonl --val_size 200
"""

import argparse
import ast
import glob
import json
import os
import random


def read_txt(path):
    """Returns (transcript, approx_duration_sec) or None if unparseable."""
    try:
        with open(path) as f:
            first = f.readline().rstrip('\n')
            second = f.readline().strip()
        transcript = first.split('\t', 1)[1].strip()
        spans = ast.literal_eval(second)
        return transcript, spans[-1][1] / 1000.0
    except (IndexError, ValueError, SyntaxError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_dir', required=True,
                        help='directory containing WenetSpeech4TTS_Premium_* subdirs')
    parser.add_argument('--output', required=True)
    parser.add_argument('--num_samples', type=int, default=40000, help='0 = keep all')
    parser.add_argument('--min_sec', type=float, default=3.0)
    parser.add_argument('--max_sec', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=1986)
    args = parser.parse_args()

    txts = glob.glob(os.path.join(args.extracted_dir, 'WenetSpeech4TTS_*', 'txts', '*.txt'))
    print(f'found {len(txts)} transcripts')
    kept, skipped = [], 0
    for txt in txts:
        parsed = read_txt(txt)
        if parsed is None:
            skipped += 1
            continue
        transcript, duration = parsed
        wav = os.path.join(os.path.dirname(os.path.dirname(txt)), 'wavs',
                           os.path.basename(txt)[:-4] + '.wav')
        if not transcript or not (args.min_sec <= duration <= args.max_sec) or not os.path.exists(wav):
            skipped += 1
            continue
        kept.append((transcript, os.path.abspath(wav)))
    print(f'kept {len(kept)} utterances ({args.min_sec}-{args.max_sec}s), skipped {skipped}')

    random.seed(args.seed)
    random.shuffle(kept)
    if args.num_samples > 0:
        kept = kept[:args.num_samples]
    with open(args.output, 'w') as f:
        for i, (transcript, wav) in enumerate(kept):
            f.write(json.dumps({
                'text': kept[(i + 1) % len(kept)][0],
                'prompt_text': transcript,
                'prompt_wav': wav,
            }, ensure_ascii=False) + '\n')
    print(f'wrote {len(kept)} prompts to {args.output}')


if __name__ == '__main__':
    main()
