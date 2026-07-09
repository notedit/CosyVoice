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
"""Builds data/raw.jsonl for the FlowTTS-GRPO recipe from yuekai/CV3-Eval.

CV3-Eval (the CosyVoice3 evaluation set, HF: yuekai/CV3-Eval) ships
{prompt_text, target_text, prompt_audio} triples - exactly the
{text, prompt_text, prompt_wav} format this recipe trains on. This script
saves the embedded prompt audio to wav files and writes the jsonl. Intended
for pipeline verification; for real training use a large corpus such as
WenetSpeech4TTS Premium (zh) / LibriTTS-960 (en) in the same jsonl format.
"""

import argparse
import json
import os

import io

import soundfile as sf
from datasets import Audio, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='zero_shot_zh')
    parser.add_argument('--num_samples', type=int, default=32)
    parser.add_argument('--output_dir', default='data')
    args = parser.parse_args()

    wav_dir = os.path.join(args.output_dir, 'prompt_wavs')
    os.makedirs(wav_dir, exist_ok=True)
    ds = load_dataset('yuekai/CV3-Eval', split=args.split, streaming=True)
    # decode=False: raw bytes + soundfile, avoiding the torchcodec dependency
    # that datasets>=3 audio decoding requires
    ds = ds.cast_column('prompt_audio', Audio(decode=False))

    out_path = os.path.join(args.output_dir, 'raw.jsonl')
    kept = 0
    with open(out_path, 'w') as f:
        for item in ds:
            if kept >= args.num_samples:
                break
            array, sampling_rate = sf.read(io.BytesIO(item['prompt_audio']['bytes']))
            duration = len(array) / sampling_rate
            if not (3.0 <= duration <= 30.0):  # prepare_data.py filter, skip early
                continue
            wav_path = os.path.abspath(os.path.join(wav_dir, f'{item["id"]}.wav'))
            sf.write(wav_path, array, sampling_rate)
            f.write(json.dumps({'text': item['target_text'],
                                'prompt_text': item['prompt_text'],
                                'prompt_wav': wav_path}, ensure_ascii=False) + '\n')
            kept += 1
    print(f'wrote {kept} prompts to {out_path}, wavs in {wav_dir}')


if __name__ == '__main__':
    main()
