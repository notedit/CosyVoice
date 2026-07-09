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
"""Zero-shot TTS evaluation: CER/WER + speaker similarity (ERes2Net) + DNSMOS P.835.

Test data: jsonl lines {"text", "prompt_text", "prompt_wav"} - the format used
by Seed-TTS-Eval / CV3-Eval style test sets. Pass --flow_ckpt to evaluate a
GRPO-merged flow checkpoint (from export_merged.py) against the same base model.

    python evaluate.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B \
        --test_data data/test_zh.jsonl --output_dir exp/eval_baseline
    python evaluate.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B \
        --flow_ckpt flow_grpo.pt --test_data data/test_zh.jsonl --output_dir exp/eval_grpo

Note: the paper's SS1 uses WavLM embeddings (as in the official seed-tts-eval
kit); here we report ERes2Net cosine (the paper's SS2 and its training reward).
For paper-comparable SS1/UTMOS numbers, run the official seed-tts-eval toolkit
on the wavs saved in --output_dir/wavs.
"""

import argparse
import json
import logging
import os
import sys

import torch
import torchaudio
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'third_party/Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402
from cosyvoice.utils.file_utils import load_wav  # noqa: E402

from rewards.asr import ASRReward, contains_chinese, error_rate  # noqa: E402
from rewards.dnsmos import DNSMOSReward  # noqa: E402
from rewards.speaker_sim import SpeakerSimilarityReward  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--flow_ckpt', default='', help='optional merged flow.pt from export_merged.py')
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', default='conf/grpo.yaml', help='reward model ids are read from here')
    parser.add_argument('--skip_synthesis', action='store_true', help='reuse wavs already in output_dir/wavs')
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()


def synthesize(args, items, wav_dir):
    cosyvoice = CosyVoice3(args.model_dir, fp16=args.fp16)
    if args.flow_ckpt:
        cosyvoice.model.flow.load_state_dict(
            torch.load(args.flow_ckpt, map_location='cpu', weights_only=True), strict=True)
        cosyvoice.model.flow.to(cosyvoice.model.device).eval()
        logging.info(f'loaded flow checkpoint {args.flow_ckpt}')
    for idx, item in enumerate(items):
        out_path = os.path.join(wav_dir, f'{idx:06d}.wav')
        if os.path.exists(out_path):
            continue
        chunks = [out['tts_speech'] for out in cosyvoice.inference_zero_shot(
            item['text'], item['prompt_text'], item['prompt_wav'], stream=False)]
        torchaudio.save(out_path, torch.concat(chunks, dim=1), cosyvoice.sample_rate)
        if (idx + 1) % 50 == 0:
            logging.info(f'synthesized {idx + 1}/{len(items)}')


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    wav_dir = os.path.join(args.output_dir, 'wavs')
    os.makedirs(wav_dir, exist_ok=True)
    with open(args.test_data) as f:
        items = [json.loads(line) for line in f if line.strip()]
    if not args.skip_synthesis:
        synthesize(args, items, wav_dir)

    with open(args.config) as f:
        rcfg = yaml.safe_load(f)['rewards']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    asr = ASRReward(rcfg['zh_asr_model'], rcfg['en_asr_model'], device=device)
    ss = SpeakerSimilarityReward(rcfg['speaker_model'], device=device)
    mos = DNSMOSReward(rcfg['dnsmos_onnx'], device=device)

    per_item, zh_errs, en_errs, ss_scores, mos_scores = [], [], [], [], []
    for idx, item in enumerate(items):
        wav_path = os.path.join(wav_dir, f'{idx:06d}.wav')
        wav, sr = torchaudio.load(wav_path)
        wav_16k = torchaudio.functional.resample(wav, sr, 16000)
        zh = contains_chinese(item['text'])
        hyp = asr.transcribe(wav_16k, zh)
        err = error_rate(item['text'], hyp)
        (zh_errs if zh else en_errs).append(err)
        sim = ss([wav_16k], load_wav(item['prompt_wav'], 16000))[0]
        ss_scores.append(sim)
        quality = mos.score(wav_16k)
        mos_scores.append(quality)
        per_item.append({'idx': idx, 'text': item['text'], 'hyp': hyp,
                         'err': err, 'ss': sim, 'dnsmos_p835_ovrl': quality})

    summary = {
        'num_items': len(items),
        'cer_zh': sum(zh_errs) / len(zh_errs) if zh_errs else None,
        'wer_en': sum(en_errs) / len(en_errs) if en_errs else None,
        'ss_eres2net': sum(ss_scores) / len(ss_scores),
        'dnsmos_p835_ovrl': sum(mos_scores) / len(mos_scores),
        'flow_ckpt': args.flow_ckpt or 'baseline',
    }
    with open(os.path.join(args.output_dir, 'per_item.jsonl'), 'w') as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in per_item)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
