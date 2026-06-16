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
"""Pre-flight validation for CosyVoice3 LM training parquet shards.

Run this BEFORE launching a multi-GPU from-scratch job to catch the crashes
that otherwise only surface minutes into training:

  * missing / empty ``instruct`` column -> ``CosyVoice3LM.forward`` raises
    ``KeyError`` on ``batch['instruct_token']``. ``padding`` only emits that
    key when EVERY sample in a batch carries an instruct token
    (cosyvoice/dataset/processor.py), so one missing row poisons whole batches.
  * ``speech_token`` ids outside ``[0, speech_token_size)`` -> embedding index
    error or silently-wrong training (usually means the tokens were extracted
    with the wrong speech tokenizer, not speech_tokenizer_v3).
  * speaker-embedding dimension != expected (192 for CAM++).

Exits non-zero if any shard fails, so it can gate run.sh.

Examples
--------
    # validate the first 20 shards referenced by a data.list (fast gate)
    python validate_parquet.py --data_list data/train.data.list --num_parquet 20
    # validate every shard under a directory
    python validate_parquet.py --dir data/train/parquet --num_parquet 0
    # validate a single shard
    python validate_parquet.py --parquet data/train/parquet/parquet_000000000.tar
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ENDOFPROMPT = '<|endofprompt|>'


def resolve_parquets(args):
    """Collect parquet paths from --parquet / --dir / --data_list (deduped)."""
    files = []
    if args.parquet:
        files.append(args.parquet)
    if args.dir:
        # make_parquet_list.py writes parquet content into '*.tar' filenames.
        for pat in ('parquet_*.tar', '*.parquet'):
            files.extend(sorted(glob.glob(os.path.join(args.dir, pat))))
    if args.data_list:
        with open(args.data_list) as f:
            files.extend([l.strip() for l in f if l.strip()])
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    if args.num_parquet and args.num_parquet > 0:
        out = out[:args.num_parquet]
    return out


def _is_empty(x):
    if x is None:
        return True
    try:
        return len(x) == 0
    except TypeError:
        return True


def validate_shard(path, args):
    """Return (n_rows, [problem strings]) for a single parquet shard."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001 - surface any read error as a failure
        return 0, ['cannot read parquet: {}'.format(e)]

    n = len(df)
    cols = set(df.columns)
    problems = []

    # 1. required columns
    for c in ('utt', 'text', 'speech_token'):
        if c not in cols:
            problems.append("missing required column '{}'".format(c))
    if 'utt_embedding' not in cols and 'spk_embedding' not in cols:
        problems.append("missing speaker embedding column "
                        "(need 'utt_embedding' or 'spk_embedding')")
    if args.require_instruct and 'instruct' not in cols:
        problems.append("missing 'instruct' column -- CV3 requires a per-utt "
                        "system prompt; (re)generate the instruct file and "
                        "re-pack parquet (tools/make_parquet_list.py)")

    def utt_at(i):
        return str(df['utt'].iloc[i]) if 'utt' in cols else '#{}'.format(i)

    # 2. instruct content: non-empty and carries <|endofprompt|>
    if args.require_instruct and 'instruct' in cols:
        empty, no_eop = [], []
        for i, v in enumerate(df['instruct'].tolist()):
            s = '' if v is None else str(v)
            if s.strip() == '':
                empty.append(utt_at(i))
            elif ENDOFPROMPT not in s:
                no_eop.append(utt_at(i))
        if empty:
            problems.append('{}/{} rows have EMPTY instruct (e.g. {})'.format(
                len(empty), n, empty[:3]))
        if no_eop:
            problems.append("{}/{} rows have instruct without '{}' (e.g. {})".format(
                len(no_eop), n, ENDOFPROMPT, no_eop[:3]))

    # 3. speech_token: non-empty and within [0, speech_token_size)
    if 'speech_token' in cols:
        empty, out_of_range = [], []
        for i, v in enumerate(df['speech_token'].tolist()):
            if _is_empty(v):
                empty.append(utt_at(i))
                continue
            arr = np.asarray(v)
            lo, hi = int(arr.min()), int(arr.max())
            if lo < 0 or hi >= args.speech_token_size:
                out_of_range.append('{}(min={},max={})'.format(utt_at(i), lo, hi))
        if empty:
            problems.append('{}/{} rows have EMPTY speech_token (e.g. {})'.format(
                len(empty), n, empty[:3]))
        if out_of_range:
            problems.append('{}/{} rows have speech_token outside [0,{}) (e.g. {})'.format(
                len(out_of_range), n, args.speech_token_size, out_of_range[:3]))

    # 4. embedding dimension
    for col in ('utt_embedding', 'spk_embedding'):
        if col in cols:
            bad = []
            for i, v in enumerate(df[col].tolist()):
                try:
                    d = int(np.asarray(v).reshape(-1).shape[0])
                except Exception:  # noqa: BLE001
                    d = -1
                if d != args.embed_dim:
                    bad.append('{}(dim={})'.format(utt_at(i), d))
            if bad:
                problems.append('{}/{} rows have {} dim != {} (e.g. {})'.format(
                    len(bad), n, col, args.embed_dim, bad[:3]))

    return n, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_argument_group('input (give at least one)')
    src.add_argument('--data_list', type=str, default=None,
                     help='a data.list file listing parquet shard paths')
    src.add_argument('--dir', type=str, default=None,
                     help='a directory of parquet shards (parquet_*.tar / *.parquet)')
    src.add_argument('--parquet', type=str, default=None,
                     help='a single parquet shard')
    parser.add_argument('--num_parquet', type=int, default=10,
                        help='validate only the first N shards (0 = all). '
                             'Sampling is enough to catch systemic issues fast.')
    parser.add_argument('--speech_token_size', type=int, default=6561,
                        help='valid speech-token ids are [0, speech_token_size)')
    parser.add_argument('--embed_dim', type=int, default=192,
                        help='expected speaker-embedding dimension (CAM++ = 192)')
    parser.add_argument('--require_instruct', dest='require_instruct',
                        action='store_true', default=True,
                        help='require a non-empty instruct column (CV3 default)')
    parser.add_argument('--no_require_instruct', dest='require_instruct',
                        action='store_false',
                        help='skip the instruct checks (e.g. for CV2 data)')
    args = parser.parse_args()

    files = resolve_parquets(args)
    if not files:
        print('[error] no parquet files given; use --data_list / --dir / --parquet',
              file=sys.stderr)
        sys.exit(2)

    print('validating {} parquet shard(s) '
          '(speech_token_size={}, embed_dim={}, require_instruct={})'.format(
              len(files), args.speech_token_size, args.embed_dim, args.require_instruct))

    total_rows, bad_shards = 0, 0
    for path in files:
        n, problems = validate_shard(path, args)
        total_rows += n
        if problems:
            bad_shards += 1
            print('  [FAIL] {} ({} rows)'.format(path, n))
            for p in problems:
                print('         - {}'.format(p))
        else:
            print('  [ok]   {} ({} rows)'.format(path, n))

    print('---')
    print('shards: {} ok / {} failed   rows checked: {}'.format(
        len(files) - bad_shards, bad_shards, total_rows))
    if bad_shards:
        print('RESULT: FAIL -- fix the above before training '
              '(do NOT launch the multi-GPU job).')
        sys.exit(1)
    print('RESULT: PASS')


if __name__ == '__main__':
    main()
