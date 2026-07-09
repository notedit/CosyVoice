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
"""Hard-case text synthesis (paper Sec 3.4): repetition-pattern augmentation.

Three strategies mimic the dominant TTS failure modes on repeated text:
  * LWR - local word repetition: one word repeated 3-5 times in place
  * SMR - sparse multi-word repetition: 2-3 words, each repeated 2-3 times
  * GSR - global sentence repetition: the whole sentence repeated 2-3 times

Chinese text is segmented with jieba; other text splits on whitespace.
"""

import argparse
import json
import random
import re

_ZH_CHARS = re.compile(r'[一-鿿]')


def tokenize(text):
    if _ZH_CHARS.search(text):
        import jieba
        return list(jieba.cut(text)), ''
    return text.split(), ' '


def lwr(text, rng):
    words, sep = tokenize(text)
    if not words:
        return text
    i = rng.randrange(len(words))
    words[i] = sep.join([words[i]] * rng.randint(3, 5)) if sep else words[i] * rng.randint(3, 5)
    return sep.join(words)


def smr(text, rng):
    words, sep = tokenize(text)
    if not words:
        return text
    for i in rng.sample(range(len(words)), min(rng.randint(2, 3), len(words))):
        words[i] = sep.join([words[i]] * rng.randint(2, 3)) if sep else words[i] * rng.randint(2, 3)
    return sep.join(words)


def gsr(text, rng):
    sep = '' if _ZH_CHARS.search(text) else ' '
    return sep.join([text] * rng.randint(2, 3))


STRATEGIES = {'lwr': lwr, 'smr': smr, 'gsr': gsr}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='clean prompt jsonl')
    parser.add_argument('--output', required=True, help='hard-case jsonl')
    parser.add_argument('--num_samples', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=1986)
    return parser.parse_args()


def main():
    args = get_args()
    rng = random.Random(args.seed)
    with open(args.input) as f:
        items = [json.loads(line) for line in f if line.strip()]
    assert items, f'no items in {args.input}'
    with open(args.output, 'w') as f:
        for _ in range(args.num_samples):
            item = dict(rng.choice(items))
            strategy = rng.choice(list(STRATEGIES))
            item['text'] = STRATEGIES[strategy](item['text'], rng)
            item['hard_case'] = strategy
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'wrote {args.num_samples} hard-case prompts to {args.output}')


if __name__ == '__main__':
    main()
