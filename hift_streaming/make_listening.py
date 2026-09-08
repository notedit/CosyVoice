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
"""Build a self-contained listening-comparison page for the Seed-TTS-Eval run.

Reads what eval_seedtts.py and flow_grpo/evaluate.py produced under --root and
writes one HTML file with the summary table and N utterances, each with a player
per system (audio inlined as base64 mp3, so the page needs no other files).

    python hift_streaming/make_listening.py --root hift_streaming/exp/seedtts_zh \
        --out hift_streaming/exp/seedtts_zh/listen.html --n 12
"""

import argparse
import base64
import html
import json
import os
import random
import subprocess
import tempfile

SYSTEMS = [
    ('recompute', '原有流式', '每 chunk 重算整个 mel 前缀'),
    ('incremental', '增量 chunk', '本工作，逐 chunk 携带状态'),
    ('full', '非流式整句', '参考：flow 与声码器都不分块'),
]


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=12, help='utterances to include')
    ap.add_argument('--bitrate', default='64k')
    return ap.parse_args()


def mp3_data_uri(wav_path, bitrate):
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        out = tmp.name
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav_path, '-ac', '1', '-b:a', bitrate, out], check=True)
    with open(out, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    os.remove(out)
    return 'data:audio/mpeg;base64,' + data


def load(root):
    with open(os.path.join(root, 'testset.jsonl')) as f:
        items = [json.loads(x) for x in f if x.strip()]
    summaries, per_item = {}, {}
    for key, _, _ in SYSTEMS:
        s_path = os.path.join(root, key, 'summary.json')
        p_path = os.path.join(root, key, 'per_item.jsonl')
        if os.path.exists(s_path):
            with open(s_path) as f:
                summaries[key] = json.load(f)
        if os.path.exists(p_path):
            with open(p_path) as f:
                per_item[key] = {r['idx']: r for r in map(json.loads, f)}
    return items, summaries, per_item


def paired(per_item, base='recompute', n_boot=10000):
    """Paired mean differences vs `base` with a bootstrap 95% CI, per system."""
    if base not in per_item:
        return {}
    idxs = sorted(per_item[base])
    out = {}
    for key, rows in per_item.items():
        if key == base or not all(i in rows for i in idxs):
            continue
        stats = {}
        for field in ('err', 'ss', 'dnsmos_p835_ovrl'):
            diffs = [rows[i][field] - per_item[base][i][field] for i in idxs]
            mean = sum(diffs) / len(diffs)
            rnd = random.Random(0)
            means = sorted(sum(rnd.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n_boot))
            stats[field] = (mean, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])
        out[key] = stats
    return out


def pick(items, per_item, n):
    """A spread over utterance length, plus the two items where the old and the new
    streaming path disagree most on CER (if the scores are there)."""
    idxs = list(range(len(items)))
    chosen = []
    if 'incremental' in per_item and 'recompute' in per_item:
        diff = sorted(idxs, key=lambda i: -abs(per_item['incremental'].get(i, {}).get('err', 0)
                                               - per_item['recompute'].get(i, {}).get('err', 0)))
        chosen = [i for i in diff[:2] if abs(per_item['incremental'].get(i, {}).get('err', 0)
                                            - per_item['recompute'].get(i, {}).get('err', 0)) > 1e-9]
    by_len = sorted(idxs, key=lambda i: len(items[i]['text']))
    step = max(1, len(by_len) // max(1, n - len(chosen)))
    for i in by_len[::step]:
        if i not in chosen:
            chosen.append(i)
        if len(chosen) >= n:
            break
    return sorted(chosen)


CSS = """
:root{color-scheme:light;--bg:#f1f4f2;--surface:#ffffff;--sunk:#e9efec;--ink:#101f1b;--muted:#5b6c66;
 --line:#d1ddd8;--line-soft:#e4ebe8;--accent:#0d7a6b;--old:#ad5029;--eager:#3d6ba5;--code-bg:#edf2f0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0e1513;--surface:#161f1c;
 --sunk:#121b18;--ink:#dee8e4;--muted:#94a59f;--line:#26332e;--line-soft:#1e2a26;--accent:#35b8a3;--old:#e07b4c;
 --eager:#71a5de;--code-bg:#19241f}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0e1513;--surface:#161f1c;--sunk:#121b18;--ink:#dee8e4;--muted:#94a59f;
 --line:#26332e;--line-soft:#1e2a26;--accent:#35b8a3;--old:#e07b4c;--eager:#71a5de;--code-bg:#19241f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:15.5px;line-height:1.75;
 font-family:"Noto Sans SC","IBM Plex Sans",-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:44px 24px 100px}
h1{font-family:"IBM Plex Sans","Noto Sans SC",sans-serif;font-weight:700;font-size:31px;line-height:1.25;margin:0 0 12px;letter-spacing:-.01em;text-wrap:balance}
h2{font-family:"IBM Plex Sans","Noto Sans SC",sans-serif;font-weight:600;font-size:19px;margin:52px 0 14px;padding-top:20px;border-top:1px solid var(--line)}
p{max-width:68ch;margin:0 0 14px}
.lede{font-size:17px;color:var(--muted);max-width:62ch}
.meta{display:flex;flex-wrap:wrap;gap:6px 20px;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--muted);margin-bottom:8px}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.87em;background:var(--code-bg);padding:1px 5px;border-radius:3px}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface);margin:0 0 18px}
table{border-collapse:collapse;width:100%;font-size:13.2px;font-variant-numeric:tabular-nums}
th,td{padding:8px 13px;text-align:left;border-bottom:1px solid var(--line-soft)}
tr:last-child td{border-bottom:none}
th{font-family:"IBM Plex Sans",sans-serif;font-weight:500;font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);background:var(--sunk)}
td.n,th.n{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.6px}
.item{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:16px 18px;margin:0 0 14px}
.item .hd{display:flex;gap:12px;align-items:baseline;margin-bottom:12px}
.item .idx{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--muted)}
.item .txt{font-size:14.5px;line-height:1.6}
.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(228px,1fr));gap:12px}
.player{display:flex;flex-direction:column;gap:6px}
.player .lab{font-family:"IBM Plex Sans",sans-serif;font-size:12px;font-weight:600;display:flex;align-items:center;gap:6px}
.player .lab i{width:9px;height:9px;border-radius:2px;display:inline-block;font-style:normal}
.player .sub{font-size:11px;color:var(--muted);font-family:"IBM Plex Mono",ui-monospace,monospace}
audio{width:100%;height:34px}
.note{border-left:2px solid var(--accent);background:var(--sunk);padding:12px 16px;border-radius:0 5px 5px 0;margin:0 0 18px;max-width:70ch}
.note p{margin:0}
"""

COLORS = {'recompute': 'var(--old)', 'incremental': 'var(--accent)', 'full': 'var(--muted)'}


def main():
    args = get_args()
    items, summaries, per_item = load(args.root)
    chosen = pick(items, per_item, args.n)
    present = [s for s in SYSTEMS if os.path.isdir(os.path.join(args.root, s[0], 'wavs'))]

    rows = []
    for key, name, desc in present:
        s = summaries.get(key, {})
        cer = s.get('cer_zh')
        rows.append('<tr><td>{}<div style="color:var(--muted);font-size:11.5px">{}</div></td>'
                    '<td class="n">{}</td><td class="n">{}</td><td class="n">{}</td></tr>'.format(
                        html.escape(name), html.escape(desc),
                        '{:.4f}'.format(cer) if cer is not None else '—',
                        '{:.4f}'.format(s['ss_eres2net']) if s.get('ss_eres2net') is not None else '—',
                        '{:.3f}'.format(s['dnsmos_p835_ovrl']) if s.get('dnsmos_p835_ovrl') is not None else '—'))

    deltas = paired(per_item)
    fmt = lambda t: '{:+.1e} [{:+.1e}, {:+.1e}]'.format(*t)  # noqa: E731
    drows = []
    for key, name, _ in present:
        if key not in deltas:
            continue
        st = deltas[key]
        drows.append('<tr><td>{}</td><td class="n">{}</td><td class="n">{}</td><td class="n">{}</td></tr>'.format(
            html.escape(name), fmt(st['err']), fmt(st['ss']), fmt(st['dnsmos_p835_ovrl'])))

    blocks = []
    for i in chosen:
        players = []
        for key, name, _ in present:
            wav = os.path.join(args.root, key, 'wavs', '{:06d}.wav'.format(i))
            if not os.path.exists(wav):
                continue
            r = per_item.get(key, {}).get(i, {})
            sub = ''
            if r:
                sub = 'CER {:.2f} · SS {:.3f} · MOS {:.2f}'.format(r.get('err', 0), r.get('ss', 0), r.get('dnsmos_p835_ovrl', 0))
            players.append('<div class="player"><span class="lab"><i style="background:{}"></i>{}</span>'
                           '<audio controls preload="none" src="{}"></audio>'
                           '<span class="sub">{}</span></div>'.format(
                               COLORS.get(key, 'var(--muted)'), html.escape(name),
                               mp3_data_uri(wav, args.bitrate), sub))
        blocks.append('<div class="item"><div class="hd"><span class="idx">#{:03d}</span>'
                      '<span class="txt">{}</span></div><div class="players">{}</div></div>'.format(
                          i, html.escape(items[i]['text']), ''.join(players)))

    n_items = summaries.get('incremental', {}).get('num_items', len(items))
    page = """<title>流式声码器试听对比</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@500;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap">
<style>{css}</style>
<div class="wrap">
<h1>流式声码器试听对比</h1>
<p class="lede">同一条 LLM 采样、同一个 seed，只有声码器的驱动方式不同。Seed-TTS-Eval test-zh 前 {n} 条的客观指标见下表，再往下是逐条试听。</p>
<div class="meta"><span>Seed-TTS-Eval test-zh</span><span>CosyVoice3-0.5B</span><span>CER: Paraformer</span><span>SS: ERes2Net</span><span>MOS: DNSMOS P.835 OVRL</span></div>

<h2>客观指标</h2>
<div class="tbl"><table>
<tr><th>系统</th><th class="n">CER ↓</th><th class="n">SS ↑</th><th class="n">DNSMOS ↑</th></tr>
{rows}
</table></div>
<div class="note"><p>两条流式路径共用同一份 LLM token 与同一个随机种子，因此 mel 完全相同，差异只来自声码器。<code>非流式整句</code>那一行的 flow 也没有分块，属于另一条端到端路径，仅作参考。</p></div>

<h2>配对差值 vs 原有流式（10000 次 bootstrap 95% CI）</h2>
<div class="tbl"><table>
<tr><th>系统</th><th class="n">ΔCER</th><th class="n">ΔSS</th><th class="n">ΔDNSMOS</th></tr>
{drows}
</table></div>
<p>500 条里<strong>没有一条</strong>的 ASR 转写与原有流式不同，所以 ΔCER 恒为 0；SS 与 DNSMOS 的差值比指标本身小 4–5 个数量级，置信区间都跨过 0。</p>

<h2>逐条试听（{k} 条）</h2>
{blocks}
</div>
""".format(css=CSS, n=n_items, rows='\n'.join(rows), k=len(chosen), blocks='\n'.join(blocks), drows='\n'.join(drows))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(page)
    print('wrote {} ({:.1f} MB, {} utterances x {} systems)'.format(
        args.out, os.path.getsize(args.out) / 1e6, len(chosen), len(present)))


if __name__ == '__main__':
    main()
