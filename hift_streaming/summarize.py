"""Aggregate bench.jsonl (bench_schedule.py) into a markdown table."""
import json
import sys


def med(x):
    x = sorted(x)
    return x[len(x) // 2]


rows = [json.loads(l) for l in open(sys.argv[1])]
n = len(rows)
print(f'utterances: {n}, duration median {med([r["sec"] for r in rows]):.1f}s, chunks median {med([len(r["chunks"]) for r in rows])}')
print()
print('| system | pass | total vocoder ms (median) | first chunk ms | later chunks ms (median) | last chunk ms (median) | max chunk ms (median over utts) |')
print('|---|---|---|---|---|---|---|')
for sysn in ['recompute', 'incremental']:
    for pas in ['cold', 'warm']:
        k = f'{sysn}_{pas}_ms'
        tot = med([sum(r[k]) for r in rows])
        first = med([r[k][0] for r in rows])
        mid = med([x for r in rows for x in r[k][1:-1]]) if any(len(r[k]) > 2 for r in rows) else float('nan')
        last = med([r[k][-1] for r in rows])
        mx = med([max(r[k]) for r in rows])
        print(f'| {sysn} | {pas} | {tot:.0f} | {first:.0f} | {mid:.0f} | {last:.0f} | {mx:.0f} |')
print(f'| full (non-stream, 1 call) | - | {med([r["full_ms"] for r in rows]):.0f} | | | | |')
print()
for sysn in ['recompute', 'incremental']:
    s = [r[f'{sysn}_snr_db'] for r in rows]
    m = [r[f'{sysn}_maxabs'] for r in rows]
    print(f'{sysn} vs full: SNR median {med(s):.1f} dB (min {min(s):.1f}), max|diff| median {med(m):.2e} (max {max(m):.2e})')
