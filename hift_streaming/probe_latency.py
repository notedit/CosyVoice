import os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Matcha-TTS'))
from test_equivalence import load_hift
torch.backends.cudnn.deterministic = os.environ.get('DET', '0') == '1'
torch.backends.cudnn.benchmark = os.environ.get('BENCH', '0') == '1'
dev = torch.device('cuda')
hift = load_hift(sys.argv[1], dev)
if os.environ.get('FOLD','0')=='1':
    hift.fold_weight_norm(); print('weight norm folded')
torch.manual_seed(0)
mel = torch.rand(1, 80, 2000, device=dev)
chunk = int(os.environ.get('CHUNK', '50'))
with torch.inference_mode():
    for rep in range(2):
        st = hift.new_stream_state()
        lat = []
        for i in range(0, 1000, chunk):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            hift.inference_chunk(mel[:, :, i:i + chunk], st)
            torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1e3)
        print(f'stream {rep} chunk={chunk}: ' + ' '.join(f'{x:.1f}' for x in lat))
    # profile steady-state chunk (CPU op breakdown)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        hift.inference_chunk(mel[:, :, 1000:1000 + chunk], st)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=12, max_name_column_width=40))
