"""Where does the ~46 dB full-vs-prefix discrepancy come from? + profile."""
import os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Matcha-TTS'))
from test_equivalence import load_hift, snr_db, patch_reference_phase

torch.backends.cudnn.deterministic = os.environ.get('DET', '1') == '1'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = os.environ.get('TF32', '1') == '1'
torch.backends.cuda.matmul.allow_tf32 = os.environ.get('TF32', '1') == '1'
print('deterministic', torch.backends.cudnn.deterministic, 'tf32', torch.backends.cudnn.allow_tf32)
dev = torch.device('cuda')
hift = load_hift(sys.argv[1], dev)
patch_reference_phase(hift)
torch.manual_seed(0)
mel = torch.rand(1, 80, 300, device=dev)
with torch.inference_mode():
    a, _ = hift.inference(mel, finalize=True)
    b, _ = hift.inference(mel, finalize=True)
    print('full twice          maxabs %.2e' % (a - b).abs().max())
    c, _ = hift.inference(mel[:, :, :250], finalize=True)
    n = 240 * 480
    print('full vs full[:250]  maxabs %.2e snr %.1f (first 240 frames)' % ((a[:, :n] - c[:, :n]).abs().max(), snr_db(a[:, :n], c[:, :n])))
    d, _ = hift.inference(mel[:, :, :250], finalize=False)
    n = d.shape[1]
    print('full vs prefix(finalize=False) maxabs %.2e snr %.1f' % ((a[:, :n] - d).abs().max(), snr_db(a[:, :n], d)))
    # f0 only
    hift.f0_predictor.to(torch.float64)
    f_full = hift.f0_predictor(mel.double(), finalize=True)
    f_pre = hift.f0_predictor(mel[:, :, :250].double(), finalize=False)
    print('f0 full vs prefix   maxabs %.2e' % (f_full[:, :f_pre.shape[1]] - f_pre).abs().max())
    # decode only with identical source: is the conv stack length-dependent?
    f0 = f_full.float()
    s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
    s, _, _ = hift.m_source(s)
    s = s.transpose(1, 2)
    y1 = hift.decode(x=mel, s=s, finalize=True)
    y2 = hift.decode(x=mel[:, :, :250], s=s[:, :, :250 * 480], finalize=True)
    n = 240 * 480
    print('decode full vs decode[:250], same source: maxabs %.2e snr %.1f' % ((y1[:, :n] - y2[:, :n]).abs().max(), snr_db(y1[:, :n], y2[:, :n])))
    # source only
    s2 = hift.f0_upsamp(f0[:, None, :250]).transpose(1, 2)
    s2, _, _ = hift.m_source(s2)
    s2 = s2.transpose(1, 2)
    print('source full vs [:250] maxabs %.2e' % (s[:, :, :250 * 480] - s2).abs().max())

    # ---- profile
    from torch.profiler import profile, ProfilerActivity
    hift.inference(mel, finalize=True)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        hift.inference(mel, finalize=True)
        torch.cuda.synchronize()
    print('\n==== full inference T=300 ====')
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=8, max_name_column_width=45))
    st = hift.new_stream_state()
    hift.inference_chunk(mel[:, :, :50], st)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        hift.inference_chunk(mel[:, :, 50:100], st)
        torch.cuda.synchronize()
    print('\n==== incremental chunk 50 frames ====')
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=8, max_name_column_width=45))
    ev = [e for e in prof.events()]
    print('num kernel launches (cuda events):', sum(1 for e in prof.events() if e.device_type.name == 'CUDA'))
