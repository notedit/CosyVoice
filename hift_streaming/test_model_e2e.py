"""Drive CosyVoice3Model.tts(stream=True) end-to-end (flow + hift, cached LLM tokens)
with hift_mode incremental/legacy and compare the streamed audio and vocoder timing."""
import os
import sys
import time
import glob

import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'third_party', 'Matcha-TTS'))
sys.path.insert(0, os.path.dirname(__file__))
from hyperpyyaml import load_hyperpyyaml  # noqa: E402
from cosyvoice.cli.model import CosyVoice3Model  # noqa: E402
from test_equivalence import snr_db  # noqa: E402

MODEL_DIR = sys.argv[1]
CACHE_DIR = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 5


def build(hift_mode):
    with open(os.path.join(MODEL_DIR, 'cosyvoice3.yaml')) as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None})
    model = CosyVoice3Model(None, configs['flow'], configs['hift'], fp16=False, hift_mode=hift_mode)
    model.flow.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'flow.pt'), map_location='cpu', weights_only=True), strict=True)
    model.flow.to(model.device).eval()
    sd = torch.load(os.path.join(MODEL_DIR, 'hift.pt'), map_location='cpu', weights_only=True)
    model.hift.load_state_dict({k.replace('generator.', ''): v for k, v in sd.items()}, strict=True)
    model.hift.to(model.device).eval()
    model.hift.fold_weight_norm()
    return model


def stream(model, cache):
    tokens = cache['speech_token'][0].tolist()

    def fake_llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        for t in tokens:
            model.tts_speech_token_dict[uuid].append(t)
        model.llm_end_dict[uuid] = True
    model.llm_job = fake_llm_job
    # time only the vocoder calls
    hift_t = []
    orig_inf, orig_chunk = model.hift.inference, model.hift.inference_chunk

    def timed(fn):
        def w(*a, **k):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            r = fn(*a, **k)
            torch.cuda.synchronize(); hift_t.append((time.perf_counter() - t0) * 1e3)
            return r
        return w
    model.hift.inference, model.hift.inference_chunk = timed(orig_inf), timed(orig_chunk)
    model.token_hop_len = 25  # tts() mutates it; reset per utterance
    outs = [o['tts_speech'] for o in model.tts(flow_embedding=cache['flow_embedding'],
                                               flow_prompt_speech_token=cache['flow_prompt_speech_token'],
                                               prompt_speech_feat=cache['prompt_speech_feat'], stream=True)]
    model.hift.inference, model.hift.inference_chunk = orig_inf, orig_chunk
    return torch.cat(outs, dim=1), hift_t


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    models = {'incremental': build('incremental'), 'recompute': build('legacy')}
    paths = sorted(glob.glob(os.path.join(CACHE_DIR, '*.pt')))[:N]
    for p in paths:
        cache = torch.load(p, map_location='cpu', weights_only=True)
        res = {}
        for name, m in models.items():
            torch.manual_seed(0)
            res[name] = stream(m, cache)
        a, ta = res['incremental']
        b, tb = res['recompute']
        print(f"{os.path.basename(p)[:-3]} tokens={cache['speech_token'].shape[1]} len {a.shape[1]} / {b.shape[1]} "
              f"snr incr-vs-recompute {snr_db(b, a):.1f} dB | hift ms "
              f"incremental={[round(x) for x in ta]} recompute={[round(x) for x in tb]}", flush=True)
    # empty final chunk edge case on the raw API
    h = models['incremental'].hift
    mel = torch.load(sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'exp', 'mels', '*.pt')))[0], map_location='cuda')
    with torch.inference_mode():
        ref, _ = h.inference(mel, finalize=True)
        st = h.new_stream_state()
        w = torch.cat([h.inference_chunk(mel[:, :, :100], st), h.inference_chunk(mel[:, :, 100:], st),
                       h.inference_chunk(mel[:, :, :0], st, finalize=True)], dim=1)
        print('empty finalize chunk: len', w.shape[1], ref.shape[1], 'snr %.1f' % snr_db(ref, w))
        st = h.new_stream_state()
        w = torch.cat([h.inference_chunk(mel[:, :, i:i + 1], st, finalize=i == mel.shape[2] - 1) for i in range(mel.shape[2])], dim=1)
        print('1-frame chunks: len', w.shape[1], ref.shape[1], 'snr %.1f' % snr_db(ref, w))
