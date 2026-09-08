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
"""Incremental (chunk-level) inference for CausalHiFTGenerator.

`CausalHiFTGenerator.inference(mel, finalize=False)` is numerically chunk-safe but
re-runs the whole vocoder on the full mel prefix at every call, so streaming a
T-frame utterance in N chunks costs O(N * T). This module keeps the per-layer
left context (the `cache` argument every causal conv already accepts), the NSF
phase accumulator, the source STFT tail and the iSTFT overlap-add carry, so every
chunk only pays for its own frames plus a fixed 7-frame look-ahead.

Frame bookkeeping (mel rate, 480 samples per frame for the 24 kHz model):
  f0 predictor  : first conv is right-causal, kernel 4 -> 3-frame look-ahead
  conv_pre      : right-causal, kernel conv_pre_look_right + 1 -> 4-frame look-ahead
  source STFT   : frame f needs source samples [4f - 8, 4f + 8); 120 * n_x + 1 frames
                  are consumed for n_x conv_pre frames (the +1 is the reflection pad
                  applied before the last fusion stage)
  iSTFT         : overlap-add with n_fft 16 / hop 4; a frame is complete once the
                  three following frames have been added, i.e. we emit 4 * F samples
                  per F new frames and keep a 12-sample carry.
The output of feeding the same mel chunk by chunk equals `inference(mel)` on the
full sequence up to floating point summation order.

A step-by-step walk-through of each stage (with diagrams) is in
hift_streaming/DESIGN.zh.md.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from cosyvoice.transformer.convolution import CausalConv1d, CausalConv1dDownSample, CausalConv1dUpsample


class HiFTStreamState:
    """Per-utterance state for `CausalHiFTGenerator.inference_chunk`."""

    def __init__(self):
        self.conv_cache: Dict[int, torch.Tensor] = {}   # left context per causal conv (keyed by id)
        self.down_buf: Dict[int, torch.Tensor] = {}     # unconsumed input per strided downsample conv
        self.pending_f0: Optional[torch.Tensor] = None  # mel frames (float64) still waiting for f0 look-ahead
        self.pending_pre: Optional[torch.Tensor] = None  # mel frames waiting for conv_pre look-ahead
        self.n_f0 = 0            # f0 frames produced so far
        self.n_x = 0             # conv_pre frames produced so far
        self.phase: Optional[torch.Tensor] = None  # (B, harmonics + 1) running sum of rad values
        self.s_buf: Optional[torch.Tensor] = None  # padded source samples not yet consumed by the STFT
        self.s_buf_start = 0     # padded-sequence index of s_buf[..., 0]
        self.s_total = 0         # source samples produced so far (unpadded)
        self.n_stft = 0          # source STFT frames produced so far
        self.ola_num: Optional[torch.Tensor] = None  # (B, n_fft - hop) numerator carry of the iSTFT
        self.ola_env: Optional[torch.Tensor] = None  # (n_fft - hop) window-envelope carry
        self.n_emitted = 0       # samples returned so far (after dropping the n_fft // 2 center pad)
        self.finalize_real_frames: Optional[int] = None  # set while finalizing a zero-padded last chunk
        self.finished = False


def _left_conv(conv: CausalConv1d, x: torch.Tensor, state: HiFTStreamState) -> torch.Tensor:
    """Left-causal conv with carried context; x is (B, C, T) of any T >= 0."""
    p = conv.causal_padding
    key = id(conv)
    cache = state.conv_cache.get(key)
    if cache is None:
        cache = x.new_zeros(x.shape[0], x.shape[1], p)
    if x.shape[2] == 0:
        return x.new_zeros(x.shape[0], conv.out_channels, 0)
    out = conv(x, cache)
    if p > 0:
        state.conv_cache[key] = torch.cat([cache, x], dim=2)[:, :, -p:]
    return out


def _up_conv(conv: CausalConv1dUpsample, x: torch.Tensor, state: HiFTStreamState) -> torch.Tensor:
    """Nearest upsample + left-causal conv; the cache lives in the upsampled domain.

    The conv sees `conv.upsample(x)`, so its left context is the tail of the *upsampled*
    signal: caching the tail of `x` instead would be short by a factor of the upsample rate.
    """
    p = conv.causal_padding
    key = id(conv)
    x_up = conv.upsample(x)
    cache = state.conv_cache.get(key)
    if cache is None:
        cache = x_up.new_zeros(x_up.shape[0], x_up.shape[1], p)
    if x.shape[2] == 0:
        return x.new_zeros(x.shape[0], conv.out_channels, 0)
    out = conv(x, cache)
    state.conv_cache[key] = torch.cat([cache, x_up], dim=2)[:, :, -p:]
    return out


def _down_conv(conv: torch.nn.Conv1d, x: torch.Tensor, state: HiFTStreamState) -> torch.Tensor:
    """Strided left-causal conv (CausalConv1dDownSample) fed from an input buffer.

    With stride > 1 the number of new inputs is not a multiple of the stride, so a
    fixed-length cache is not enough: keep an input buffer, consume as many whole strides as
    it holds and leave the remainder for the next chunk. In steady state the remainder is
    constant (15 frames for the 30/15 stage, 3 for the 6/3 stage) - it is the +1 STFT frame
    of the stream start carried through the two stages - so every steady-state chunk has the
    same shapes throughout.
    """
    if isinstance(conv, CausalConv1d):  # 1x1 stage
        return _left_conv(conv, x, state)
    assert isinstance(conv, CausalConv1dDownSample)
    k, s = conv.kernel_size[0], conv.stride[0]
    key = id(conv)
    buf = state.down_buf.get(key)
    if buf is None:
        buf = x.new_zeros(x.shape[0], x.shape[1], conv.causal_padding)
    buf = torch.cat([buf, x], dim=2)
    n_out = (buf.shape[2] - k) // s + 1 if buf.shape[2] >= k else 0
    if n_out > 0:
        out = torch.nn.Conv1d.forward(conv, buf[:, :, :s * (n_out - 1) + k])
    else:
        out = x.new_zeros(x.shape[0], conv.out_channels, 0)
    state.down_buf[key] = buf[:, :, s * n_out:]
    return out


def _window(gen, like: torch.Tensor) -> torch.Tensor:
    """gen.stft_window on like's device/dtype, cached (a host->device copy per call is
    both slower and a synchronisation point)."""
    cache = gen.__dict__.setdefault('_stft_window_cache', {})
    key = (like.device, like.dtype)
    if key not in cache:
        cache[key] = gen.stft_window.to(device=like.device, dtype=like.dtype)
    return cache[key]


def _resblock(block, x: torch.Tensor, state: HiFTStreamState) -> torch.Tensor:
    for idx in range(len(block.convs1)):
        xt = block.activations1[idx](x)
        xt = _left_conv(block.convs1[idx], xt, state)
        xt = block.activations2[idx](xt)
        xt = _left_conv(block.convs2[idx], xt, state)
        x = xt + x
    return x


def _f0_chunk(gen, mel: torch.Tensor, state: HiFTStreamState, finalize: bool):
    """Runs the causal f0 predictor on new mel frames; returns f0 (B, n) and the
    float32 mel frames those f0 values cover (the ones conv_pre may now consume)."""
    fp = gen.f0_predictor
    x = mel.to(torch.float64)
    if state.pending_f0 is not None:
        x = torch.cat([state.pending_f0, x], dim=2)
    first = fp.condnet[0]
    la = first.causal_padding
    if finalize:
        covered, state.pending_f0 = x, None
        h = first(covered) if covered.shape[2] > 0 else None
    elif x.shape[2] > la:
        covered, state.pending_f0 = x[:, :, :-la], x[:, :, -la:]
        h = first(covered, x[:, :, -la:])
    else:
        state.pending_f0 = x
        covered, h = x[:, :, :0], None
    if h is None:
        return mel.new_zeros(mel.shape[0], 0), covered.to(mel.dtype)
    for i in range(1, len(fp.condnet)):
        layer = fp.condnet[i]
        h = _left_conv(layer, h, state) if isinstance(layer, CausalConv1d) else layer(h)
    f0 = torch.abs(fp.classifier(h.transpose(1, 2)).squeeze(-1)).to(mel.dtype)
    return f0, covered.to(mel.dtype)


def _source_chunk(gen, f0: torch.Tensor, state: HiFTStreamState) -> torch.Tensor:
    """NSF source for new f0 frames, replicating SourceModuleHnNSF / SineGen2 in causal
    eval mode with the phase accumulator and noise-buffer offset carried over."""
    m_source = gen.m_source
    sg = m_source.l_sin_gen
    B, n = f0.shape
    scale = int(sg.upsample_scale)
    if n == 0:
        return f0.new_zeros(B, 1, 0)
    # sample-level f0 (nearest upsample, as gen.f0_upsamp does)
    f0_s = gen.f0_upsamp(f0[:, None]).transpose(1, 2)  # (B, L, 1)
    harmonics = torch.arange(1, sg.harmonic_num + 2, device=f0.device, dtype=f0.dtype)[None, None, :]
    fn = f0_s * harmonics  # (B, L, H)
    rad = (fn / sg.sampling_rate) % 1
    if state.n_f0 == 0:
        rad[:, 0, :] = rad[:, 0, :] + sg.rand_ini.to(rad)
    rad = F.interpolate(rad.transpose(1, 2), scale_factor=1 / scale, mode='linear').transpose(1, 2)  # (B, n, H)
    # NOTE the reference accumulates the phase in float32 over the whole utterance
    # (phase = 2 * pi * scale * cumsum(rad)); at 2e6 rad a float32 ulp is 0.25 rad, so
    # the reference's own sines are rounding-limited after ~10 s. Here the sum is kept
    # in float64 and wrapped modulo 1 (scale is an integer, so sin(2 pi scale cum) is
    # unchanged), which stays exact for arbitrarily long streams.
    cum = torch.cumsum(rad.to(torch.float64), dim=1)
    if state.phase is not None:
        cum = cum + state.phase[:, None, :]
    state.phase = torch.remainder(cum[:, -1, :], 1.0)
    frac = torch.remainder(cum * scale, 1.0).to(rad.dtype) * (2 * np.pi)
    phase = F.interpolate(frac.transpose(1, 2), scale_factor=scale, mode='nearest').transpose(1, 2)
    sines = torch.sin(phase) * sg.sine_amp
    uv = (f0_s > sg.voiced_threshold).to(sines.dtype)
    noise_amp = uv * sg.noise_std + (1 - uv) * sg.sine_amp / 3
    off = state.n_f0 * scale
    L = n * scale
    assert off + L <= sg.sine_waves.shape[1], 'utterance longer than the fixed noise buffer of SineGen2'
    noise = noise_amp * sg.sine_waves[:, off:off + L].to(sines)
    sine_waves = sines * uv + noise
    sine_merge = m_source.l_tanh(m_source.l_linear(sine_waves))  # (B, L, 1)
    return sine_merge.transpose(1, 2)  # (B, 1, L)


def _stft_chunk(gen, state: HiFTStreamState, n_frames_total: int, finalize: bool) -> torch.Tensor:
    """Source STFT frames [state.n_stft, n_frames_total) from the buffered source.

    torch.stft(center=True) puts frame f over source samples [hop*f - n_fft//2, hop*f + n_fft//2),
    i.e. [4f-8, 4f+8) here. `s_buf` holds the not-yet-consumed source tail and `s_buf_start` is
    its index in that padded sequence, so frame f starts at `hop*f - s_buf_start` inside the
    buffer; everything before the next frame's start is dropped after each chunk (1924 samples
    remain in steady state). The reflect padding is reproduced explicitly: at the stream start
    in _chunk_core, on the right here when finalizing.
    """
    n_fft, hop = gen.istft_params['n_fft'], gen.istft_params['hop_len']
    half = n_fft // 2
    if finalize and state.s_buf is not None and state.s_total > half:
        # right reflect pad; s_buf must still hold the last `half + 1` samples
        tail = state.s_buf[:, :, -(half + 1):]
        state.s_buf = torch.cat([state.s_buf, tail.flip(-1)[:, :, 1:]], dim=2)
    new = n_frames_total - state.n_stft
    if new <= 0 or state.s_buf is None:
        B = state.s_buf.shape[0] if state.s_buf is not None else 1
        return gen.conv_post.weight.new_zeros(B, n_fft + 2, 0)
    avail = state.s_buf_start + state.s_buf.shape[2]
    assert state.s_buf_start <= hop * state.n_stft, 'source buffer was trimmed past the next frame'
    assert avail >= hop * (n_frames_total - 1) + n_fft, 'not enough source samples for the requested STFT frames'
    start = hop * state.n_stft - state.s_buf_start
    seg = state.s_buf[:, :, start:start + hop * (new - 1) + n_fft]
    frames = seg.squeeze(1).unfold(-1, n_fft, hop) * _window(gen, seg)  # (B, new, n_fft)
    spec = torch.fft.rfft(frames, dim=-1)  # (B, new, n_fft//2+1)
    spec = torch.view_as_real(spec)  # (B, new, F, 2)
    out = torch.cat([spec[..., 0], spec[..., 1]], dim=2).transpose(1, 2)  # (B, n_fft+2, new)
    state.n_stft = n_frames_total
    # keep what the next frame needs
    keep_from = hop * state.n_stft - state.s_buf_start
    state.s_buf = state.s_buf[:, :, keep_from:]
    state.s_buf_start += keep_from
    return out


def _istft_chunk(gen, magnitude: torch.Tensor, phase: torch.Tensor, state: HiFTStreamState, finalize: bool) -> torch.Tensor:
    """Streaming inverse of torch.istft(center=True) with overlap-add carry.

    n_fft // hop = 4 frames overlap every output sample, so a sample is only final once the
    three following frames have been added: F new frames complete hop*F samples and leave an
    (n_fft - hop) = 12 sample carry in `ola_num` (numerator) and `ola_env` (window-square
    envelope, needed because istft divides by it). Two one-off corrections match the
    full-sequence length: the n_fft//2 samples of the center padding are dropped at the very
    start (tracked by `n_emitted`), and `hop` extra samples are flushed from the carry when
    finalizing. Total: 4*(120*T + 1) - 8 + 4 = 480*T samples for T mel frames.
    """
    n_fft, hop = gen.istft_params['n_fft'], gen.istft_params['hop_len']
    half = n_fft // 2
    B, _, Fn = magnitude.shape
    window = _window(gen, magnitude)
    if Fn > 0:
        magnitude = torch.clip(magnitude, max=1e2)
        spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase)).transpose(1, 2)  # (B, Fn, F)
        frames = torch.fft.irfft(spec, n=n_fft, dim=-1) * window  # (B, Fn, n_fft)
        n_seg = n_fft // hop
        y = frames.reshape(B, Fn, n_seg, hop)
        w2 = (window * window).reshape(n_seg, hop)
        num = magnitude.new_zeros(B, hop * Fn + n_fft - hop)
        env = magnitude.new_zeros(hop * Fn + n_fft - hop)
        for j in range(n_seg):
            num[:, hop * j:hop * j + hop * Fn] += y[:, :, j, :].reshape(B, hop * Fn)
            env[hop * j:hop * j + hop * Fn] += w2[j].repeat(Fn)
        if state.ola_num is not None:
            num[:, :n_fft - hop] += state.ola_num
            env[:n_fft - hop] += state.ola_env
        done = num[:, :hop * Fn] / env[:hop * Fn]
        state.ola_num, state.ola_env = num[:, hop * Fn:], env[hop * Fn:]
    else:
        done = magnitude.new_zeros(B, 0)
    if finalize and state.ola_num is not None:
        # torch.istft returns (n_frames - 1) * hop samples starting at index n_fft // 2,
        # i.e. `hop` samples beyond the last fully overlapped one
        done = torch.cat([done, state.ola_num[:, :hop] / state.ola_env[:hop]], dim=1)
    # drop the n_fft // 2 center padding once, at the start of the stream
    skip = max(0, half - state.n_emitted)
    out = done[:, skip:]
    state.n_emitted += done.shape[1]
    return out


@torch.inference_mode()
def inference_chunk(gen, speech_feat: torch.Tensor, state: HiFTStreamState, finalize: bool = False,
                    finalize_pad_multiple: Optional[int] = None) -> torch.Tensor:
    """Feed new mel frames (B, 80, T_new) and return the newly completed waveform (B, S_new).

    Concatenating the outputs over all chunks (the last one with finalize=True)
    reproduces `gen.inference(full_mel, finalize=True)`.

    finalize_pad_multiple: on the final chunk, zero-pad the mel so that the chunk length
    is a multiple of this value and trim the waveform back. cuDNN builds (and caches) an
    execution plan per input shape, which costs tens of ms for a never-seen shape; a
    fixed set of chunk shapes avoids that on the last chunk. Only the last few samples
    (the STFT reflect pad of the source) can differ from the unpadded result.
    """
    assert not state.finished, 'stream already finalized'
    gen.f0_predictor.to(torch.float64)
    n_fft, hop = gen.istft_params['n_fft'], gen.istft_params['hop_len']
    up = int(np.prod(gen.upsample_rates))
    if finalize and finalize_pad_multiple:
        pad = (-speech_feat.shape[2]) % finalize_pad_multiple
        if pad:
            n_real = state.n_f0 + (state.pending_f0.shape[2] if state.pending_f0 is not None else 0) + speech_feat.shape[2]
            state.finalize_real_frames = n_real
            wav = inference_chunk(gen, F.pad(speech_feat, (0, pad)), state, finalize=True)
            # state.n_emitted counts the n_fft // 2 center-pad samples that were dropped
            emitted_before = state.n_emitted - wav.shape[1] - n_fft // 2
            return wav[:, :n_real * up * hop - emitted_before]
    return _chunk_core(gen, speech_feat, state, finalize)


def _chunk_core(gen, speech_feat: torch.Tensor, state: HiFTStreamState, finalize: bool) -> torch.Tensor:
    """One chunk through the whole vocoder. Rates, per mel frame fed in (steady state):

        f0 1 frame -> source 480 samples -> source STFT 120 frames
        conv_pre 1 frame -> ups 8 -> 40 -> 120 frames -> iSTFT 480 samples

    The source branch is downsampled back onto the main branch at each stage (120 -> 8 -> 40
    -> 120 by source_downs stride 15 / 3 / 1).
    """
    n_fft, hop = gen.istft_params['n_fft'], gen.istft_params['hop_len']
    up = int(np.prod(gen.upsample_rates))
    # mel -> f0 -> source
    f0, covered = _f0_chunk(gen, speech_feat, state, finalize)
    s = _source_chunk(gen, f0, state)
    if finalize and state.finalize_real_frames is not None and s.shape[2] > 0:
        # zero-padded last chunk: keep the source of the real frames only and continue it
        # the way torch.stft(center=True, pad_mode='reflect') pads the full-sequence source,
        # so the frames of the real region are bit-identical to the unpadded finalize
        n_real_new = (state.finalize_real_frames - state.n_f0) * up * hop
        assert 0 < n_real_new <= s.shape[2]
        half = n_fft // 2
        real = s[:, :, :n_real_new]
        if real.shape[2] > half:
            refl = real[:, :, -(half + 1):].flip(-1)[:, :, 1:]
        else:  # not enough samples in this chunk alone; take them from the source history
            hist = torch.cat([state.s_buf[:, :, -(half + 1):], real], dim=2) if state.s_buf is not None else real
            refl = hist[:, :, -(half + 1):].flip(-1)[:, :, 1:]
        s = torch.cat([real, refl, s.new_zeros(s.shape[0], 1, s.shape[2] - n_real_new - refl.shape[2])], dim=2)
    state.n_f0 += f0.shape[1]
    if s.shape[2] > 0:
        if state.s_buf is None:
            # left reflect pad of the source, as torch.stft(center=True) does
            assert s.shape[2] > n_fft // 2, 'first source chunk shorter than the STFT half window'
            state.s_buf = torch.cat([s[:, :, 1:n_fft // 2 + 1].flip(-1), s], dim=2)
            state.s_buf_start = 0
        else:
            state.s_buf = torch.cat([state.s_buf, s], dim=2)
        state.s_total += s.shape[2]
    # conv_pre with right look-ahead
    x_in = covered
    if state.pending_pre is not None:
        x_in = torch.cat([state.pending_pre, x_in], dim=2)
    la = gen.conv_pre.causal_padding
    if finalize:
        x = gen.conv_pre(x_in) if x_in.shape[2] > 0 else x_in.new_zeros(x_in.shape[0], gen.conv_pre.out_channels, 0)
        state.pending_pre = None
    elif x_in.shape[2] > la:
        x = gen.conv_pre(x_in[:, :, :-la], x_in[:, :, -la:])
        state.pending_pre = x_in[:, :, -la:]
    else:
        state.pending_pre = x_in
        x = x_in.new_zeros(x_in.shape[0], gen.conv_pre.out_channels, 0)
    first_x = state.n_x == 0
    state.n_x += x.shape[2]
    # source STFT frames needed for these conv_pre frames: the main branch runs at up = 120
    # frames per conv_pre frame after the three upsamples, plus the single frame the
    # ReflectionPad(1, 0) before the last stage adds once at the start of the stream
    s_stft = _stft_chunk(gen, state, up * state.n_x + 1 if state.n_x > 0 else 0, finalize)
    if x.shape[2] == 0 and s_stft.shape[2] == 0:
        out = speech_feat.new_zeros(speech_feat.shape[0], 0)
        if finalize:
            state.finished = True
        return out
    for i in range(gen.num_upsamples):
        x = F.leaky_relu(x, gen.lrelu_slope)
        x = _up_conv(gen.ups[i], x, state)
        if i == gen.num_upsamples - 1 and first_x:
            x = gen.reflection_pad(x)
        si = _down_conv(gen.source_downs[i], s_stft, state)
        si = _resblock(gen.source_resblocks[i], si, state)
        assert si.shape[2] == x.shape[2], f'stage {i}: source {si.shape[2]} vs main {x.shape[2]}'
        x = x + si
        xs = None
        for j in range(gen.num_kernels):
            r = _resblock(gen.resblocks[i * gen.num_kernels + j], x, state)
            xs = r if xs is None else xs + r
        x = xs / gen.num_kernels
    x = F.leaky_relu(x)
    x = _left_conv(gen.conv_post, x, state)
    magnitude = torch.exp(x[:, :n_fft // 2 + 1, :])
    phase = torch.sin(x[:, n_fft // 2 + 1:, :])
    if finalize and state.finalize_real_frames is not None:
        # zero-padded last chunk: the overlap-add of the last real samples must only see the
        # frames the unpadded finalize would have (up * T_real + 1 in total)
        keep = up * state.finalize_real_frames + 1 - state.n_emitted // hop
        magnitude, phase = magnitude[:, :, :keep], phase[:, :, :keep]
    wav = _istft_chunk(gen, magnitude, phase, state, finalize)
    wav = torch.clamp(wav, -gen.audio_limit, gen.audio_limit)
    if finalize:
        state.finished = True
    return wav

