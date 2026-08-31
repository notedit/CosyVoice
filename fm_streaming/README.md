# FM Streaming & Few-Step Generation for CosyVoice3

Design proposal for upgrading the CosyVoice3 flow-matching (token2mel) module
along three coupled axes, while keeping the LLM, speech tokenizer and HiFT
vocoder frozen:

1. **Streaming attention (CWA)** — chunk-aligned causal attention with a
   bounded sliding window plus a prompt *anchor* (attention-sink style),
   trained and inferred with the same mask, enabling constant per-chunk
   compute (L1: bounded-window recompute) and bit-exact incremental KV-cache
   inference (L2).
2. **Cross-attention injection of semantic latents** — a 25 Hz causal
   `SemanticEncoder` over speech tokens (optionally fused with LLM hidden
   states) injected into every other DiT block via zero-init gated
   cross-attention (Flamingo-style), so the 22-layer acoustic renderer
   re-reads prosody/emotion cues at every depth instead of a single
   input-level channel concat. Warm-startable from the released checkpoint.
3. **Few-step distillation** — CFG-internalization (20→10 NFE), consistency
   distillation (→2–4 NFE fallback tier), then DMD2 + light mel-GAN for the
   1-NFE flagship. One-step + single-branch is what makes the per-step KV
   cache memory-feasible (~40 MB/session vs ~800 MB today).

Estimated impact: ~20× (no cache) to ~100× (with cache) less flow compute per
chunk at steady state, constant instead of growing per-chunk latency, and
first-packet latency dropping from ~0.5–0.7 s to ~0.25–0.35 s (LLM-bound).

**Full design document (Chinese): [DESIGN.zh.md](DESIGN.zh.md)** — includes a
code-level analysis of the current bottlenecks, mask/RoPE/cache specifications,
distillation losses, training stages with go/no-go gates, latency/memory
budgets, an evaluation plan reusing the `flow_grpo/` reward stack, risks, and
a file-level implementation plan.

Status: design only — no model/runtime code is changed by this directory yet.
