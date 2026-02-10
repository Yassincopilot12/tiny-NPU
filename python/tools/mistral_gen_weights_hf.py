#!/usr/bin/env python3
"""Generate LLaMA weights.bin + golden tokens from HuggingFace Mistral-300M weights.

Downloads ayousanz/mistral-300M (355M params, Mistral/LLaMA architecture with GQA),
slices each weight tensor to our tiny NPU dimensions, quantizes to int8,
runs golden inference, and outputs the same files as llama_gen_weights.py.

Mistral uses an identical architecture to LLaMA (RMSNorm, RoPE, GQA, SwiGLU)
with identical HuggingFace tensor naming, so the same extraction and golden
inference code works unchanged.

Outputs:
  weights.bin        - packed int8 weights (180800 bytes)
  prompt_tokens.txt  - initial prompt token IDs
  golden_tokens.txt  - greedy-generated token IDs
  golden_logits.bin  - int8 logits at each step [N_GEN, VOCAB_SIZE]

Dependencies: huggingface_hub, safetensors, numpy, torch
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from llama_map import (
    HIDDEN, N_Q_HEADS, N_KV_HEADS, HEAD_DIM, FFN_DIM, N_LAYERS,
    VOCAB_SIZE, MAX_SEQ,
    WTE_OFFSET, BLOCKS_OFFSET, LLAMA_BLOCK_SIZE,
    BLK_RMS1_GAMMA, BLK_WQ, BLK_WK, BLK_WV, BLK_WO,
    BLK_RMS2_GAMMA, BLK_W_GATE, BLK_W_UP, BLK_W_DOWN,
    LN_F_OFFSET, LM_HEAD_OFFSET, WEIGHTS_TOTAL,
    quantize_tensor, quantize_gamma,
    GEMM_SCALE, GEMM_SHIFT, GEMM_SHIFT_K16, GEMM_SHIFT_K128,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from golden.llama_infer_golden import TinyLLaMAGolden

# ── Mistral-300M source dimensions ───────────────────────────────────
HF_REPO       = "ayousanz/mistral-300M"
HF_HIDDEN     = 1024
HF_N_HEADS    = 16
HF_N_KV_HEADS = 8
HF_HEAD_DIM   = 64   # = 1024 / 16
HF_FFN_DIM    = 2400


def load_hf_tensors(repo_id):
    """Download and load Mistral safetensors as numpy arrays."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    print(f"Loaded model from: {path}")
    tensors = load_file(path)
    return tensors


def slice_qkv(weight, src_n_heads, src_head_dim, src_hidden,
               tgt_n_heads, tgt_head_dim, tgt_hidden):
    """Slice Q/K/V projection weight to our tiny dimensions.

    HF weight: [src_n_heads * src_head_dim, src_hidden]  (PyTorch row = output)
    Returns:   [tgt_n_heads, tgt_hidden, tgt_head_dim]   (our per-head format)
    """
    # Reshape to per-head: [src_n_heads, src_head_dim, src_hidden]
    w = weight.reshape(src_n_heads, src_head_dim, src_hidden)
    # Take first tgt_n_heads, first tgt_head_dim, first tgt_hidden
    w = w[:tgt_n_heads, :tgt_head_dim, :tgt_hidden]
    # Transpose to [tgt_n_heads, tgt_hidden, tgt_head_dim]
    w = np.transpose(w, (0, 2, 1))
    return w


def slice_o_proj(weight, src_head_dim, tgt_n_heads, tgt_head_dim, tgt_hidden):
    """Slice O-projection weight, gathering correct head indices.

    HF o_proj.weight: [src_hidden, src_n_heads * src_head_dim]
    Our Wo:           [tgt_hidden, tgt_hidden]  (since n_q_heads * head_dim = hidden)

    The input to O-proj is concatenated head outputs. We took heads 0..tgt_n_heads-1
    and dims 0..tgt_head_dim-1 from each, so we need to gather the correct source
    column indices.
    """
    # Transpose to [src_n_heads * src_head_dim, src_hidden]
    wt = weight.T  # [src_n_heads * src_head_dim, src_hidden]

    # Gather input indices corresponding to our selected heads/dims
    input_indices = []
    for h in range(tgt_n_heads):
        start = h * src_head_dim
        input_indices.extend(range(start, start + tgt_head_dim))

    # Slice: gather selected input dims, take first tgt_hidden output dims
    wt_sliced = wt[input_indices, :tgt_hidden]  # [tgt_hidden, tgt_hidden]
    return wt_sliced.T  # [tgt_hidden, tgt_hidden] — matches our Wo format


def extract_weights(tensors):
    """Extract and quantize all weights from HF tensors to our NPU format."""
    # ── Embedding ────────────────────────────────────────────────────
    wte_fp = tensors["model.embed_tokens.weight"][:VOCAB_SIZE, :HIDDEN]
    wte = quantize_tensor(wte_fp)

    # ── Per-layer weights ────────────────────────────────────────────
    blocks_weights = []
    for i in range(N_LAYERS):
        prefix = f"model.layers.{i}"
        bw = {}

        # RMSNorm 1
        bw['rms1_gamma'] = quantize_gamma(
            tensors[f"{prefix}.input_layernorm.weight"][:HIDDEN])

        # Q projection
        q_weight = tensors[f"{prefix}.self_attn.q_proj.weight"]
        q_sliced = slice_qkv(q_weight, HF_N_HEADS, HF_HEAD_DIM, HF_HIDDEN,
                             N_Q_HEADS, HEAD_DIM, HIDDEN)
        bw['Wq'] = np.zeros((N_Q_HEADS, HIDDEN, HEAD_DIM), dtype=np.int8)
        for h in range(N_Q_HEADS):
            bw['Wq'][h] = quantize_tensor(q_sliced[h])

        # K projection
        k_weight = tensors[f"{prefix}.self_attn.k_proj.weight"]
        k_sliced = slice_qkv(k_weight, HF_N_KV_HEADS, HF_HEAD_DIM, HF_HIDDEN,
                             N_KV_HEADS, HEAD_DIM, HIDDEN)
        bw['Wk'] = np.zeros((N_KV_HEADS, HIDDEN, HEAD_DIM), dtype=np.int8)
        for h in range(N_KV_HEADS):
            bw['Wk'][h] = quantize_tensor(k_sliced[h])

        # V projection
        v_weight = tensors[f"{prefix}.self_attn.v_proj.weight"]
        v_sliced = slice_qkv(v_weight, HF_N_KV_HEADS, HF_HEAD_DIM, HF_HIDDEN,
                             N_KV_HEADS, HEAD_DIM, HIDDEN)
        bw['Wv'] = np.zeros((N_KV_HEADS, HIDDEN, HEAD_DIM), dtype=np.int8)
        for h in range(N_KV_HEADS):
            bw['Wv'][h] = quantize_tensor(v_sliced[h])

        # O projection
        o_weight = tensors[f"{prefix}.self_attn.o_proj.weight"]
        o_sliced = slice_o_proj(o_weight, HF_HEAD_DIM,
                                N_Q_HEADS, HEAD_DIM, HIDDEN)
        bw['Wo'] = quantize_tensor(o_sliced)

        # RMSNorm 2
        bw['rms2_gamma'] = quantize_gamma(
            tensors[f"{prefix}.post_attention_layernorm.weight"][:HIDDEN])

        # MLP: gate, up, down
        # HF gate_proj.weight: [FFN_DIM, HIDDEN] → transpose then slice
        gate_w = tensors[f"{prefix}.mlp.gate_proj.weight"]
        bw['W_gate'] = quantize_tensor(gate_w.T[:HIDDEN, :FFN_DIM])

        up_w = tensors[f"{prefix}.mlp.up_proj.weight"]
        bw['W_up'] = quantize_tensor(up_w.T[:HIDDEN, :FFN_DIM])

        down_w = tensors[f"{prefix}.mlp.down_proj.weight"]
        bw['W_down'] = quantize_tensor(down_w.T[:FFN_DIM, :HIDDEN])

        blocks_weights.append(bw)

    # ── Final RMSNorm ────────────────────────────────────────────────
    ln_f_gamma = quantize_gamma(tensors["model.norm.weight"][:HIDDEN])

    # ── LM head ──────────────────────────────────────────────────────
    lm_head_w = quantize_tensor(
        tensors["lm_head.weight"][:VOCAB_SIZE, :HIDDEN])

    return wte, blocks_weights, ln_f_gamma, lm_head_w


def main():
    ap = argparse.ArgumentParser(
        description="Generate LLaMA demo weights from HuggingFace Mistral-300M")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--prompt", default="1,2,3,4",
                    help="Comma-separated prompt tokens")
    ap.add_argument("--gen-tokens", type=int, default=4,
                    help="Number of tokens to generate")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    prompt = [int(t) for t in args.prompt.split(",")]
    n_gen = args.gen_tokens

    print(f"Config: HIDDEN={HIDDEN} N_Q_HEADS={N_Q_HEADS} N_KV_HEADS={N_KV_HEADS}")
    print(f"        FFN_DIM={FFN_DIM} N_LAYERS={N_LAYERS} VOCAB={VOCAB_SIZE}")
    print(f"        prompt={prompt} gen_tokens={n_gen}")
    print(f"Source: {HF_REPO} (hidden={HF_HIDDEN}, heads={HF_N_HEADS}, "
          f"kv_heads={HF_N_KV_HEADS}, ffn={HF_FFN_DIM})")

    # ── Download and extract weights ─────────────────────────────────
    print("\nDownloading Mistral-300M weights...")
    tensors = load_hf_tensors(HF_REPO)
    print(f"Loaded {len(tensors)} tensors")

    print("Slicing and quantizing to NPU format...")
    wte, blocks_weights, ln_f_gamma, lm_head_w = extract_weights(tensors)

    # ── Pack weights.bin ─────────────────────────────────────────────
    buf = bytearray(WEIGHTS_TOTAL)

    def put(offset, arr):
        flat = arr.flatten().astype(np.int8)
        buf[offset:offset + len(flat)] = flat.tobytes()

    put(WTE_OFFSET, wte)

    for i in range(N_LAYERS):
        base = BLOCKS_OFFSET + i * LLAMA_BLOCK_SIZE
        bw = blocks_weights[i]
        put(base + BLK_RMS1_GAMMA, bw['rms1_gamma'])
        wq_blocked = np.concatenate([bw['Wq'][h] for h in range(N_Q_HEADS)], axis=0)
        put(base + BLK_WQ, wq_blocked)
        wk_blocked = np.concatenate([bw['Wk'][h] for h in range(N_KV_HEADS)], axis=0)
        put(base + BLK_WK, wk_blocked)
        wv_blocked = np.concatenate([bw['Wv'][h] for h in range(N_KV_HEADS)], axis=0)
        put(base + BLK_WV, wv_blocked)
        put(base + BLK_WO, bw['Wo'])
        put(base + BLK_RMS2_GAMMA, bw['rms2_gamma'])
        put(base + BLK_W_GATE, bw['W_gate'])
        put(base + BLK_W_UP, bw['W_up'])
        put(base + BLK_W_DOWN, bw['W_down'])

    put(LN_F_OFFSET, ln_f_gamma)
    put(LM_HEAD_OFFSET, lm_head_w)

    weights_path = os.path.join(args.outdir, "weights.bin")
    with open(weights_path, "wb") as f:
        f.write(buf)
    print(f"Wrote {len(buf)} bytes -> {weights_path}")

    # ── Run golden inference ─────────────────────────────────────────
    model = TinyLLaMAGolden(
        hidden=HIDDEN, n_q_heads=N_Q_HEADS, n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM, ffn_dim=FFN_DIM, max_seq=MAX_SEQ,
        scale=GEMM_SCALE, shift_k64=GEMM_SHIFT, shift_k16=GEMM_SHIFT_K16,
        shift_k128=GEMM_SHIFT_K128,
    )

    tokens = list(prompt)
    generated = []
    all_logits = []

    for step in range(n_gen):
        logits = model.forward(
            np.array(tokens, dtype=np.int32),
            wte, blocks_weights, ln_f_gamma, lm_head_w,
        )
        logits_flat = logits[0]
        all_logits.append(logits_flat.copy())

        next_tok = int(np.argmax(logits_flat))
        generated.append(next_tok)
        tokens.append(next_tok)

        print(f"  Step {step}: token={next_tok}  logits[0:8]={logits_flat[:8].tolist()}")

    # ── Write outputs ────────────────────────────────────────────────
    prompt_path = os.path.join(args.outdir, "prompt_tokens.txt")
    with open(prompt_path, "w") as f:
        for t in prompt:
            f.write(f"{t}\n")
    print(f"Prompt tokens -> {prompt_path}")

    golden_path = os.path.join(args.outdir, "golden_tokens.txt")
    with open(golden_path, "w") as f:
        for t in generated:
            f.write(f"{t}\n")
    print(f"Golden tokens ({len(generated)}): {generated} -> {golden_path}")

    logits_path = os.path.join(args.outdir, "golden_logits.bin")
    logits_arr = np.array(all_logits, dtype=np.int8)
    with open(logits_path, "wb") as f:
        f.write(logits_arr.tobytes())
    print(f"Golden logits [{logits_arr.shape}] -> {logits_path}")

    print("\nDone. Run: ./llama_demo_infer --datadir", args.outdir)


if __name__ == "__main__":
    main()
