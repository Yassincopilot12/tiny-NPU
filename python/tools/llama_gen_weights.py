#!/usr/bin/env python3
"""Generate random LLaMA weights.bin + golden tokens for NPU demo inference.

Outputs:
  weights.bin        - packed int8 weights (180800 bytes)
  prompt_tokens.txt  - initial prompt token IDs
  golden_tokens.txt  - greedy-generated token IDs
  golden_logits.bin  - int8 logits at each step [N_GEN, VOCAB_SIZE]
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
    BLK_BQ, BLK_BK, BLK_BV,
    LN_F_OFFSET, LM_HEAD_OFFSET, WEIGHTS_TOTAL,
    GEMM_SCALE, GEMM_SHIFT, GEMM_SHIFT_K16, GEMM_SHIFT_K128,
)

# Add golden module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from golden.llama_infer_golden import TinyLLaMAGolden


def main():
    ap = argparse.ArgumentParser(description="Generate LLaMA demo weights + golden")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default="1,2,3,4", help="Comma-separated prompt tokens")
    ap.add_argument("--gen-tokens", type=int, default=4, help="Number of tokens to generate")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    prompt = [int(t) for t in args.prompt.split(",")]
    n_gen = args.gen_tokens

    print(f"Config: HIDDEN={HIDDEN} N_Q_HEADS={N_Q_HEADS} N_KV_HEADS={N_KV_HEADS}")
    print(f"        FFN_DIM={FFN_DIM} N_LAYERS={N_LAYERS} VOCAB={VOCAB_SIZE}")
    print(f"        prompt={prompt} gen_tokens={n_gen}")

    # ── Generate random weights ──────────────────────────────────────
    # WTE: large range for embedding diversity
    wte = rng.randint(-32, 33, (VOCAB_SIZE, HIDDEN)).astype(np.int8)

    # Per-block weights
    blocks_weights = []
    for _ in range(N_LAYERS):
        bw = {}
        bw['rms1_gamma'] = rng.randint(-64, 64, HIDDEN).astype(np.int8)

        bw['Wq'] = np.zeros((N_Q_HEADS, HIDDEN, HEAD_DIM), dtype=np.int8)
        for h in range(N_Q_HEADS):
            bw['Wq'][h] = rng.randint(-4, 5, (HIDDEN, HEAD_DIM)).astype(np.int8)

        bw['Wk'] = np.zeros((N_KV_HEADS, HIDDEN, HEAD_DIM), dtype=np.int8)
        for h in range(N_KV_HEADS):
            bw['Wk'][h] = rng.randint(-4, 5, (HIDDEN, HEAD_DIM)).astype(np.int8)

        bw['Wv'] = np.zeros((N_KV_HEADS, HIDDEN, HEAD_DIM), dtype=np.int8)
        for h in range(N_KV_HEADS):
            bw['Wv'][h] = rng.randint(-4, 5, (HIDDEN, HEAD_DIM)).astype(np.int8)

        bw['Wo'] = rng.randint(-4, 5, (HIDDEN, HIDDEN)).astype(np.int8)
        bw['rms2_gamma'] = rng.randint(-64, 64, HIDDEN).astype(np.int8)
        bw['W_gate'] = rng.randint(-4, 5, (HIDDEN, FFN_DIM)).astype(np.int8)
        bw['W_up'] = rng.randint(-4, 5, (HIDDEN, FFN_DIM)).astype(np.int8)
        bw['W_down'] = rng.randint(-4, 5, (FFN_DIM, HIDDEN)).astype(np.int8)
        blocks_weights.append(bw)

    ln_f_gamma = rng.randint(-64, 64, HIDDEN).astype(np.int8)
    lm_head_w = rng.randint(-4, 5, (VOCAB_SIZE, HIDDEN)).astype(np.int8)

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
        # Wq: head-blocked [N_Q_HEADS * HIDDEN, HEAD_DIM]
        wq_blocked = np.concatenate([bw['Wq'][h] for h in range(N_Q_HEADS)], axis=0)
        put(base + BLK_WQ, wq_blocked)
        # Wk: head-blocked [N_KV_HEADS * HIDDEN, HEAD_DIM]
        wk_blocked = np.concatenate([bw['Wk'][h] for h in range(N_KV_HEADS)], axis=0)
        put(base + BLK_WK, wk_blocked)
        # Wv: head-blocked
        wv_blocked = np.concatenate([bw['Wv'][h] for h in range(N_KV_HEADS)], axis=0)
        put(base + BLK_WV, wv_blocked)
        put(base + BLK_WO, bw['Wo'])
        put(base + BLK_RMS2_GAMMA, bw['rms2_gamma'])
        put(base + BLK_W_GATE, bw['W_gate'])
        put(base + BLK_W_UP, bw['W_up'])
        put(base + BLK_W_DOWN, bw['W_down'])
        # Zero QKV bias (no bias for random LLaMA)
        put(base + BLK_BQ, np.zeros(N_Q_HEADS * HEAD_DIM, dtype=np.int8))
        put(base + BLK_BK, np.zeros(N_KV_HEADS * HEAD_DIM, dtype=np.int8))
        put(base + BLK_BV, np.zeros(N_KV_HEADS * HEAD_DIM, dtype=np.int8))

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
        # logits: [1, VOCAB_SIZE]
        logits_flat = logits[0]
        all_logits.append(logits_flat.copy())

        # Greedy argmax
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
