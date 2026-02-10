"""Reference LLaMA transformer block using fixed-point arithmetic."""
import numpy as np
from .quant import clamp_i8, requantize
from .gemm_ref import gemm_int8
from .softmax_ref import softmax_fixed
from .rmsnorm_ref import rmsnorm_fixed
from .rope_ref import rope_fixed, make_rope_tables
from .silu_ref import silu_fixed


class TinyLLaMAGolden:
    """Reference implementation of one LLaMA transformer block."""

    def __init__(self, hidden=64, n_q_heads=4, n_kv_heads=2, head_dim=16,
                 ffn_dim=128, max_seq=16, scale=1, shift_k64=9, shift_k16=7,
                 shift_k128=10, rope_base=10000.0):
        self.hidden = hidden
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.gqa_ratio = n_q_heads // n_kv_heads
        self.ffn_dim = ffn_dim
        self.max_seq = max_seq
        self.scale = scale
        self.shift_k64 = shift_k64
        self.shift_k16 = shift_k16
        self.shift_k128 = shift_k128

        # Generate RoPE tables
        self.sin_table, self.cos_table = make_rope_tables(
            max_seq, head_dim, base=rope_base)

        # Initialize random int8 weights
        rng = np.random.RandomState(42)

        # Per-head Q weights [n_q_heads, hidden, head_dim]
        self.Wq = np.zeros((n_q_heads, hidden, head_dim), dtype=np.int8)
        for h in range(n_q_heads):
            self.Wq[h] = rng.randint(-4, 5, (hidden, head_dim)).astype(np.int8)

        # Per-head K weights [n_kv_heads, hidden, head_dim]
        self.Wk = np.zeros((n_kv_heads, hidden, head_dim), dtype=np.int8)
        for h in range(n_kv_heads):
            self.Wk[h] = rng.randint(-4, 5, (hidden, head_dim)).astype(np.int8)

        # Per-head V weights [n_kv_heads, hidden, head_dim]
        self.Wv = np.zeros((n_kv_heads, hidden, head_dim), dtype=np.int8)
        for h in range(n_kv_heads):
            self.Wv[h] = rng.randint(-4, 5, (hidden, head_dim)).astype(np.int8)

        self.Wo = rng.randint(-4, 5, (hidden, hidden)).astype(np.int8)

        self.rms1_gamma = rng.randint(-64, 64, hidden).astype(np.int8)
        self.rms2_gamma = rng.randint(-64, 64, hidden).astype(np.int8)

        self.W_gate = rng.randint(-4, 5, (hidden, ffn_dim)).astype(np.int8)
        self.W_up = rng.randint(-4, 5, (hidden, ffn_dim)).astype(np.int8)
        self.W_down = rng.randint(-4, 5, (ffn_dim, hidden)).astype(np.int8)

    def run_block(self, x):
        """
        Full LLaMA transformer block forward pass.
        x: int8 [seq_len, hidden]
        Returns: int8 [seq_len, hidden]
        """
        S, H = x.shape
        assert H == self.hidden

        # 1. RMSNorm 1
        rms1_out = np.zeros_like(x)
        for s in range(S):
            rms1_out[s] = rmsnorm_fixed(x[s], self.rms1_gamma)

        # 2. Multi-head GQA attention with RoPE
        attn = np.zeros((S, self.hidden), dtype=np.int8)

        for h in range(self.n_q_heads):
            kv_h = h // self.gqa_ratio

            # Q/K/V projections
            q_h = gemm_int8(rms1_out, self.Wq[h],
                            scale=self.scale, shift=self.shift_k64)
            k_h = gemm_int8(rms1_out, self.Wk[kv_h],
                            scale=self.scale, shift=self.shift_k64)
            v_h = gemm_int8(rms1_out, self.Wv[kv_h],
                            scale=self.scale, shift=self.shift_k64)

            # Apply RoPE to Q and K
            q_h = rope_fixed(q_h, self.sin_table, self.cos_table, pos_offset=0)
            k_h = rope_fixed(k_h, self.sin_table, self.cos_table, pos_offset=0)

            # Attention scores: Q * K^T
            scores = gemm_int8(q_h, k_h, transpose_b=True,
                               scale=self.scale, shift=self.shift_k16)

            # Softmax with causal mask
            probs = np.zeros((S, S), dtype=np.int8)
            for s in range(S):
                causal = np.arange(S) <= s
                probs[s] = softmax_fixed(scores[s], causal_mask=causal)

            # Context: P * V
            attn_h = gemm_int8(probs, v_h,
                               scale=self.scale, shift=self.shift_k16)

            # Scatter to concat buffer
            attn[:, h * self.head_dim:(h + 1) * self.head_dim] = attn_h

        # 3. Output projection
        wo_out = gemm_int8(attn, self.Wo, scale=self.scale, shift=self.shift_k64)

        # 4. Residual add
        x2 = clamp_i8(x.astype(np.int16) + wo_out.astype(np.int16))

        # 5. RMSNorm 2
        rms2_out = np.zeros_like(x2)
        for s in range(S):
            rms2_out[s] = rmsnorm_fixed(x2[s], self.rms2_gamma)

        # 6. SwiGLU FFN
        # Gate projection
        ffn_gate = gemm_int8(rms2_out, self.W_gate,
                             scale=self.scale, shift=self.shift_k64)

        # Up projection
        ffn_up = gemm_int8(rms2_out, self.W_up,
                           scale=self.scale, shift=self.shift_k64)

        # SiLU on gate
        ffn_gate = silu_fixed(ffn_gate)

        # Elementwise multiply (gate * up), requantize with >>7 rounding
        ffn_mul = np.zeros_like(ffn_gate)
        for idx in np.ndindex(ffn_gate.shape):
            mul_raw = np.int16(ffn_gate[idx]) * np.int16(ffn_up[idx])
            mul_round = (mul_raw + 64) >> 7
            ffn_mul[idx] = np.clip(mul_round, -128, 127).astype(np.int8)

        # Down projection (K=128)
        ffn_down = gemm_int8(ffn_mul, self.W_down,
                             scale=self.scale, shift=self.shift_k128)

        # 7. Residual add
        x_out = clamp_i8(x2.astype(np.int16) + ffn_down.astype(np.int16))

        return x_out

    def embed(self, tokens, wte):
        """
        Embed tokens using WTE only (no WPE, RoPE handles position).
        tokens: int array [seq_len]
        wte: int8 [vocab_size, hidden]
        Returns: int8 [seq_len, hidden]
        """
        return wte[tokens].astype(np.int8)

    def forward(self, tokens, wte, blocks_weights, ln_f_gamma, lm_head):
        """
        Full LLaMA inference: embed -> blocks -> final RMSNorm -> lm_head.
        Returns: int8 logits [1, vocab_size]
        """
        x = self.embed(tokens, wte)

        for block_w in blocks_weights:
            self.rms1_gamma = block_w['rms1_gamma']
            self.rms2_gamma = block_w['rms2_gamma']
            self.Wq = block_w['Wq']
            self.Wk = block_w['Wk']
            self.Wv = block_w['Wv']
            self.Wo = block_w['Wo']
            self.W_gate = block_w['W_gate']
            self.W_up = block_w['W_up']
            self.W_down = block_w['W_down']
            x = self.run_block(x)

        # Final RMSNorm
        x_final = np.zeros_like(x)
        for s in range(x.shape[0]):
            x_final[s] = rmsnorm_fixed(x[s], ln_f_gamma)

        # LM head (last token only)
        last_hidden = x_final[-1:, :]  # [1, hidden]
        logits = gemm_int8(last_hidden, lm_head, transpose_b=True,
                           scale=self.scale, shift=self.shift_k64)

        return logits


def run_reference_block(hidden=64, n_q_heads=4, n_kv_heads=2, head_dim=16,
                        ffn_dim=128, seq_len=16, seed=42):
    """Run a reference LLaMA block and return inputs/outputs for verification."""
    rng = np.random.RandomState(seed)
    x = rng.randint(-64, 64, (seq_len, hidden)).astype(np.int8)

    block = TinyLLaMAGolden(hidden=hidden, n_q_heads=n_q_heads,
                            n_kv_heads=n_kv_heads, head_dim=head_dim,
                            ffn_dim=ffn_dim, max_seq=seq_len)
    x_out = block.run_block(x)

    return {
        'input': x,
        'output': x_out,
        'weights': {
            'Wq': block.Wq,
            'Wk': block.Wk,
            'Wv': block.Wv,
            'Wo': block.Wo,
            'W_gate': block.W_gate,
            'W_up': block.W_up,
            'W_down': block.W_down,
            'rms1_gamma': block.rms1_gamma,
            'rms2_gamma': block.rms2_gamma,
        },
        'rope_tables': {
            'sin': block.sin_table,
            'cos': block.cos_table,
        }
    }


if __name__ == '__main__':
    result = run_reference_block()
    print(f"Input shape: {result['input'].shape}")
    print(f"Output shape: {result['output'].shape}")
    print(f"Output sample: {result['output'][0, :8]}")
