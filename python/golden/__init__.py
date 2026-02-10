"""Golden reference models for NPU verification."""
from .quant import *
from .gemm_ref import gemm_int8
from .softmax_ref import softmax_fixed
from .layernorm_ref import layernorm_fixed
from .gelu_ref import gelu_fixed
from .gpt2_block_ref import GPT2BlockRef, run_reference_block
from .rmsnorm_ref import rmsnorm_fixed
from .rope_ref import rope_fixed, make_rope_tables
from .silu_ref import silu_fixed
from .llama_infer_golden import TinyLLaMAGolden
