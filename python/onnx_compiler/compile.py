#!/usr/bin/env python3
"""
ONNX → NPU Graph Mode Compiler

Compiles an ONNX model (MLP or CNN) into:
  - program.bin   : Graph ISA instructions
  - tdesc.bin     : Tensor descriptor table
  - ddr_image.bin : DDR memory image (weights, inputs)
  - golden.bin    : Expected output (int8)
  - manifest.json : Metadata

Supports: Gemm, Relu, Conv (via im2col), Reshape/Flatten

Usage:
  python compile.py --model models/mlp_32_16_8.onnx --outdir build/graph
"""

import argparse
import json
import math
import os
import struct
import sys
from collections import OrderedDict

import numpy as np

try:
    import onnx
    from onnx import numpy_helper, shape_inference
except ImportError:
    print("ERROR: onnx package required. Run: pip install onnx")
    sys.exit(1)

# =========================================================================
# Constants matching include/graph_isa.h and include/ddr_graph.h
# =========================================================================
OP_G_END         = 0x00
OP_G_DMA_LOAD    = 0x10
OP_G_DMA_STORE   = 0x11
OP_G_DMA_STRIDED = 0x12
OP_G_GEMM        = 0x20
OP_G_EW_ADD      = 0x30
OP_G_EW_MUL      = 0x31
OP_G_EW_SUB      = 0x32
OP_G_RELU        = 0x38
OP_G_SOFTMAX     = 0x40

GFLAG_TRANSPOSE_B = 0x01
GFLAG_BIAS_EN     = 0x02
GFLAG_REQUANT     = 0x04
GFLAG_RELU        = 0x08

DDR_GRAPH_DATA_BASE = 0x00300000
DDR_GRAPH_IO_BASE   = 0x00400000

# =========================================================================
# Instruction encoding
# =========================================================================
def encode_instr(opcode, flags=0, dst=0, src0=0, src1=0, src2=0,
                 imm0=0, imm1=0, imm2=0):
    """Encode a 128-bit (16-byte) graph instruction."""
    word0 = (src0 << 24) | (dst << 16) | (flags << 8) | opcode
    word1 = (imm0 << 16) | (src2 << 8) | src1
    word2 = imm1 & 0xFFFFFFFF
    word3 = imm2 & 0xFFFFFFFF
    return struct.pack('<IIII', word0, word1, word2, word3)


# =========================================================================
# Tensor descriptor encoding (32 bytes)
# =========================================================================
def encode_tdesc(ddr_addr, sram_addr, size_bytes, shape, rank=2, dtype=0, flags=0):
    """Encode a 256-bit (32-byte) tensor descriptor."""
    s = [0, 0, 0, 0]
    for i, v in enumerate(shape[:4]):
        s[i] = v
    buf = struct.pack('<I', ddr_addr)            # [31:0]
    buf += struct.pack('<H', sram_addr & 0xFFFF) # [47:32]
    buf += struct.pack('<H', size_bytes & 0xFFFF) # [63:48]
    buf += struct.pack('<HHHH', s[0], s[1], s[2], s[3]) # shapes
    buf += struct.pack('<BBB', rank, dtype, flags) # rank, dtype, flags
    buf += b'\x00' * 13  # reserved, pad to 32 bytes
    assert len(buf) == 32
    return buf


# =========================================================================
# Quantization helpers
# =========================================================================
def quantize_int8(arr):
    """Symmetric int8 quantization: scale = max(|arr|) / 127"""
    amax = np.max(np.abs(arr))
    if amax < 1e-10:
        return np.zeros_like(arr, dtype=np.int8), 1.0
    scale = amax / 127.0
    q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return q, scale


def int8_gemm_golden(A_q, B_q, bias_q, scale, shift):
    """INT8 GEMM with INT32 accumulate, then requantize to INT8.
    C = requant(A_q @ B_q + bias_q)
    """
    # A_q: [M, K] int8, B_q: [K, N] int8
    M, K = A_q.shape
    _, N = B_q.shape

    # INT32 accumulate
    acc = np.zeros((M, N), dtype=np.int32)
    for m in range(M):
        for n in range(N):
            s = np.int32(0)
            for k in range(K):
                s += np.int32(A_q[m, k]) * np.int32(B_q[k, n])
            acc[m, n] = s

    # Add bias (broadcast)
    if bias_q is not None:
        for m in range(M):
            for n in range(N):
                acc[m, n] += np.int32(bias_q[n])

    # Requantize: (acc * scale_int) >> shift, round-to-nearest
    result = np.zeros((M, N), dtype=np.int8)
    for m in range(M):
        for n in range(N):
            product = int(acc[m, n]) * int(scale)
            if shift > 0:
                product += (1 << (shift - 1))  # rounding
            shifted = product >> shift
            result[m, n] = np.clip(shifted, -128, 127).astype(np.int8)

    return result


# =========================================================================
# im2col for Conv lowering
# =========================================================================
def im2col(input_data, kh, kw, stride=1, pad=0):
    """Convert [C, H, W] input to [out_h*out_w, C*kh*kw] matrix."""
    C, H, W = input_data.shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    if pad > 0:
        input_data = np.pad(input_data,
                           ((0, 0), (pad, pad), (pad, pad)),
                           mode='constant')

    col = np.zeros((out_h * out_w, C * kh * kw), dtype=input_data.dtype)
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            patch = input_data[:, i*stride:i*stride+kh, j*stride:j*stride+kw]
            col[idx] = patch.flatten()
            idx += 1
    return col


# =========================================================================
# Compiler
# =========================================================================
class GraphCompiler:
    def __init__(self, model_path):
        self.model = onnx.load(model_path)
        self.model = shape_inference.infer_shapes(self.model)
        self.graph = self.model.graph

        # Initializer lookup
        self.initializers = {}
        for init in self.graph.initializer:
            self.initializers[init.name] = numpy_helper.to_array(init)

        # Shape lookup from value_info + graph inputs/outputs
        self.shapes = {}
        for vi in list(self.graph.value_info) + list(self.graph.input) + list(self.graph.output):
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if all(d > 0 for d in dims):
                self.shapes[vi.name] = tuple(dims)

        # DDR allocation
        self.ddr_offset = DDR_GRAPH_DATA_BASE
        self.ddr_allocs = {}  # name -> (offset, size)
        self.ddr_image = bytearray()

        # SRAM allocation
        self.sram_offset = 0
        self.sram_allocs = {}  # name -> (offset, size)

        # Tensor descriptors
        self.tdesc_list = []  # list of (name, bytes)
        self.tdesc_id = {}    # name -> index

        # Instructions
        self.instructions = []

        # Track which tensors have valid data in SRAM (skip DMA_LOAD for these)
        self.sram_live = set()

        # Quantization scales
        self.quant_scales = {}

    def _alloc_ddr(self, name, data_bytes):
        """Allocate DDR space, 64-byte aligned."""
        offset = self.ddr_offset
        # Align to 64 bytes
        offset = (offset + 63) & ~63
        size = len(data_bytes)
        self.ddr_allocs[name] = (offset, size)
        # Extend ddr_image
        needed = (offset - DDR_GRAPH_DATA_BASE) + size
        while len(self.ddr_image) < needed:
            self.ddr_image.extend(b'\x00' * 1024)
        self.ddr_image[offset - DDR_GRAPH_DATA_BASE:
                       offset - DDR_GRAPH_DATA_BASE + size] = data_bytes
        self.ddr_offset = offset + size
        return offset

    def _alloc_sram(self, name, size):
        """Allocate SRAM0 space (bump allocator)."""
        if self.sram_offset + size > 65536:
            raise RuntimeError(f"SRAM0 overflow: need {self.sram_offset + size}, have 65536")
        offset = self.sram_offset
        self.sram_allocs[name] = (offset, size)
        self.sram_offset = offset + size
        # Align next alloc to 16 bytes
        self.sram_offset = (self.sram_offset + 15) & ~15
        return offset

    def _add_tdesc(self, name, ddr_addr, sram_addr, size_bytes, shape, rank=2):
        """Register a tensor descriptor."""
        idx = len(self.tdesc_list)
        self.tdesc_id[name] = idx
        self.tdesc_list.append((name, encode_tdesc(
            ddr_addr, sram_addr, size_bytes, shape, rank)))
        return idx

    def _emit(self, instr_bytes):
        """Emit an instruction."""
        self.instructions.append(instr_bytes)

    def compile(self):
        """Main compilation pipeline."""
        # 1. Topological sort (ONNX graph is already sorted)
        ops = list(self.graph.node)

        # 2. Quantize initializers
        quant_data = {}
        for name, arr in self.initializers.items():
            if arr.dtype in (np.float32, np.float64):
                q, scale = quantize_int8(arr.flatten().astype(np.float32))
                quant_data[name] = q.reshape(arr.shape)
                self.quant_scales[name] = scale
            elif arr.dtype == np.int64:
                # Shape tensors (for Reshape) - keep as-is
                quant_data[name] = arr
                self.quant_scales[name] = 1.0
            else:
                quant_data[name] = arr
                self.quant_scales[name] = 1.0

        # 3. Allocate DDR for weights/biases
        for name, arr in quant_data.items():
            if arr.dtype == np.int64:
                continue  # skip shape constants
            data = arr.astype(np.int8).tobytes()
            self._alloc_ddr(name, data)

        # 4. Allocate input in DDR
        input_name = self.graph.input[0].name
        input_shape = self.shapes.get(input_name, (1,))
        input_size = int(np.prod(input_shape))
        # Quantize input (use small random for testing)
        np.random.seed(99)
        input_fp = np.random.randn(*input_shape).astype(np.float32) * 0.5
        input_q, input_scale = quantize_int8(input_fp)
        self.quant_scales[input_name] = input_scale
        quant_data[input_name] = input_q

        input_ddr_offset = self._alloc_ddr(input_name, input_q.tobytes())

        # 5. Allocate SRAM and create tensor descriptors
        # Process ops to determine what needs to be in SRAM
        tensor_shapes = dict(self.shapes)

        # Allocate SRAM for input
        input_sram = self._alloc_sram(input_name, input_size)
        self._add_tdesc(input_name, input_ddr_offset, input_sram,
                       input_size, list(input_shape), len(input_shape))

        # 6. Lower ops
        current_data = {input_name: quant_data[input_name]}

        for op in ops:
            op_type = op.op_type

            if op_type == 'Gemm':
                self._lower_gemm(op, quant_data, current_data, tensor_shapes)
            elif op_type == 'Relu':
                self._lower_relu(op, quant_data, current_data, tensor_shapes)
            elif op_type == 'Conv':
                self._lower_conv(op, quant_data, current_data, tensor_shapes)
            elif op_type in ('Reshape', 'Flatten'):
                self._lower_reshape(op, quant_data, current_data, tensor_shapes)
            else:
                print(f"WARNING: Unsupported op '{op_type}', skipping")

        # 7. DMA_STORE output
        output_name = self.graph.output[0].name
        if output_name in self.tdesc_id:
            out_id = self.tdesc_id[output_name]
            # Update the output tensor descriptor DDR addr to IO base
            out_sram, out_size = self.sram_allocs[output_name]
            out_shape = tensor_shapes.get(output_name, (1,))
            # Re-create descriptor with IO base
            self.tdesc_list[out_id] = (output_name, encode_tdesc(
                DDR_GRAPH_IO_BASE, out_sram, out_size,
                list(out_shape), len(out_shape)))

            self._emit(encode_instr(OP_G_DMA_STORE, src0=out_id))

        # 8. END instruction
        self._emit(encode_instr(OP_G_END))

        # 9. Compute golden output
        golden = self._compute_golden(ops, quant_data, current_data, tensor_shapes)

        return golden

    def _lower_gemm(self, op, quant_data, current_data, tensor_shapes):
        """Lower Gemm op: DMA_LOAD weight, DMA_LOAD bias, GEMM, EW_ADD bias."""
        input_name = op.input[0]
        weight_name = op.input[1]
        bias_name = op.input[2] if len(op.input) > 2 else None
        output_name = op.output[0]

        # Get transB attribute
        transB = 0
        for attr in op.attribute:
            if attr.name == 'transB':
                transB = attr.i

        # Weight shape
        W = quant_data[weight_name]
        if transB:
            # W is [N, K], GEMM computes input @ W^T = [M, K] @ [K, N]
            N, K = W.shape
        else:
            K, N = W.shape

        # Input shape: [M, K]
        in_sram, in_size = self.sram_allocs[input_name]
        in_shape = tensor_shapes.get(input_name, (1, K))
        M = in_shape[0] if len(in_shape) > 1 else 1

        # Allocate SRAM for weight
        w_size = W.size
        w_sram = self._alloc_sram(weight_name, w_size)
        w_ddr = self.ddr_allocs[weight_name][0]

        # Weight needs to be stored as [K, N] for non-transposed GEMM
        # If transB=1, weight is [N, K] and we use TRANSPOSE_B flag
        if transB:
            # Store weight as [N, K] in row-major, use transpose flag
            w_shape = [N, K]
        else:
            w_shape = [K, N]

        w_id = self._add_tdesc(weight_name, w_ddr, w_sram, w_size, w_shape)

        # DMA_LOAD weight
        self._emit(encode_instr(OP_G_DMA_LOAD, src0=w_id))
        self.sram_live.add(weight_name)

        # DMA_LOAD input (only if not already in SRAM from a previous op)
        in_id = self.tdesc_id[input_name]
        if input_name not in self.sram_live:
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=in_id))
            self.sram_live.add(input_name)

        # Allocate output SRAM
        out_size = M * N
        out_sram = self._alloc_sram(output_name, out_size)
        out_shape = [M, N]
        tensor_shapes[output_name] = tuple(out_shape)
        out_ddr = DDR_GRAPH_IO_BASE  # placeholder
        out_id = self._add_tdesc(output_name, out_ddr, out_sram, out_size, out_shape)

        # Compute requant params
        shift = max(0, int(math.ceil(math.log2(max(K, 1)))))
        scale_int = 1
        # Encode: imm0 = scale[7:0] | (shift[7:0] << 8)
        imm0 = (scale_int & 0xFF) | ((shift & 0xFF) << 8)

        # GEMM flags
        flags = GFLAG_REQUANT
        if transB:
            flags |= GFLAG_TRANSPOSE_B

        # GEMM instruction: src0=input, src1=weight, dst=output
        self._emit(encode_instr(OP_G_GEMM, flags=flags,
                                src0=in_id, src1=w_id, dst=out_id,
                                imm0=imm0))

        # Handle bias
        if bias_name and bias_name in quant_data:
            bias_q = quant_data[bias_name]
            b_size = bias_q.size
            b_sram = self._alloc_sram(bias_name, b_size)
            b_ddr = self.ddr_allocs[bias_name][0]
            # Bias descriptor: shape [1, N] for broadcast add
            b_id = self._add_tdesc(bias_name, b_ddr, b_sram, b_size, [1, N])

            # DMA_LOAD bias
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=b_id))

            # We need a "broadcasted bias" tensor in SRAM matching output shape
            # For simplicity with EW_ADD, we'll create a temporary with bias
            # replicated for each row
            bcast_name = bias_name + '_bcast'
            bcast_size = M * N
            bcast_sram = self._alloc_sram(bcast_name, bcast_size)
            bcast_id = self._add_tdesc(bcast_name, 0, bcast_sram, bcast_size, [M, N])

            # The bias needs to be broadcast - we handle this by writing
            # the bias data M times in the compiler (pre-materialized in DDR)
            bias_bcast = np.tile(bias_q.flatten(), M)
            bcast_ddr = self._alloc_ddr(bcast_name, bias_bcast.astype(np.int8).tobytes())
            # Update the descriptor with correct DDR addr
            self.tdesc_list[bcast_id] = (bcast_name, encode_tdesc(
                bcast_ddr, bcast_sram, bcast_size, [M, N]))

            # DMA_LOAD broadcast bias
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=bcast_id))

            # EW_ADD: output = output + bias_broadcast
            # src0=GEMM output, src1=bias_broadcast, dst=output (in-place)
            self._emit(encode_instr(OP_G_EW_ADD, src0=out_id, src1=bcast_id, dst=out_id))

        # Mark output as live in SRAM
        self.sram_live.add(output_name)
        current_data[output_name] = None  # computed at runtime

    def _lower_relu(self, op, quant_data, current_data, tensor_shapes):
        """Lower Relu op."""
        input_name = op.input[0]
        output_name = op.output[0]

        in_id = self.tdesc_id[input_name]
        in_sram, in_size = self.sram_allocs[input_name]
        in_shape = tensor_shapes.get(input_name, (1,))

        # For in-place RELU, output uses same SRAM as input
        out_sram = in_sram
        out_size = in_size
        tensor_shapes[output_name] = in_shape
        self.sram_allocs[output_name] = (out_sram, out_size)

        out_id = self._add_tdesc(output_name, 0, out_sram, out_size,
                                list(in_shape), len(in_shape))

        # RELU: src0=input, dst=output (can be same SRAM location)
        self._emit(encode_instr(OP_G_RELU, src0=in_id, dst=out_id))

        self.sram_live.add(output_name)
        current_data[output_name] = None

    def _lower_conv(self, op, quant_data, current_data, tensor_shapes):
        """Lower Conv via im2col: pre-materialize im2col in DDR, then GEMM."""
        input_name = op.input[0]
        weight_name = op.input[1]
        bias_name = op.input[2] if len(op.input) > 2 else None
        output_name = op.output[0]

        # Get attributes
        kernel_shape = [3, 3]
        pads = [0, 0, 0, 0]
        strides = [1, 1]
        for attr in op.attribute:
            if attr.name == 'kernel_shape':
                kernel_shape = list(attr.ints)
            elif attr.name == 'pads':
                pads = list(attr.ints)
            elif attr.name == 'strides':
                strides = list(attr.ints)

        kh, kw = kernel_shape
        pad = pads[0]  # assume symmetric
        stride = strides[0]

        # Input shape: [N, C, H, W]
        in_shape = tensor_shapes.get(input_name)
        if in_shape is None or len(in_shape) != 4:
            raise RuntimeError(f"Conv input {input_name} shape unknown or not 4D")
        _, C_in, H, W_dim = in_shape

        # Weight shape: [C_out, C_in, kh, kw]
        W_fp = self.initializers[weight_name]
        C_out = W_fp.shape[0]

        out_h = (H + 2 * pad - kh) // stride + 1
        out_w = (W_dim + 2 * pad - kw) // stride + 1

        # im2col: [out_h*out_w, C_in*kh*kw]
        # Use quantized input for im2col
        input_q = quant_data.get(input_name)
        if input_q is None:
            # Input was computed by previous op - use zeros as placeholder
            # (golden will be computed separately)
            input_q = np.zeros(in_shape, dtype=np.int8)

        im2col_data = im2col(input_q[0], kh, kw, stride, pad)  # [out_h*out_w, C_in*kh*kw]
        im2col_name = input_name + '_im2col'
        im2col_M = out_h * out_w
        im2col_K = C_in * kh * kw

        # Allocate im2col in DDR and SRAM
        im2col_ddr = self._alloc_ddr(im2col_name, im2col_data.astype(np.int8).tobytes())
        im2col_sram = self._alloc_sram(im2col_name, im2col_M * im2col_K)
        im2col_id = self._add_tdesc(im2col_name, im2col_ddr, im2col_sram,
                                    im2col_M * im2col_K, [im2col_M, im2col_K])

        # Reshape weight: [C_out, C_in*kh*kw] -> this is already transposed form
        W_q = quant_data[weight_name].reshape(C_out, -1)  # [C_out, im2col_K]
        w_size = W_q.size
        w_sram = self._alloc_sram(weight_name + '_reshaped', w_size)
        w_ddr = self._alloc_ddr(weight_name + '_reshaped', W_q.astype(np.int8).tobytes())
        w_id = self._add_tdesc(weight_name + '_reshaped', w_ddr, w_sram,
                               w_size, [C_out, im2col_K])

        # DMA_LOAD im2col data and weights
        self._emit(encode_instr(OP_G_DMA_LOAD, src0=im2col_id))
        self._emit(encode_instr(OP_G_DMA_LOAD, src0=w_id))

        # Output: [im2col_M, C_out] = [out_h*out_w, C_out]
        out_size = im2col_M * C_out
        out_sram = self._alloc_sram(output_name, out_size)
        tensor_shapes[output_name] = (1, C_out, out_h, out_w)
        out_id = self._add_tdesc(output_name, DDR_GRAPH_IO_BASE, out_sram,
                                out_size, [im2col_M, C_out])

        # GEMM: im2col @ W^T = [M, K] @ [K, N] where W stored as [N, K]
        shift = max(0, int(math.ceil(math.log2(max(im2col_K, 1)))))
        scale_int = 1
        imm0 = (scale_int & 0xFF) | ((shift & 0xFF) << 8)
        flags = GFLAG_REQUANT | GFLAG_TRANSPOSE_B

        self._emit(encode_instr(OP_G_GEMM, flags=flags,
                                src0=im2col_id, src1=w_id, dst=out_id,
                                imm0=imm0))

        # Handle bias
        if bias_name and bias_name in quant_data:
            bias_q = quant_data[bias_name]
            bcast_name = bias_name + '_conv_bcast'
            bias_bcast = np.tile(bias_q.flatten(), im2col_M)
            bcast_size = bias_bcast.size
            bcast_ddr = self._alloc_ddr(bcast_name, bias_bcast.astype(np.int8).tobytes())
            bcast_sram = self._alloc_sram(bcast_name, bcast_size)
            bcast_id = self._add_tdesc(bcast_name, bcast_ddr, bcast_sram,
                                       bcast_size, [im2col_M, C_out])

            self._emit(encode_instr(OP_G_DMA_LOAD, src0=bcast_id))
            self._emit(encode_instr(OP_G_EW_ADD, src0=out_id, src1=bcast_id, dst=out_id))

        self.sram_live.add(output_name)
        current_data[output_name] = None
        quant_data[im2col_name] = im2col_data

    def _lower_reshape(self, op, quant_data, current_data, tensor_shapes):
        """Lower Reshape/Flatten: just alias the SRAM location."""
        input_name = op.input[0]
        output_name = op.output[0]

        if input_name not in self.sram_allocs:
            print(f"WARNING: Reshape input {input_name} not in SRAM")
            return

        in_sram, in_size = self.sram_allocs[input_name]
        out_shape = self.shapes.get(output_name)
        if out_shape is None:
            # Compute from reshape target
            if len(op.input) > 1 and op.input[1] in self.initializers:
                out_shape = tuple(self.initializers[op.input[1]].astype(int).tolist())
            else:
                out_shape = (1, in_size)

        tensor_shapes[output_name] = out_shape
        self.sram_allocs[output_name] = (in_sram, in_size)

        # Create descriptor with new shape (same SRAM location)
        in_id = self.tdesc_id.get(input_name)
        out_id = self._add_tdesc(output_name, 0, in_sram, in_size,
                                list(out_shape), len(out_shape))
        self.tdesc_id[output_name] = out_id

        # Input is already live in SRAM; the reshape is just an alias
        if input_name in self.sram_live:
            self.sram_live.add(output_name)
        current_data[output_name] = None

    def _compute_golden(self, ops, quant_data, current_data, tensor_shapes):
        """Compute golden output using pure int8 arithmetic."""
        input_name = self.graph.input[0].name
        tensors = {input_name: quant_data[input_name].flatten().astype(np.int8)}

        for op in ops:
            if op.op_type == 'Gemm':
                inp = tensors[op.input[0]]
                W = quant_data[op.input[1]]
                bias = quant_data[op.input[2]] if len(op.input) > 2 else None

                transB = 0
                for attr in op.attribute:
                    if attr.name == 'transB':
                        transB = attr.i

                in_shape = tensor_shapes.get(op.input[0], (1, inp.size))
                M = in_shape[0] if len(in_shape) > 1 else 1
                if transB:
                    N, K = W.shape
                else:
                    K, N = W.shape

                A = inp.reshape(M, K).astype(np.int8)
                if transB:
                    B = W.reshape(N, K).T.astype(np.int8)  # [K, N]
                else:
                    B = W.reshape(K, N).astype(np.int8)

                shift = max(0, int(math.ceil(math.log2(max(K, 1)))))
                scale_int = 1

                # GEMM without bias (RTL adds bias via separate EW_ADD after requant)
                result = int8_gemm_golden(A, B, None, scale_int, shift)

                # Add bias after requant (matches RTL: saturating INT8 add)
                if bias is not None:
                    bias_flat = bias.flatten().astype(np.int8)
                    for m in range(M):
                        for n in range(N):
                            val = int(result[m, n]) + int(bias_flat[n])
                            result[m, n] = np.clip(val, -128, 127).astype(np.int8)

                tensors[op.output[0]] = result.flatten()
                tensor_shapes[op.output[0]] = (M, N)

            elif op.op_type == 'Relu':
                inp = tensors[op.input[0]]
                tensors[op.output[0]] = np.maximum(inp, 0).astype(np.int8)
                tensor_shapes[op.output[0]] = tensor_shapes.get(op.input[0])

            elif op.op_type == 'Conv':
                inp = tensors[op.input[0]]
                W_fp = self.initializers[op.input[1]]
                W_q = quant_data[op.input[1]]
                bias = quant_data[op.input[2]] if len(op.input) > 2 else None

                kernel_shape = [3, 3]
                pads = [0, 0, 0, 0]
                strides = [1, 1]
                for attr in op.attribute:
                    if attr.name == 'kernel_shape': kernel_shape = list(attr.ints)
                    elif attr.name == 'pads': pads = list(attr.ints)
                    elif attr.name == 'strides': strides = list(attr.ints)

                kh, kw = kernel_shape
                pad = pads[0]
                stride = strides[0]

                in_shape = tensor_shapes.get(op.input[0])
                C_in, H, W_dim = in_shape[1], in_shape[2], in_shape[3]
                C_out = W_fp.shape[0]
                out_h = (H + 2 * pad - kh) // stride + 1
                out_w = (W_dim + 2 * pad - kw) // stride + 1

                # im2col
                input_4d = inp.reshape(1, C_in, H, W_dim).astype(np.int8)
                col = im2col(input_4d[0], kh, kw, stride, pad)  # [M, K]

                # Weight as [C_out, K] -> transpose to [K, C_out]
                W_2d = W_q.reshape(C_out, -1)  # [C_out, K]
                # GEMM: col @ W_2d^T = [M, K] @ [K, C_out]
                B_mat = W_2d.T.astype(np.int8)  # [K, C_out]

                K_dim = col.shape[1]
                shift = max(0, int(math.ceil(math.log2(max(K_dim, 1)))))
                M_dim = col.shape[0]
                # GEMM without bias (RTL adds bias via separate EW_ADD after requant)
                result = int8_gemm_golden(col.astype(np.int8), B_mat, None, 1, shift)

                # Add bias after requant (matches RTL)
                if bias is not None:
                    bias_flat = bias.flatten().astype(np.int8)
                    for m in range(M_dim):
                        for n in range(C_out):
                            val = int(result[m, n]) + int(bias_flat[n])
                            result[m, n] = np.clip(val, -128, 127).astype(np.int8)

                tensors[op.output[0]] = result.flatten()
                tensor_shapes[op.output[0]] = (1, C_out, out_h, out_w)

            elif op.op_type in ('Reshape', 'Flatten'):
                tensors[op.output[0]] = tensors[op.input[0]]
                out_shape = self.shapes.get(op.output[0])
                if out_shape is None:
                    if len(op.input) > 1 and op.input[1] in self.initializers:
                        out_shape = tuple(self.initializers[op.input[1]].astype(int).tolist())
                    else:
                        out_shape = (1, tensors[op.input[0]].size)
                tensor_shapes[op.output[0]] = out_shape

        output_name = self.graph.output[0].name
        return tensors.get(output_name, np.array([], dtype=np.int8))

    def write_outputs(self, outdir, golden):
        """Write all artifacts to output directory."""
        os.makedirs(outdir, exist_ok=True)

        # program.bin
        prog_path = os.path.join(outdir, 'program.bin')
        with open(prog_path, 'wb') as f:
            for instr in self.instructions:
                f.write(instr)
        print(f"  Written {len(self.instructions)} instructions to {prog_path}")

        # tdesc.bin
        tdesc_path = os.path.join(outdir, 'tdesc.bin')
        with open(tdesc_path, 'wb') as f:
            for name, data in self.tdesc_list:
                f.write(data)
        print(f"  Written {len(self.tdesc_list)} descriptors to {tdesc_path}")

        # ddr_image.bin
        # Build full DDR image: data region starts at DDR_GRAPH_DATA_BASE
        ddr_path = os.path.join(outdir, 'ddr_image.bin')
        full_ddr = bytearray(DDR_GRAPH_DATA_BASE) + self.ddr_image
        with open(ddr_path, 'wb') as f:
            f.write(full_ddr)
        print(f"  Written DDR image ({len(full_ddr)} bytes) to {ddr_path}")

        # golden.bin
        golden_path = os.path.join(outdir, 'golden.bin')
        golden_bytes = golden.astype(np.int8).tobytes()
        with open(golden_path, 'wb') as f:
            f.write(golden_bytes)
        print(f"  Written golden ({len(golden_bytes)} bytes) to {golden_path}")

        # golden.npy
        np.save(os.path.join(outdir, 'golden.npy'), golden.astype(np.int8))

        # manifest.json
        manifest = {
            'num_instructions': len(self.instructions),
            'num_descriptors': len(self.tdesc_list),
            'ddr_image_size': len(full_ddr),
            'golden_size': len(golden_bytes),
            'sram_used': self.sram_offset,
            'descriptors': [(name, idx) for idx, (name, _) in enumerate(self.tdesc_list)],
            'sram_allocs': {k: {'offset': v[0], 'size': v[1]}
                           for k, v in self.sram_allocs.items()},
        }
        manifest_path = os.path.join(outdir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"  Written manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description='ONNX → NPU Graph ISA Compiler')
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--outdir', default='build/graph', help='Output directory')
    args = parser.parse_args()

    print(f"=== ONNX Compiler ===")
    print(f"Model: {args.model}")
    print(f"Output: {args.outdir}")

    compiler = GraphCompiler(args.model)
    golden = compiler.compile()

    print(f"\nCompilation results:")
    print(f"  Instructions: {len(compiler.instructions)}")
    print(f"  Descriptors:  {len(compiler.tdesc_list)}")
    print(f"  SRAM used:    {compiler.sram_offset} / 65536 bytes")
    print(f"  Golden shape: {golden.shape}")

    compiler.write_outputs(args.outdir, golden)
    print(f"\nDone!")


if __name__ == '__main__':
    main()
