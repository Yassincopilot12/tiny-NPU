cmake_minimum_required(VERSION 3.12)
project(npu_sim)

# Find Verilator
find_package(verilator HINTS $ENV{VERILATOR_ROOT})
if (NOT verilator_FOUND)
    message(FATAL_ERROR "Verilator not found. Set VERILATOR_ROOT or add to PATH.")
endif()

# SystemVerilog source files
set(RTL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../rtl)
set(SIM_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Shared RTL sources (packages, memory, compute engines)
set(SV_COMMON
    ${RTL_DIR}/pkg/npu_pkg.sv
    ${RTL_DIR}/pkg/isa_pkg.sv
    ${RTL_DIR}/pkg/fixed_pkg.sv
    ${RTL_DIR}/graph/fp16_utils.sv
    ${RTL_DIR}/mem/sram_dp.sv
    ${RTL_DIR}/mem/banked_sram.sv
    ${RTL_DIR}/mem/kv_cache_bank.sv
    ${RTL_DIR}/gemm/mac_int8.sv
    ${RTL_DIR}/gemm/mac_fp16.sv
    ${RTL_DIR}/gemm/pe.sv
    ${RTL_DIR}/gemm/systolic_array.sv
    ${RTL_DIR}/ops/vec_engine.sv
    ${RTL_DIR}/ops/reduce_max.sv
    ${RTL_DIR}/ops/reduce_sum.sv
    ${RTL_DIR}/ops/exp_lut.sv
    ${RTL_DIR}/ops/recip_lut.sv
    ${RTL_DIR}/graph/graph_exp_lut_fp16.sv
    ${RTL_DIR}/ops/softmax_engine.sv
    ${RTL_DIR}/ops/mean_var_engine.sv
    ${RTL_DIR}/ops/rsqrt_lut.sv
    ${RTL_DIR}/graph/graph_rsqrt_lut_fp16.sv
    ${RTL_DIR}/ops/layernorm_engine.sv
    ${RTL_DIR}/ops/gelu_lut.sv
    ${RTL_DIR}/ops/gelu_engine.sv
    ${RTL_DIR}/ops/silu_lut.sv
    ${RTL_DIR}/ops/rmsnorm_engine.sv
    ${RTL_DIR}/ops/rope_engine.sv
)

# Top-level sim sources (includes bus, ctrl, full top)
set(SV_TOP_EXTRA
    ${RTL_DIR}/bus/axi_types.sv
    ${RTL_DIR}/bus/axi_lite_regs.sv
    ${RTL_DIR}/bus/axi_dma_rd.sv
    ${RTL_DIR}/bus/axi_dma_wr.sv
    ${RTL_DIR}/ctrl/addr_gen.sv
    ${RTL_DIR}/ctrl/scoreboard.sv
    ${RTL_DIR}/ctrl/barrier.sv
    ${RTL_DIR}/ctrl/ucode_fetch.sv
    ${RTL_DIR}/ctrl/ucode_decode.sv
    ${RTL_DIR}/gemm/gemm_ctrl.sv
    ${RTL_DIR}/gemm/gemm_post.sv
    ${RTL_DIR}/top.sv
)

set(VERILATOR_COMMON_ARGS
    --cc
    --trace
    --trace-structs
    -Wall
    -Wno-UNUSED
    -Wno-UNDRIVEN
    -Wno-PINCONNECTEMPTY
    -Wno-DECLFILENAME
    -Wno-IMPORTSTAR
    -Wno-VARHIDDEN
    -Wno-WIDTH
    -Wno-BLKSEQ
    -Wno-UNSIGNED
    -Wno-PINMISSING
    -Wno-UNOPTFLAT
    --x-assign unique
    --x-initial unique
    -DSIMULATION
    -I${RTL_DIR}/pkg
    -I${RTL_DIR}/bus
    -I${RTL_DIR}/mem
    -I${RTL_DIR}/ctrl
    -I${RTL_DIR}/gemm
    -I${RTL_DIR}/ops
    -I${RTL_DIR}/graph
)

# Graph mode RTL sources
set(SV_GRAPH
    ${RTL_DIR}/graph/graph_isa_pkg.sv
    ${RTL_DIR}/graph/tensor_table.sv
    ${RTL_DIR}/graph/graph_fetch.sv
    ${RTL_DIR}/graph/graph_decode.sv
    ${RTL_DIR}/graph/graph_dispatch.sv
    ${RTL_DIR}/graph/graph_top.sv
)

# Phase 3 graph engines
set(SV_GRAPH_P3
    ${RTL_DIR}/graph/reduce_engine.sv
    ${RTL_DIR}/graph/graph_exp_lut.sv
    ${RTL_DIR}/graph/graph_log_lut.sv
    ${RTL_DIR}/graph/graph_sqrt_lut.sv
    ${RTL_DIR}/graph/graph_rsqrt_lut.sv
    ${RTL_DIR}/graph/math_engine.sv
    ${RTL_DIR}/graph/gather_engine.sv
    ${RTL_DIR}/graph/slice_engine.sv
    ${RTL_DIR}/graph/concat_engine.sv
    ${RTL_DIR}/graph/avgpool2d_engine.sv
)

# Phase 4 graph engines + FP16 support
set(SV_GRAPH_P4
    ${RTL_DIR}/graph/maxpool2d_engine.sv
    ${RTL_DIR}/graph/pad_engine.sv
    ${RTL_DIR}/graph/resize_nearest_engine.sv
    ${RTL_DIR}/graph/cast_engine.sv
    ${RTL_DIR}/graph/graph_log_lut_fp16.sv
    ${RTL_DIR}/graph/graph_sqrt_lut_fp16.sv
    ${RTL_DIR}/mem/tile_buffer.sv
)

# Control plane RTL needed for integration/GPT-2/LLaMA tests
set(SV_CTRL
    ${RTL_DIR}/ctrl/scoreboard.sv
    ${RTL_DIR}/ctrl/barrier.sv
    ${RTL_DIR}/ctrl/ucode_fetch.sv
    ${RTL_DIR}/ctrl/ucode_decode.sv
    ${RTL_DIR}/ctrl/kv_ctrl.sv
)

# ONNX sim RTL (graph + all engines + gemm_ctrl + onnx_sim_top)
set(SV_ONNX_SIM
    ${SV_COMMON} ${SV_GRAPH} ${SV_GRAPH_P3} ${SV_GRAPH_P4}
    ${RTL_DIR}/gemm/gemm_ctrl.sv
    ${SIM_DIR}/onnx_sim_top.sv
)

# Stub source for library targets (verilate() adds the real sources)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/verilator_stub.cpp "")

# =============================================================================
# UNIQUE TARGETS (each has a different top module / params — verilated once each)
# =============================================================================

# ---- Target 1: Control Plane Smoke Test (original top-level) ----
add_executable(npu_sim tb_top.cpp)

verilate(npu_sim
    SOURCES ${SV_COMMON} ${SV_TOP_EXTRA}
    TOP_MODULE top
    PREFIX Vtop
    VERILATOR_ARGS ${VERILATOR_COMMON_ARGS}
)

# ---- Target 2: Engine Compute Tests ----
add_executable(engine_sim tb_engines.cpp)

verilate(engine_sim
    SOURCES ${SV_COMMON} ${SIM_DIR}/engine_tb_top.sv
    TOP_MODULE engine_tb_top
    PREFIX Vengine_tb_top
    VERILATOR_ARGS ${VERILATOR_COMMON_ARGS}
)

# ---- Target 3: Integration Test (single attention head) ----
add_executable(integration_sim tb_integration.cpp)

verilate(integration_sim
    SOURCES ${SV_COMMON} ${SV_CTRL} ${SIM_DIR}/integration_top.sv
    TOP_MODULE integration_top
    PREFIX Vintegration_top
    VERILATOR_ARGS ${VERILATOR_COMMON_ARGS}
)

# ---- Target 4: GPT-2 Transformer Block Test (SRAM0=64K only) ----
add_executable(gpt2_block_sim tb_gpt2_block.cpp)

verilate(gpt2_block_sim
    SOURCES ${SV_COMMON} ${SV_CTRL} ${SIM_DIR}/gpt2_block_top.sv
    TOP_MODULE gpt2_block_top
    PREFIX Vgpt2_block_top
    VERILATOR_ARGS ${VERILATOR_COMMON_ARGS}
        -GSRAM0_DEPTH=65536
)

# =============================================================================
# SHARED LIBRARY: GPT-2 Demo (SRAM0=64K, SRAM1=8K) — targets 5, 6
# Verilated ONCE, linked by demo_infer + kv_cache_sim
# =============================================================================
add_library(gpt2_demo_verilated STATIC ${CMAKE_CURRENT_BINARY_DIR}/verilator_stub.cpp)

verilate(gpt2_demo_verilated
    SOURCES ${SV_COMMON} ${SV_CTRL} ${SIM_DIR}/gpt2_block_top.sv
    TOP_MODULE gpt2_block_top
    PREFIX Vgpt2_block_top
    VERILATOR_ARGS ${VERILATOR_COMMON_ARGS}
        -GSRAM0_DEPTH=65536
        -GSRAM1_DEPTH=8192
)

# ---- Target 5: GPT-2 Inference Demo ----
add_executable(demo_infer tb_demo_infer.cpp)
target_link_libraries(demo_infer PRIVATE gpt2_demo_verilated)

# ---- Target 6: KV Cache Correctness Test ----
add_executable(kv_cache_sim tb_kv_cache_sim.cpp)
target_link_libraries(kv_cache_sim PRIVATE gpt2_demo_verilated)

# =============================================================================
# SHARED LIBRARY: LLaMA (SRAM0=64K, SRAM1=8K) — targets 7, 8
# Verilated ONCE, linked by llama_block_sim + llama_demo_infer
# =============================================================================
add_library(llama_verilated STATIC ${CMAKE_CURRENT_BINARY_DIR}/verilator_stub.cpp)

verilate(llama_verilated
    SOURCES ${SV_COMMON} ${SV_CTRL} ${SIM_DIR}/llama_block_top.sv
    TOP_MODULE llama_block_top
    PREFIX Vllama_block_top
    VERILATOR_ARGS ${VERILATOR_COMMON_ARGS}
        -GSRAM0_DEPTH=65536
        -GSRAM1_DEPTH=8192
)

# ---- Target 7: LLaMA Transformer Block Test ----
add_executable(llama_block_sim tb_llama_block.cpp)
target_link_libraries(llama_block_sim PRIVATE llama_verilated)

# ---- Target 8: LLaMA Inference Demo ----
add_executable(llama_demo_infer tb_llama_demo_infer.cpp)
target_link_libraries(llama_demo_infer PRIVATE llama_verilated)

# =============================================================================
# SHARED LIBRARY: ONNX Graph Sim (SRAM0=64K) — targets 9-21
# Verilated ONCE, linked by all 13 ONNX test executables
# This is the biggest win: 13 redundant Verilator runs → 1
# =============================================================================
add_library(onnx_sim_verilated STATIC ${CMAKE_CURRENT_BINARY_DIR}/verilator_stub.cpp)

verilate(onnx_sim_verilated
    SOURCES ${SV_ONNX_SIM}
    TOP_MODULE onnx_sim_top
    PREFIX Vonnx_sim_top
    VERILATOR_ARGS ${VERILATOR_COMMON_ARGS}
        -GSRAM0_DEPTH=65536
)

# ---- Target 9: ONNX Graph Mode MLP Smoke Test ----
add_executable(onnx_smoke_sim tb_onnx_smoke.cpp)
target_link_libraries(onnx_smoke_sim PRIVATE onnx_sim_verilated)

# ---- Target 10: ONNX Graph Mode CNN Smoke Test ----
add_executable(onnx_cnn_smoke_sim tb_onnx_cnn_smoke.cpp)
target_link_libraries(onnx_cnn_smoke_sim PRIVATE onnx_sim_verilated)

# ---- Target 11: ONNX Reduce Engine Test ----
add_executable(onnx_reduce_sim tb_onnx_reduce.cpp)
target_link_libraries(onnx_reduce_sim PRIVATE onnx_sim_verilated)

# ---- Target 12: ONNX Math Engine Test ----
add_executable(onnx_math_sim tb_onnx_math.cpp)
target_link_libraries(onnx_math_sim PRIVATE onnx_sim_verilated)

# ---- Target 13: ONNX Gather Engine Test ----
add_executable(onnx_gather_sim tb_onnx_gather.cpp)
target_link_libraries(onnx_gather_sim PRIVATE onnx_sim_verilated)

# ---- Target 14: ONNX Slice+Concat Engine Test ----
add_executable(onnx_slice_concat_sim tb_onnx_slice_concat.cpp)
target_link_libraries(onnx_slice_concat_sim PRIVATE onnx_sim_verilated)

# ---- Target 15: ONNX BatchNorm+Pool Test ----
add_executable(onnx_batchnorm_pool_sim tb_onnx_batchnorm_pool.cpp)
target_link_libraries(onnx_batchnorm_pool_sim PRIVATE onnx_sim_verilated)

# ---- Target 16: ONNX Fuzz Test ----
add_executable(onnx_fuzz_sim tb_onnx_fuzz.cpp)
target_link_libraries(onnx_fuzz_sim PRIVATE onnx_sim_verilated)

# ---- Target 17: ONNX Stress Test ----
add_executable(onnx_stress_sim tb_onnx_stress.cpp)
target_link_libraries(onnx_stress_sim PRIVATE onnx_sim_verilated)

# ---- Target 18: ONNX Overlap Performance Test ----
# Tests SERIAL vs OVERLAP scheduling; verifies DMA+GEMM overlap with perf counters
add_executable(onnx_overlap_perf_sim tb_onnx_overlap_perf.cpp)
target_link_libraries(onnx_overlap_perf_sim PRIVATE onnx_sim_verilated)

# ---- Target 19: ONNX FP16 Smoke Test ----
# FP16 GEMM -> LayerNorm -> Softmax -> Math ops
add_executable(onnx_fp16_smoke_sim tb_onnx_fp16_smoke.cpp)
target_link_libraries(onnx_fp16_smoke_sim PRIVATE onnx_sim_verilated)

# ---- Target 20: Mixed Precision CNN Test ----
# Conv(INT8) -> BN(FP16) -> ReLU -> Pool with CAST insertion
add_executable(mixed_precision_cnn_sim tb_mixed_precision_cnn.cpp)
target_link_libraries(mixed_precision_cnn_sim PRIVATE onnx_sim_verilated)

# ---- Target 21: ONNX Resize+Pad Test ----
# Pad(constant=0) -> Resize(nearest, 2x) exact match
add_executable(onnx_resize_pad_sim tb_onnx_resize_pad.cpp)
target_link_libraries(onnx_resize_pad_sim PRIVATE onnx_sim_verilated)
