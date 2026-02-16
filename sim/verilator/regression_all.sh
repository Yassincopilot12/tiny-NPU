#!/bin/bash
# regression_all.sh - Run all NPU simulation targets
# Phase 1-4 complete regression

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: Build directory not found. Run cmake first:"
    echo "  mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR} && cmake .. && cmake --build . -j\$(nproc)"
    exit 1
fi

cd "$BUILD_DIR"

PASS=0
FAIL=0
SKIP=0
FAILED_TESTS=""

run_test() {
    local name="$1"
    shift
    local args="$@"

    if [ ! -f "./${name}" ]; then
        echo "  [SKIP] ${name} (not built)"
        SKIP=$((SKIP + 1))
        return
    fi

    echo -n "  [RUN]  ${name} ${args}... "
    if ./${name} ${args} > "${name}.log" 2>&1; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL (see ${name}.log)"
        FAIL=$((FAIL + 1))
        FAILED_TESTS="${FAILED_TESTS} ${name}"
    fi
}

echo "=========================================="
echo " NPU Regression Suite (Phase 1-4)"
echo "=========================================="
echo ""

echo "--- Phase 1: Legacy Tests ---"
run_test npu_sim
run_test engine_sim

echo ""
echo "--- Phase 2: Integration Tests ---"
run_test gpt2_block_sim --datadir demo_data
run_test kv_cache_sim --datadir demo_data
run_test demo_infer --datadir demo_data

echo ""
echo "--- Phase 2b: LLaMA Tests ---"
run_test llama_block_sim --datadir llama_data
run_test llama_demo_infer --datadir llama_data

echo ""
echo "--- Phase 3: ONNX Graph Mode Tests ---"
run_test onnx_smoke_sim --datadir graph
run_test onnx_cnn_smoke_sim --datadir graph_cnn
run_test onnx_reduce_sim --datadir graph_reduce
run_test onnx_math_sim --datadir graph_math
run_test onnx_gather_sim --datadir graph_gather
run_test onnx_slice_concat_sim --datadir graph_slice_concat
run_test onnx_batchnorm_pool_sim --datadir graph_batchnorm_pool
run_test onnx_fuzz_sim --datadir graph_fuzz
run_test onnx_stress_sim --datadir graph_stress

echo ""
echo "--- Phase 4: Performance + FP16 Tests ---"
run_test onnx_overlap_perf_sim --datadir graph_overlap_perf
run_test onnx_fp16_smoke_sim --datadir graph_fp16_smoke
run_test mixed_precision_cnn_sim --datadir graph_mixed_cnn
run_test onnx_resize_pad_sim --datadir graph_resize_pad

echo ""
echo "=========================================="
echo " Results: ${PASS} PASS, ${FAIL} FAIL, ${SKIP} SKIP"
echo "=========================================="

if [ $FAIL -gt 0 ]; then
    echo " Failed tests:${FAILED_TESTS}"
    exit 1
fi

if [ $PASS -eq 0 ]; then
    echo " WARNING: No tests ran successfully"
    exit 1
fi

echo " All tests passed!"
exit 0
