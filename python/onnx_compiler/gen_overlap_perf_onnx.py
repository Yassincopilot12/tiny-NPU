#!/usr/bin/env python3
"""
Generate ONNX model for overlap performance testing.
3-layer MLP with 64x64 GEMMs to stress DMA+GEMM overlap.
  [1, 64] -> Gemm(64,64) -> Relu -> Gemm(64,64) -> Relu -> Gemm(64,32) -> [1, 32]
"""
import os
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("ERROR: onnx package not installed")
    exit(1)


def main():
    np.random.seed(123)

    # Layer 1: 64 -> 64
    W1 = np.random.randn(64, 64).astype(np.float32) * 0.1
    b1 = np.random.randn(64).astype(np.float32) * 0.01

    # Layer 2: 64 -> 64
    W2 = np.random.randn(64, 64).astype(np.float32) * 0.1
    b2 = np.random.randn(64).astype(np.float32) * 0.01

    # Layer 3: 64 -> 32
    W3 = np.random.randn(32, 64).astype(np.float32) * 0.1
    b3 = np.random.randn(32).astype(np.float32) * 0.01

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 64])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 32])

    nodes = [
        helper.make_node('Gemm', ['input', 'W1', 'b1'], ['g1'], transB=1),
        helper.make_node('Relu', ['g1'], ['r1']),
        helper.make_node('Gemm', ['r1', 'W2', 'b2'], ['g2'], transB=1),
        helper.make_node('Relu', ['g2'], ['r2']),
        helper.make_node('Gemm', ['r2', 'W3', 'b3'], ['output'], transB=1),
    ]

    initializers = [
        numpy_helper.from_array(W1, 'W1'),
        numpy_helper.from_array(b1, 'b1'),
        numpy_helper.from_array(W2, 'W2'),
        numpy_helper.from_array(b2, 'b2'),
        numpy_helper.from_array(W3, 'W3'),
        numpy_helper.from_array(b3, 'b3'),
    ]

    graph = helper.make_graph(nodes, 'overlap_perf_test', [X], [Y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'overlap_perf_test.onnx')
    onnx.save(model, out_path)
    print(f"Saved overlap perf model to {out_path}")
    print(f"  [1,64] -> Gemm(64,64) -> Relu -> Gemm(64,64) -> Relu -> Gemm(64,32) -> [1,32]")


if __name__ == '__main__':
    main()
