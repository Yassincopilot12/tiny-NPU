#!/usr/bin/env python3
"""
Generate ONNX model for FP16 smoke testing.
Simple 2-layer MLP for basic FP16 GEMM verification.
  [1, 16] -> Gemm(16,8) -> Relu -> Gemm(8,4) -> [1, 4]
Note: Actual FP16 path testing requires dtype propagation in compiler.
For now this generates an INT8 model that exercises the Phase 4 infrastructure.
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
    np.random.seed(77)

    W1 = np.random.randn(8, 16).astype(np.float32) * 0.1
    b1 = np.random.randn(8).astype(np.float32) * 0.01
    W2 = np.random.randn(4, 8).astype(np.float32) * 0.1
    b2 = np.random.randn(4).astype(np.float32) * 0.01

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    nodes = [
        helper.make_node('Gemm', ['input', 'W1', 'b1'], ['g1'], transB=1),
        helper.make_node('Relu', ['g1'], ['r1']),
        helper.make_node('Gemm', ['r1', 'W2', 'b2'], ['output'], transB=1),
    ]

    initializers = [
        numpy_helper.from_array(W1, 'W1'),
        numpy_helper.from_array(b1, 'b1'),
        numpy_helper.from_array(W2, 'W2'),
        numpy_helper.from_array(b2, 'b2'),
    ]

    graph = helper.make_graph(nodes, 'fp16_smoke_test', [X], [Y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fp16_smoke_test.onnx')
    onnx.save(model, out_path)
    print(f"Saved FP16 smoke model to {out_path}")


if __name__ == '__main__':
    main()
