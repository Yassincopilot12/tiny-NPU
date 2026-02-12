#!/usr/bin/env python3
"""
Generate a small MLP ONNX model: [1,32] -> Gemm -> Relu -> Gemm -> [1,8]
Architecture: 32 -> 16 -> 8
Saves to models/mlp_32_16_8.onnx
"""
import os
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("ERROR: onnx package not installed. Run: pip install onnx")
    exit(1)

def main():
    np.random.seed(42)

    # Layer 1: 32 -> 16
    W1 = np.random.randn(16, 32).astype(np.float32) * 0.1
    b1 = np.random.randn(16).astype(np.float32) * 0.01

    # Layer 2: 16 -> 8
    W2 = np.random.randn(8, 16).astype(np.float32) * 0.1
    b2 = np.random.randn(8).astype(np.float32) * 0.01

    # Build ONNX graph
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 32])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 8])

    W1_init = numpy_helper.from_array(W1, name='W1')
    b1_init = numpy_helper.from_array(b1, name='b1')
    W2_init = numpy_helper.from_array(W2, name='W2')
    b2_init = numpy_helper.from_array(b2, name='b2')

    gemm1 = helper.make_node('Gemm', ['input', 'W1', 'b1'], ['gemm1_out'],
                              alpha=1.0, beta=1.0, transB=1)
    relu1 = helper.make_node('Relu', ['gemm1_out'], ['relu1_out'])
    gemm2 = helper.make_node('Gemm', ['relu1_out', 'W2', 'b2'], ['output'],
                              alpha=1.0, beta=1.0, transB=1)

    graph = helper.make_graph(
        [gemm1, relu1, gemm2],
        'mlp_32_16_8',
        [X], [Y],
        initializer=[W1_init, b1_init, W2_init, b2_init]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7

    onnx.checker.check_model(model)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'mlp_32_16_8.onnx')
    onnx.save(model, out_path)
    print(f"Saved MLP model to {out_path}")
    print(f"  Input:  [1, 32]")
    print(f"  Hidden: 16 (ReLU)")
    print(f"  Output: [1, 8]")

if __name__ == '__main__':
    main()
