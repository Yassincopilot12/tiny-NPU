#!/usr/bin/env python3
"""
Generate ONNX model testing mixed precision CNN ops.
  [1, 2, 8, 8] -> Conv(4, k=3, pad=1) -> ReLU -> MaxPool(k=2,s=2) -> [1, 4, 4, 4]
Tests Conv lowering + MaxPool2D engine + ReLU.
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
    np.random.seed(55)

    C_in, C_out = 2, 4
    H, W = 8, 8
    kh, kw = 3, 3

    W_conv = np.random.randn(C_out, C_in, kh, kw).astype(np.float32) * 0.1
    b_conv = np.random.randn(C_out).astype(np.float32) * 0.01

    out_h = (H + 2*1 - kh) // 1 + 1  # 8 with pad=1
    out_w = (W + 2*1 - kw) // 1 + 1  # 8 with pad=1
    pool_h = out_h // 2  # 4
    pool_w = out_w // 2  # 4

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, C_in, H, W])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, C_out, pool_h, pool_w])

    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'W_conv', 'b_conv'],
        outputs=['conv_out'],
        kernel_shape=[kh, kw],
        strides=[1, 1],
        pads=[1, 1, 1, 1]
    )

    relu_node = helper.make_node('Relu', ['conv_out'], ['relu_out'])

    pool_node = helper.make_node(
        'MaxPool',
        inputs=['relu_out'],
        outputs=['output'],
        kernel_shape=[2, 2],
        strides=[2, 2]
    )

    initializers = [
        numpy_helper.from_array(W_conv, 'W_conv'),
        numpy_helper.from_array(b_conv, 'b_conv'),
    ]

    graph = helper.make_graph(
        [conv_node, relu_node, pool_node],
        'mixed_precision_cnn_test',
        [X], [Y],
        initializer=initializers
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'mixed_precision_cnn_test.onnx')
    onnx.save(model, out_path)
    print(f"Saved mixed precision CNN model to {out_path}")
    print(f"  [1,{C_in},{H},{W}] -> Conv({C_out},k=3,pad=1) -> ReLU -> MaxPool(k=2,s=2) -> [1,{C_out},{pool_h},{pool_w}]")


if __name__ == '__main__':
    main()
