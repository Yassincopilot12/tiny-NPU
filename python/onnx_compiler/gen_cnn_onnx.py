#!/usr/bin/env python3
"""
Generate a minimal CNN ONNX model:
  [1,1,8,8] -> Conv(k=3, out=4, pad=0) -> Relu -> Flatten -> Gemm -> [1,out]
Output spatial: (8-3+1)=6, so 6x6=36 per channel, 4 channels = 144
Final Gemm: 144 -> 4

Saves to models/conv1x_mini.onnx
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
    np.random.seed(123)

    in_c, out_c, kh, kw = 1, 4, 3, 3
    H, W = 8, 8
    out_h, out_w = H - kh + 1, W - kw + 1  # 6, 6
    flat_dim = out_c * out_h * out_w  # 4 * 36 = 144
    fc_out = 4

    # Conv weights: [out_c, in_c, kh, kw]
    conv_w = np.random.randn(out_c, in_c, kh, kw).astype(np.float32) * 0.1
    conv_b = np.random.randn(out_c).astype(np.float32) * 0.01

    # FC weights: [fc_out, flat_dim]
    fc_w = np.random.randn(fc_out, flat_dim).astype(np.float32) * 0.1
    fc_b = np.random.randn(fc_out).astype(np.float32) * 0.01

    # Build graph
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, in_c, H, W])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, fc_out])

    conv_w_init = numpy_helper.from_array(conv_w, name='conv_w')
    conv_b_init = numpy_helper.from_array(conv_b, name='conv_b')
    fc_w_init = numpy_helper.from_array(fc_w, name='fc_w')
    fc_b_init = numpy_helper.from_array(fc_b, name='fc_b')

    conv_node = helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'],
                                  kernel_shape=[kh, kw], pads=[0, 0, 0, 0])
    relu_node = helper.make_node('Relu', ['conv_out'], ['relu_out'])

    # Flatten from axis=1
    flatten_shape = numpy_helper.from_array(
        np.array([1, flat_dim], dtype=np.int64), name='flatten_shape')
    reshape_node = helper.make_node('Reshape', ['relu_out', 'flatten_shape'], ['flat_out'])

    fc_node = helper.make_node('Gemm', ['flat_out', 'fc_w', 'fc_b'], ['output'],
                                alpha=1.0, beta=1.0, transB=1)

    graph = helper.make_graph(
        [conv_node, relu_node, reshape_node, fc_node],
        'conv1x_mini',
        [X], [Y],
        initializer=[conv_w_init, conv_b_init, fc_w_init, fc_b_init, flatten_shape]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'conv1x_mini.onnx')
    onnx.save(model, out_path)
    print(f"Saved CNN model to {out_path}")
    print(f"  Input:  [1, {in_c}, {H}, {W}]")
    print(f"  Conv:   {out_c} filters, {kh}x{kw}, no padding -> [{out_c}, {out_h}, {out_w}]")
    print(f"  Flatten: {flat_dim}")
    print(f"  FC:     {flat_dim} -> {fc_out}")

if __name__ == '__main__':
    main()
