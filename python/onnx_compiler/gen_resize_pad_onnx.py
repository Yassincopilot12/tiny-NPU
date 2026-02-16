#!/usr/bin/env python3
"""
Generate ONNX model testing Pad + Resize (nearest neighbor).
  [1, 2, 4, 4] -> Pad(1,1,1,1) -> [1, 2, 6, 6] -> Resize(2x) -> [1, 2, 12, 12]
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
    np.random.seed(42)

    C, H, W = 2, 4, 4

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, C, H, W])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, C, 12, 12])

    # Pad: [0,0,1,1, 0,0,1,1] = no pad on N,C, pad 1 on each side of H,W
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    pads_init = numpy_helper.from_array(pads, 'pads')
    constant_value = numpy_helper.from_array(np.array([0.0], dtype=np.float32), 'constant_value')

    pad_node = helper.make_node(
        'Pad',
        inputs=['input', 'pads', 'constant_value'],
        outputs=['padded'],
        mode='constant'
    )

    # Resize: nearest, 2x on spatial dims
    # Empty roi and scales, use sizes
    roi = numpy_helper.from_array(np.array([], dtype=np.float32), 'roi')
    scales = numpy_helper.from_array(np.array([], dtype=np.float32), 'scales')
    sizes = numpy_helper.from_array(np.array([1, C, 12, 12], dtype=np.int64), 'sizes')

    resize_node = helper.make_node(
        'Resize',
        inputs=['padded', 'roi', 'scales', 'sizes'],
        outputs=['output'],
        mode='nearest'
    )

    graph = helper.make_graph(
        [pad_node, resize_node],
        'resize_pad_test',
        [X], [Y],
        initializer=[pads_init, constant_value, roi, scales, sizes]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'resize_pad_test.onnx')
    onnx.save(model, out_path)
    print(f"Saved Resize+Pad model to {out_path}")
    print(f"  [1,{C},{H},{W}] -> Pad(1,1,1,1) -> [1,{C},6,6] -> Resize(2x) -> [1,{C},12,12]")


if __name__ == '__main__':
    main()
