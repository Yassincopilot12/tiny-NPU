#!/usr/bin/env python3
"""
Generate 50 random fuzz-test ONNX models with random combinations of ops.
Each model is a chain of element-wise / shape-preserving operations to
avoid shape mismatches. Input and output share the same shape.

Supported ops for fuzzing:
  - Relu, Exp, Abs, Neg, Sigmoid (element-wise unary, shape-preserving)
  - Add (binary with constant, shape-preserving)
  - Clip (as ReLU6: min=0, max=6)
  - Binary element-wise: Add, Sub, Mul with random constant (20% chance)
  - Pad + MaxPool block (10% chance, only for NCHW shapes with H,W >= 4)

Each case has 5-15 random nodes chained together.
Saves to models/fuzz/case_N.onnx (N=0..49)
"""
import os
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("ERROR: onnx package not installed. Run: pip install onnx")
    exit(1)


# Element-wise unary ops that preserve shape (only ops supported by compiler)
UNARY_OPS = ['Relu', 'Exp', 'Log', 'Sqrt']

# Binary element-wise ops (used with a random constant initializer)
BINARY_OPS = ['Add', 'Sub', 'Mul']


def _is_nchw(shape):
    """Check if shape is [1, C, H, W] with H >= 4 and W >= 4."""
    return (len(shape) == 4
            and shape[0] == 1
            and shape[2] >= 4
            and shape[3] >= 4)


def _shape_after_pad_pool(shape):
    """Return shape after Pad(1,1,1,1 on H,W) then MaxPool(2x2, stride 2).
    Pad adds 2 to H and W, then pool halves them.
    Net result: H' = (H+2)//2, W' = (W+2)//2.
    """
    return [shape[0], shape[1], (shape[2] + 2) // 2, (shape[3] + 2) // 2]


def make_fuzz_model(rng, case_id):
    """Build a single fuzz-test model with random op chain."""
    # Decide whether to use NCHW shape (allows Pad+Pool blocks) or flat shape
    use_nchw = rng.rand() < 0.3  # 30% chance of NCHW layout

    if use_nchw:
        # Pick small NCHW shapes that fit in SRAM (< 4096 elements total)
        c_choices = [1, 2, 4]
        hw_choices = [4, 8, 16]
        c = c_choices[rng.randint(0, len(c_choices))]
        h = hw_choices[rng.randint(0, len(hw_choices))]
        w = hw_choices[rng.randint(0, len(hw_choices))]
        # Ensure total elements < 4096 (float32 = 4 bytes, so < 16384 bytes)
        while c * h * w > 4096:
            h = max(4, h // 2)
            w = max(4, w // 2)
        shape = [1, c, h, w]
    else:
        # Flat [1, dim] shapes like before
        dim_choices = [4, 8, 16, 32, 64]
        dim = dim_choices[rng.randint(0, len(dim_choices))]
        shape = [1, dim]

    num_nodes = rng.randint(5, 16)  # 5 to 15 nodes

    # Current working shape (may change after Pad+Pool blocks)
    current_shape = list(shape)

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)

    nodes = []
    initializers = []
    prev_output = 'input'
    init_counter = 0  # unique counter for initializer names

    for i in range(num_nodes):
        roll = rng.rand()

        # Determine what kind of op to generate
        # 10% chance: Pad + MaxPool block (only if NCHW with H,W >= 4)
        # 20% chance: binary element-wise with constant
        # remaining: unary (including Clip as ReLU6 alternative)

        if roll < 0.10 and _is_nchw(current_shape) and current_shape[2] >= 4 and current_shape[3] >= 4:
            # --- Pad + MaxPool block ---
            # Check that resulting shape still has enough elements
            new_shape = _shape_after_pad_pool(current_shape)
            if new_shape[2] >= 2 and new_shape[3] >= 2:
                # Pad node
                pad_out = f'pad_{i}_out'
                pads_name = f'pads_{i}'
                # pads = [0,0,1,1, 0,0,1,1] for NCHW: pad H and W by 1 on each side
                pads_data = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
                pads_init = numpy_helper.from_array(pads_data, name=pads_name)
                initializers.append(pads_init)

                constant_value_name = f'pad_val_{i}'
                constant_value = numpy_helper.from_array(
                    np.array(0.0, dtype=np.float32), name=constant_value_name)
                initializers.append(constant_value)

                pad_node = helper.make_node(
                    'Pad',
                    inputs=[prev_output, pads_name, constant_value_name],
                    outputs=[pad_out],
                    mode='constant'
                )
                nodes.append(pad_node)
                prev_output = pad_out

                # MaxPool node
                pool_out = f'pool_{i}_out'
                pool_node = helper.make_node(
                    'MaxPool',
                    inputs=[prev_output],
                    outputs=[pool_out],
                    kernel_shape=[2, 2],
                    strides=[2, 2]
                )
                nodes.append(pool_node)
                prev_output = pool_out
                current_shape = new_shape
                continue
            # If shape too small after pool, fall through to unary

        elif roll < 0.30:
            # --- Binary element-wise op with random constant ---
            bin_op = BINARY_OPS[rng.randint(0, len(BINARY_OPS))]

            # Generate a safe constant (small values to avoid overflow)
            if bin_op == 'Mul':
                const_val = rng.uniform(0.5, 2.0)
            elif bin_op == 'Sub':
                const_val = rng.uniform(-1.0, 1.0)
            else:  # Add
                const_val = rng.uniform(-1.0, 1.0)

            const_name = f'const_{init_counter}'
            init_counter += 1

            # Broadcast-compatible scalar constant
            const_tensor = numpy_helper.from_array(
                np.array(const_val, dtype=np.float32).reshape([1]),
                name=const_name
            )
            initializers.append(const_tensor)

            out_name = f'node_{i}_out'
            bin_node = helper.make_node(
                bin_op,
                inputs=[prev_output, const_name],
                outputs=[out_name]
            )
            nodes.append(bin_node)
            prev_output = out_name
            continue

        # --- Unary op (including Clip as ReLU6 alternative) ---
        # 20% chance of Clip(0,6) instead of a regular unary op
        if rng.rand() < 0.20:
            op = 'Clip'
        else:
            op = UNARY_OPS[rng.randint(0, len(UNARY_OPS))]

        # Avoid Exp after Exp to prevent overflow; force Relu to clamp
        if (len(nodes) > 0 and nodes[-1].op_type == 'Exp'
                and op == 'Exp'):
            op = 'Relu'

        # Log/Sqrt need non-negative input; insert Relu before them if prev was not Relu/Exp
        if op in ('Log', 'Sqrt') and len(nodes) > 0 and nodes[-1].op_type not in ('Relu', 'Exp', 'Clip'):
            guard_name = f'guard_{i}_out'
            guard_node = helper.make_node('Relu', [prev_output], [guard_name])
            nodes.append(guard_node)
            prev_output = guard_name

        out_name = f'node_{i}_out'

        if op == 'Clip':
            # Clip(min=0, max=6) - acts as ReLU6
            min_name = f'clip_min_{init_counter}'
            max_name = f'clip_max_{init_counter}'
            init_counter += 1

            min_init = numpy_helper.from_array(
                np.array(0.0, dtype=np.float32), name=min_name)
            max_init = numpy_helper.from_array(
                np.array(6.0, dtype=np.float32), name=max_name)
            initializers.append(min_init)
            initializers.append(max_init)

            node = helper.make_node(
                'Clip',
                inputs=[prev_output, min_name, max_name],
                outputs=[out_name]
            )
        else:
            node = helper.make_node(op, [prev_output], [out_name])

        nodes.append(node)
        prev_output = out_name

    # Final output: need to match the final shape after any Pad+Pool blocks
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, current_shape)

    # If the last node's output is not named 'output', rename it
    if nodes[-1].output[0] != 'output':
        # Add an identity node to map to 'output'
        identity_node = helper.make_node('Relu', [prev_output], ['output'])
        nodes.append(identity_node)

    graph = helper.make_graph(
        nodes,
        f'fuzz_case_{case_id}',
        [X], [Y],
        initializer=initializers
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7

    onnx.checker.check_model(model)
    return model


def main():
    rng = np.random.RandomState(1337)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'fuzz')
    os.makedirs(out_dir, exist_ok=True)

    num_cases = 50
    for case_id in range(num_cases):
        model = make_fuzz_model(rng, case_id)
        out_path = os.path.join(out_dir, f'case_{case_id}.onnx')
        onnx.save(model, out_path)

    print(f"Saved {num_cases} fuzz models to {out_dir}")
    print(f"  Cases:      case_0.onnx .. case_{num_cases - 1}.onnx")
    print(f"  Ops/model:  5-15 random nodes")
    print(f"  Op types:   {UNARY_OPS + ['Clip'] + BINARY_OPS + ['Pad+MaxPool']}")


if __name__ == '__main__':
    main()
