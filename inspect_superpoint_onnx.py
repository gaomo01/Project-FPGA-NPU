from collections import Counter

import onnx


MODEL_PATH = "superpoint_test.onnx"
WARN_OPS = {"Reshape", "Transpose", "Softmax", "Slice"}


def tensor_shape(value_info):
    dims = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def main():
    model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(model)
    graph = model.graph

    print(f"Model: {MODEL_PATH}")
    print("Inputs:")
    for value in graph.input:
        print(f"  - {value.name}: {tensor_shape(value)}")

    print("Outputs:")
    for value in graph.output:
        print(f"  - {value.name}: {tensor_shape(value)}")

    op_counts = Counter(node.op_type for node in graph.node)
    found_warn_ops = {op: op_counts[op] for op in WARN_OPS if op_counts[op]}

    print("Operator counts:")
    for op, count in sorted(op_counts.items()):
        print(f"  - {op}: {count}")

    if found_warn_ops:
        print("WARNING: found NPU-unfriendly/postprocess operators:")
        for op, count in sorted(found_warn_ops.items()):
            print(f"  - {op}: {count}")
    else:
        print("OK: no Reshape/Transpose/Softmax/Slice operators found.")
        print("OK: pure fully-convolutional network, highly NPU-friendly.")


if __name__ == "__main__":
    main()
