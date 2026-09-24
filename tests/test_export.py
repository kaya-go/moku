"""The exported graph must stay loadable by ONNX Runtime Web (Kaya)."""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from moku.export import _float32_trig


def _trig_model(path):
    """float32 in -> float64 -> Sin and Cos -> Add -> float32 out, like the position embedding."""
    nodes = [
        helper.make_node("Cast", ["x"], ["x64"], to=TensorProto.DOUBLE),
        helper.make_node("Sin", ["x64"], ["s"], name="Sin"),
        helper.make_node("Cos", ["x64"], ["c"], name="Cos"),
        helper.make_node("Add", ["s", "c"], ["y64"]),
        helper.make_node("Cast", ["y64"], ["y"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes,
        "trig",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.save(model, str(path))


def _run(path, x):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]


def test_float32_trig_rewrites_double_sin_cos(tmp_path):
    path = tmp_path / "trig.onnx"
    _trig_model(path)
    x = np.linspace(-50, 50, 64, dtype=np.float32)[None]
    before = _run(path, x)

    assert _float32_trig(path) == 2
    model = onnx.load(str(path))
    dtypes = {
        vi.name: vi.type.tensor_type.elem_type for vi in onnx.shape_inference.infer_shapes(model).graph.value_info
    }
    trig_inputs = [dtypes[n.input[0]] for n in model.graph.node if n.op_type in ("Sin", "Cos")]
    assert trig_inputs == [TensorProto.FLOAT, TensorProto.FLOAT]
    np.testing.assert_allclose(_run(path, x), before, atol=1e-6)

    assert _float32_trig(path) == 0  # idempotent
