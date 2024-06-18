import atexit
import importlib.resources
from contextlib import ExitStack

import numpy as np
import onnxruntime  # type: ignore

_resource_files = ExitStack()
atexit.register(_resource_files.close)


SUPPORTED_SAMPLE_RATES = [8000, 16000]


def new_inference_session(force_cpu: bool) -> onnxruntime.InferenceSession:
    res = (
        importlib.resources.files("livekit.plugins.silero.resources")
        / "silero_vad.onnx"
    )
    ctx = importlib.resources.as_file(res)
    path = _resource_files.enter_context(ctx)

    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    if force_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
        session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"], ess_options=opts
        )
    else:
        session = onnxruntime.InferenceSession(str(path), sess_options=opts)

    return session


class OnnxModel:
    def __init__(
        self, *, onnx_session: onnxruntime.InferenceSession, sample_rate: int
    ) -> None:
        self._sess = onnx_session
        self._sample_rate = sample_rate

        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("Silero VAD only supports 8KHz and 16KHz sample rates")

        self.reset_states()

    def reset_states(self) -> None:
        self._h = np.zeros((2, 1, 64)).astype(np.float32)
        self._c = np.zeros((2, 1, 64)).astype(np.float32)

    def __call__(self, x: np.ndarray) -> float:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        ort_inputs = {
            "input": x,
            "h": self._h,
            "c": self._c,
            "sr": np.array(self._sample_rate, dtype=np.int64),
        }
        ort_outputs = self._sess.run(None, ort_inputs)
        out, self._h, self._c = ort_outputs
        return out.item()
