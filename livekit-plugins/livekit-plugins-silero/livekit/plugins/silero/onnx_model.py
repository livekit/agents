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

        if sample_rate == 8000:
            self._window_size_samples = 256
            self._context_size = 32
        elif sample_rate == 16000:
            self._window_size_samples = 512
            self._context_size = 64

        self.reset_states()

    @property
    def window_size_samples(self) -> int:
        return self._window_size_samples

    @property
    def context_size(self) -> int:
        return self._context_size

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def __call__(self, x: np.ndarray) -> float:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        if x.shape[1] != self._window_size_samples:
            raise ValueError(
                f"Input shape must be (N, {self._window_size_samples}), got {x.shape}"
            )

        x = np.concatenate([self._context, x], axis=1)
        ort_inputs = {
            "input": x,
            "state": self._state,
            "sr": np.array(self._sample_rate, dtype=np.int64),
        }
        out, self._state = self._sess.run(None, ort_inputs)
        self._context = x[..., -self._context_size :]
        return out.item()
