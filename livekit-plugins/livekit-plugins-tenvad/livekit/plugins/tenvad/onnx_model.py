import os
import platform
from ctypes import CDLL, POINTER, c_float, c_int, c_int32, c_size_t, c_void_p

import numpy as np

SUPPORTED_SAMPLE_RATES = [16000]


class TenVad:
    """TEN VAD implementation loaded from native library"""

    def __init__(self, hop_size: int = 256, threshold: float = 0.5):
        self.hop_size = hop_size
        self.threshold = threshold

        base_dir = os.path.join(os.path.dirname(__file__))

        if platform.system() == "Linux" and platform.machine() == "x86_64":
            git_path = os.path.join(base_dir, "prebuilt/Linux/x64/libten_vad")

        elif platform.system() == "Darwin":
            git_path = os.path.join(base_dir, "prebuilt/macOS/ten_vad.framework/Versions/A/ten_vad")

        elif platform.system().upper() == "WINDOWS":
            if platform.machine().upper() in ["X64", "X86_64", "AMD64"]:
                git_path = os.path.join(base_dir, "prebuilt/Windows/x64/ten_vad.dll")
            else:
                git_path = os.path.join(base_dir, "prebuilt/Windows/x86/ten_vad.dll")
        else:
            raise NotImplementedError(
                f"Unsupported platform: {platform.system()} {platform.machine()}"
            )

        self.vad_library = CDLL(git_path)

        self.vad_handler = c_void_p(0)
        self.out_probability = c_float()
        self.out_flags = c_int32()

        self.vad_library.ten_vad_create.argtypes = [
            POINTER(c_void_p),
            c_size_t,
            c_float,
        ]
        self.vad_library.ten_vad_create.restype = c_int

        self.vad_library.ten_vad_destroy.argtypes = [POINTER(c_void_p)]
        self.vad_library.ten_vad_destroy.restype = c_int

        self.vad_library.ten_vad_process.argtypes = [
            c_void_p,
            c_void_p,
            c_size_t,
            POINTER(c_float),
            POINTER(c_int32),
        ]
        self.vad_library.ten_vad_process.restype = c_int

        self.create_and_init_handler()

    def create_and_init_handler(self):
        assert (
            self.vad_library.ten_vad_create(
                POINTER(c_void_p)(self.vad_handler),
                c_size_t(self.hop_size),
                c_float(self.threshold),
            )
            == 0
        ), "[TEN VAD]: create handler failure!"

    def __del__(self):
        if hasattr(self, "vad_library") and hasattr(self, "vad_handler"):
            self.vad_library.ten_vad_destroy(POINTER(c_void_p)(self.vad_handler))

    def get_input_data(self, audio_data: np.ndarray):
        audio_data = np.squeeze(audio_data)
        assert len(audio_data.shape) == 1 and audio_data.shape[0] == self.hop_size, (
            f"[TEN VAD]: audio data shape should be [{self.hop_size}]"
        )
        assert audio_data.dtype == np.int16, "[TEN VAD]: audio data type error, must be int16"
        data_pointer = audio_data.__array_interface__["data"][0]
        return c_void_p(data_pointer)

    def process(self, audio_data: np.ndarray):
        input_pointer = self.get_input_data(audio_data)
        self.vad_library.ten_vad_process(
            self.vad_handler,
            input_pointer,
            c_size_t(self.hop_size),
            POINTER(c_float)(self.out_probability),
            POINTER(c_int32)(self.out_flags),
        )
        return self.out_probability.value, self.out_flags.value


class OnnxModel:
    """Wrapper around TenVad to maintain compatibility with existing VAD interface"""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        activation_threshold: float = 0.5,
    ) -> None:
        self._sample_rate = sample_rate

        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("TEN VAD only supports 8KHz and 16KHz sample rates")

        self._window_size_samples = 256
        self._ten_vad = TenVad(hop_size=self._window_size_samples, threshold=activation_threshold)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def window_size_samples(self) -> int:
        return self._window_size_samples

    @property
    def context_size(self) -> int:
        return self._context_size

    def update_threshold(self, threshold: float) -> None:
        self._ten_vad = TenVad(hop_size=self._window_size_samples, threshold=threshold)

    def __call__(self, x: np.ndarray) -> float:
        probability, _ = self._ten_vad.process(x)
        return probability
