import os
import platform
from ctypes import CDLL, POINTER, c_float, c_int, c_int32, c_size_t, c_void_p

import numpy as np


class TenVad:
    def __init__(self, hop_size: int = 256, threshold: float = 0.5):
        self.hop_size = hop_size
        self.threshold = threshold
        if platform.system() == "Linux" and platform.machine() == "x86_64":
            git_path = os.path.join(
                os.path.dirname(os.path.relpath(__file__)), "../lib/Linux/x64/libten_vad.so"
            )
            if os.path.exists(git_path):
                print(f"git_path: {git_path}")
                self.vad_library = CDLL(git_path)
            else:
                pip_path = os.path.join(
                    os.path.dirname(os.path.relpath(__file__)), "./ten_vad_library/libten_vad.so"
                )
                self.vad_library = CDLL(pip_path)

        elif platform.system() == "Darwin":
            git_path = os.path.join(
                os.path.dirname(os.path.relpath(__file__)),
                "../lib/macOS/ten_vad.framework/Versions/A/ten_vad",
            )
            if os.path.exists(git_path):
                self.vad_library = CDLL(git_path)
            else:
                pip_path = os.path.join(
                    os.path.dirname(os.path.relpath(__file__)), "./ten_vad_library/libten_vad"
                )
                self.vad_library = CDLL(pip_path)
        elif platform.system().upper() == "WINDOWS":
            if platform.machine().upper() in ["X64", "X86_64", "AMD64"]:
                git_path = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "../lib/Windows/x64/ten_vad.dll"
                )
                if os.path.exists(git_path):
                    self.vad_library = CDLL(git_path)
                else:
                    pip_path = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)), "./ten_vad_library/ten_vad.dll"
                    )
                    self.vad_library = CDLL(pip_path)
            else:
                git_path = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "../lib/Windows/x86/ten_vad.dll"
                )
                if os.path.exists(git_path):
                    self.vad_library = CDLL(git_path)
                else:
                    pip_path = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)), "./ten_vad_library/ten_vad.dll"
                    )
                    self.vad_library = CDLL(pip_path)
        else:
            raise NotImplementedError(
                f"Unsupported platform: {platform.system()} {platform.machine()}"
            )
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
        assert self.vad_library.ten_vad_destroy(POINTER(c_void_p)(self.vad_handler)) == 0, (
            "[TEN VAD]: destroy handler failure!"
        )

    def get_input_data(self, audio_data: np.ndarray):
        audio_data = np.squeeze(audio_data)
        assert len(audio_data.shape) == 1 and audio_data.shape[0] == self.hop_size, (
            "[TEN VAD]: audio data shape should be [%d]" % (self.hop_size)
        )
        assert type(audio_data[0]) == np.int16, "[TEN VAD]: audio data type error, must be int16"
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
