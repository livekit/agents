from __future__ import annotations

import json
import math
import os

import numpy as np
import phonemizer

from livekit.agents.inference_runner import _InferenceRunner
from livekit.agents.utils import hw

from .model import (
    HG_MODEL,
    ONNX_FILENAME,
    VOICES_FILENAME,
)


def basic_english_tokenize(text):
    """Basic English tokenizer that splits on whitespace and punctuation."""
    import re

    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens


class TextCleaner:
    def __init__(self, dummy=None):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

        dicts = {}
        for i in range(len(symbols)):
            dicts[symbols[i]] = i

        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
        return indexes


def _ensure_espeak_library() -> None:
    # https://github.com/bootphon/phonemizer/issues/117
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/opt/espeak/lib/libespeak.dylib"


class _KittenRunner(_InferenceRunner):
    INFERENCE_METHOD = "lk_kitten_tts"

    def initialize(self) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]
        from huggingface_hub import errors, hf_hub_download

        repo_id = os.getenv("KITTENTTS_REPO_ID", HG_MODEL)
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=ONNX_FILENAME,
                local_files_only=True,
            )
            voices_path = hf_hub_download(
                repo_id=repo_id,
                filename=VOICES_FILENAME,
                local_files_only=True,
            )
        except (errors.LocalEntryNotFoundError, OSError):
            raise RuntimeError(
                "Kitten assets not found locally. Pre-download them first via"
                " `python myagent.py download-files` (ensure `from livekit.plugins"
                " import kitten` is imported)."
            ) from None

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = max(
            1, min(math.ceil(hw.get_cpu_monitor().cpu_count()) // 2, 4)
        )
        sess_options.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        voices = np.load(voices_path)
        self._voices = {k: voices[k] for k in voices.files}

        # Ensure espeak library is discoverable before initializing phonemizer
        _ensure_espeak_library()

        self._phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        self._text_cleaner = TextCleaner()

    @staticmethod
    def _to_pcm16_bytes(audio: np.ndarray) -> bytes:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    def run(self, data: bytes) -> bytes | None:
        payload = json.loads(data)
        text = payload["text"]
        voice = payload.get("voice")
        speed = float(payload.get("speed", 1.0))

        if voice not in self._voices:
            raise ValueError(f"unknown voice: {voice}")

        phonemes_list = self._phonemizer.phonemize([text])
        tokens_like = basic_english_tokenize(phonemes_list[0])
        phonemes = " ".join(tokens_like)
        tokens = self._text_cleaner(phonemes)

        # Add start and end tokens
        tokens.insert(0, 0)
        tokens.append(0)

        input_ids = np.array([tokens], dtype=np.int64)
        style = self._voices[voice].astype("float32")
        speed_arr = np.array([speed], dtype=np.float32)

        outputs = self._session.run(
            None, {"input_ids": input_ids, "style": style, "speed": speed_arr}
        )
        waveform = outputs[0]
        return self._to_pcm16_bytes(waveform)


_InferenceRunner.register_runner(_KittenRunner)
