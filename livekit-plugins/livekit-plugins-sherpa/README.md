# Sherpa-ONNX plugin for LiveKit Agents

Local speech-to-text with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Runs in process on ONNX Runtime: no API key, no network at runtime, no torch, no GPU required.

## Installation

```bash
pip install livekit-plugins-sherpa
```

Fetch the models once while online. `download-files` gets the default streaming model; name others in `LIVEKIT_SHERPA_MODELS`:

```bash
python -m livekit.agents download-files
LIVEKIT_SHERPA_MODELS=zipformer-en-2023-06-26,moonshine-tiny-en-int8 python -m livekit.agents download-files
```

Models are pinned to Hugging Face commits and loaded with `local_files_only`, so the agent never downloads at runtime.

## Streaming STT

`sherpa.STT` runs a sherpa-onnx online recognizer. It emits interim transcripts while the user speaks and a final transcript with word timings once they pause. The final transcript always precedes `END_OF_SPEECH` for the same utterance.

```python
from livekit.plugins import sherpa

session = AgentSession(stt=sherpa.STT(), ...)
```

| Option | Default | Description |
| --- | --- | --- |
| `model` | `zipformer-en-2023-06-26` | Built-in model (English zipformer transducer, about 70M parameters, int8) |
| `endpoint_silence` | `0.5` | Trailing silence, in seconds, that ends an utterance |
| `num_threads` | `2` | ONNX Runtime threads |
| `recognizer` | `None` | Any `sherpa_onnx.OnlineRecognizer` with endpoint detection enabled |
| `language` | model language | Language reported on transcripts |

Capabilities: `streaming=True`, `interim_results=True`, `aligned_transcript="word"`.

`zipformer-en-20M-2023-02-17` is also built in. It is smaller but drops the first words of an utterance on audio resampled from 48 kHz, so it is not the default. The built-in models are wideband; for 8 kHz telephony audio pass an 8 kHz model through `recognizer=`.

## Batch STT

`sherpa.BatchSTT` runs a sherpa-onnx offline recognizer, which transcribes a whole utterance at once: Whisper, Moonshine, SenseVoice, Parakeet and the other offline models. `AgentSession` uses it live by wrapping it in `stt.StreamAdapter` with the session VAD.

```python
session = AgentSession(stt=sherpa.BatchSTT(model="moonshine-tiny-en-int8"), ...)

# any other sherpa-onnx offline model
import sherpa_onnx

recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(model=..., tokens=...)
session = AgentSession(stt=sherpa.BatchSTT(recognizer=recognizer), ...)
```

| Built-in model | Size |
| --- | --- |
| `moonshine-tiny-en-int8` (default) | about 27M parameters, 120 MB |
| `whisper-tiny.en-int8` | about 39M parameters, 100 MB |

Capabilities: `streaming=False`, `interim_results=False`.
