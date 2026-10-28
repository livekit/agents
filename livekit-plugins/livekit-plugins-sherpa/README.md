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

## STT

`sherpa.STT` runs a sherpa-onnx recognizer, and the model decides how it runs:

- **Online models** stream: interim transcripts while the user speaks, a final transcript with word timings once they pause. The final transcript always precedes `END_OF_SPEECH` for the same utterance. Capabilities: `streaming=True`, `interim_results=True`, `aligned_transcript="word"`.
- **Offline models** (Whisper, Moonshine, SenseVoice, Parakeet and the rest) transcribe a whole utterance at once. Capabilities: `streaming=False`. `AgentSession` runs them live by wrapping them in `stt.StreamAdapter` with the session VAD.

```python
from livekit.plugins import sherpa

session = AgentSession(stt=sherpa.STT(), ...)  # streaming zipformer
session = AgentSession(stt=sherpa.STT(model="moonshine-tiny-en-int8"), ...)  # batch

# any other sherpa-onnx model, online or offline
import sherpa_onnx

recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(model=..., tokens=...)
session = AgentSession(stt=sherpa.STT(recognizer=recognizer), ...)
```

| Built-in model | Kind | Size |
| --- | --- | --- |
| `zipformer-en-2023-06-26` (default) | online | about 70M parameters, int8 |
| `zipformer-en-20M-2023-02-17` | online | about 20M parameters, int8 |
| `moonshine-tiny-en-int8` | offline | about 27M parameters |
| `whisper-tiny.en-int8` | offline | about 39M parameters |

| Option | Default | Description |
| --- | --- | --- |
| `model` | `zipformer-en-2023-06-26` | Built-in model |
| `endpoint_silence` | `0.5` | Online models: trailing silence, in seconds, that ends an utterance |
| `num_threads` | `2` | ONNX Runtime threads |
| `recognizer` | `None` | Any `sherpa_onnx.OnlineRecognizer` (endpoint detection enabled) or `sherpa_onnx.OfflineRecognizer` |
| `language` | model language | Language reported on transcripts |

`zipformer-en-20M-2023-02-17` is smaller but drops the first words of an utterance on audio resampled from 48 kHz, so it is not the default. The built-in models are wideband; for 8 kHz telephony audio pass an 8 kHz model through `recognizer=`.
