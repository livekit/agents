# Sherpa-ONNX plugin for LiveKit Agents

Local speech-to-text and text-to-speech with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Runs in process on ONNX Runtime: no API key, no network at runtime, no torch, no GPU required.

## Installation

```bash
pip install livekit-plugins-sherpa
```

Fetch the models once while online. `download-files` gets the default streaming STT and Kokoro; name others in `LIVEKIT_SHERPA_MODELS`:

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

## TTS

`sherpa.TTS` runs a sherpa-onnx offline TTS model. Kokoro is the default; Kitten and Piper are built in, and any other sherpa-onnx TTS model (VITS, Matcha, newer Kokoro) works through `engine=`. The phonemizer data ships with each model, so there is no system espeak dependency.

```python
session = AgentSession(tts=sherpa.TTS(voice="af_heart"), ...)
session = AgentSession(tts=sherpa.TTS(model="kitten-nano-en-v0_2", voice="expr-voice-3-f"), ...)

# any other sherpa-onnx TTS model: pick the voice by id, or pass its speaker names
engine = sherpa_onnx.OfflineTts(config)
session = AgentSession(tts=sherpa.TTS(engine=engine, voice=0), ...)
```

| Built-in model | Size | Voices |
| --- | --- | --- |
| `kokoro-int8-multi-lang-v1_0` (default) | 82M parameters, int8, 24 kHz | 54, e.g. `af_heart` (`sherpa.KokoroVoice`) |
| `kitten-nano-en-v0_2` | about 15M parameters, 24 kHz | 8, `expr-voice-2-m` to `expr-voice-5-f` |
| `piper-en_US-amy-low` | 63 MB, 16 kHz | 1, `amy` |

| Option | Default | Description |
| --- | --- | --- |
| `model` | `kokoro-int8-multi-lang-v1_0` | Built-in model |
| `voice` | the model's default | A speaker name from the model's table, or a speaker id |
| `speed` | `1.0` | Speaking rate |
| `num_threads` | `2` | ONNX Runtime threads |
| `engine` | `None` | Any `sherpa_onnx.OfflineTts` |
| `speakers` | `None` | Speaker names of `engine`, in id order, so `voice` can be a name |

Capabilities: `streaming=False`. `AgentSession` wraps it in `tts.StreamAdapter`, which splits LLM output into sentences. Each sentence's audio is pushed through `AudioEmitter` as sherpa-onnx produces it, and an interruption stops generation at the next chunk. `aligned_transcript` is off: sherpa-onnx does not expose phoneme durations.

Kokoro's English phonemes come from [misaki](https://github.com/hexgrad/misaki)'s dictionaries (the model's `lexicon-us-en.txt`), with espeak-ng for words outside the lexicon.
