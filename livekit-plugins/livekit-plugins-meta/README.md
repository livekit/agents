# Meta plugin for LiveKit Agents

Support for Meta [Muse Voice Transcribe](https://dev.meta.ai/docs/speech-to-text) using its [realtime streaming speech-to-text interface](https://dev.meta.ai/docs/api-reference/voice/realtime).

## Installation

```bash
pip install livekit-plugins-meta
```

## Pre-requisites

Set a Meta Model API key in your environment:

```bash
MODEL_API_KEY=<your_model_api_key>
```

You can also pass the key directly with `api_key=`.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import meta

session = AgentSession(
    stt=meta.STT(
        keywords=["LiveKit", "Muse"],
        language_bias=["English"],
    ),
    # ... llm, tts, etc.
)
```

The plugin supports streaming recognition with server-side endpointing, cumulative interim transcripts, and mono PCM16 audio at 24 kHz. Connected streams must keep sending real-time PCM, including silence. `keywords` and `language_bias` are static hints applied during the initial handshake and cannot be changed on an active stream. Omit `language_bias` for automatic language detection.

Supported language names are: Arabic, Bengali, Dutch, English, French, German, Hebrew, Hindi, Indonesian, Italian, Japanese, Kannada, Korean, Malay, Mandarin Chinese, Marathi, Polish, Portuguese, Spanish, Tagalog, Tamil, Telugu, Thai, Turkish, and Vietnamese. The per-stream `language=` argument also accepts corresponding language codes and locales, such as `en-US`, `pt-BR`, and `zh-CN`, and maps them to the documented names.

Muse realtime sessions have a maximum duration of 60 minutes. This plugin reports the provider close and does not rotate an active LiveKit speech stream automatically; start a new stream to continue. Batch recognition, diarization, detected-language metadata, and active-stream keyterm updates are not supported.
