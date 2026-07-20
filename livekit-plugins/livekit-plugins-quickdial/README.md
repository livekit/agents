# LiveKit Plugins Quickdial

[Quickdial](https://quickdial.ai/) plugin for LiveKit Agents. Quickdial is a
CPU-optimized, real-time voice API — lifelike text-to-speech and accurate
speech-to-text over REST + WebSocket, no GPU, priced per character.

See [https://docs.livekit.io/agents/integrations/](https://docs.livekit.io/agents/integrations/) for more information.

## Installation

```bash
pip install livekit-plugins-quickdial
```

## Pre-requisites

You'll need an API key from Quickdial. Sign up at
[web.quickdial.ai](https://web.quickdial.ai) (1000 free credits, no card
required) and set it as an environment variable: `QUICKDIAL_API_KEY`.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import quickdial, silero

session = AgentSession(
    stt=quickdial.STT(language="en"),   # POST /v1/stt (whisper.cpp)
    tts=quickdial.TTS(voice="alba"),    # POST /v1/tts, 24 kHz
    vad=silero.VAD.load(),
    llm=...,
)
```

## Parameters

### `quickdial.TTS`

| Parameter     | Default                    | Description                                             |
| ------------- | -------------------------- | ------------------------------------------------------- |
| `voice`       | `"alba"`                   | Voice name (see `GET /v1/voices`).                      |
| `api_key`     | `QUICKDIAL_API_KEY`        | Quickdial API key.                                      |
| `base_url`    | `https://api.quickdial.ai` | API base URL.                                           |
| `sample_rate` | `24000`                    | Output sample rate (Hz).                                |

### `quickdial.STT`

| Parameter     | Default                    | Description               |
| ------------- | -------------------------- | ------------------------- |
| `language`    | `"en"`                     | Transcription language.   |
| `api_key`     | `QUICKDIAL_API_KEY`        | Quickdial API key.        |
| `base_url`    | `https://api.quickdial.ai` | API base URL.             |

## Additional resources

- [Quickdial API docs](https://quickdial.ai/docs)
- [Source repository](https://github.com/samay-ai/livekit-plugins-quickdial)
