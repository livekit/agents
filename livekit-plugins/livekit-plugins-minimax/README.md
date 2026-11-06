# MiniMax plugin for LiveKit Agents

Support for speech synthesis (TTS) and speech recognition (STT) with
[MiniMax](https://www.minimaxi.com/)'s voice AI services.

See the [MiniMax TTS integration docs](https://docs.livekit.io/agents/models/tts/minimax/)
for more information.

## Installation

```bash
pip install livekit-plugins-minimax-ai
```

## Pre-requisites

You'll need an API key from MiniMax. It can be set as an environment variable: `MINIMAX_API_KEY`

## Usage

### Speech-to-Text

MiniMax transcribes a complete audio file rather than a continuous stream, so
`capabilities.streaming` is `False`. Pass a VAD to the `AgentSession` (or the `Agent`) and
LiveKit wraps the plugin in a `StreamAdapter`, which calls `recognize()` once per detected
speech segment.

```python
from livekit.agents import AgentSession
from livekit.plugins import minimax, silero

session = AgentSession(
    vad=silero.VAD.load(),
    # leave `language` unset to enable mixed-language (code-switched) recognition
    stt=minimax.STT(),
)
```

Hint the primary spoken language with a BCP-47 primary subtag, or ask for speaker
diarization and timestamps:

```python
# hint the primary language; omit it for mixed zh/en speech
stt = minimax.STT(language="zh")

# diarization and timestamps
stt = minimax.STT(response_format="verbose_json")
```

The transcript can also be delivered incrementally with SSE by passing `stream=True`, which is
supported with `response_format="json"` only — diarization requires the complete audio.

### Text-to-Speech

```python
from livekit.plugins import minimax

tts = minimax.TTS(model="speech-02-turbo", voice="socialmedia_female_2_v1")
```
