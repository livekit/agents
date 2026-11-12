# Vakyam AI plugin for LiveKit Agents

Support for voice synthesis with [Vakyam AI](https://vakyam.ai/) Raaga 1 —
text-to-speech for Indian languages.

See [https://docs.vakyam.ai/integrations/livekit](https://docs.vakyam.ai/integrations/livekit)
for provider docs.

## Installation

```bash
pip install livekit-plugins-vakyam
```

Or with the LiveKit Agents extra:

```bash
uv add "livekit-agents[vakyam]"
```

## Pre-requisites

You'll need an API key from [Vakyam](https://dashboard.vakyam.ai/api-keys).
Set it as an environment variable:

```bash
export VAKYAM_API_KEY="vak_live_..."
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import vakyam

session = AgentSession(
    tts=vakyam.TTS(
        model="raaga-v1",
        voice="Archana",
        language="ta-IN",
        sample_rate=24000,
    ),
    # ... stt, llm, vad
)
```

`stream()` uses the realtime WebSocket API and sentence-tokenizes LLM text so
each utterance is one complete sentence (Vakyam does not accept partial
tokens). `synthesize()` uses HTTP streaming (`POST /v1/tts/stream`) and
returns PCM audio.

WebSocket connections are pooled and reused between sequential agent turns.
Each active synthesis stream has exclusive ownership of its connection, so an
overlapping stream uses a separate connection. On interruption, the plugin
sends `cancel`, drains through Vakyam's cancellation acknowledgement, and
returns the healthy connection to the pool.
