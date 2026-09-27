# Sprag plugin for LiveKit Agents

Support for [Sprag](https://sprag.ai/) speech to text, LLM, and text to speech via the
OpenAI-compatible API at `https://api.sprag.ai/v1`.

See [https://sprag.ai/docs/integrations/livekit](https://sprag.ai/docs/integrations/livekit) for more information.

## Installation

```bash
pip install livekit-plugins-sprag
```

## Pre-requisites

You'll need an API key from Sprag. It can be passed directly or set as the
`SPRAG_API_KEY` environment variable.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import sprag

session = AgentSession(
    stt=sprag.STT(model="rhythm"),
    llm=sprag.LLM(model="symphony"),
    tts=sprag.TTS(model="chorus-clone", voice="wade"),
)
```

`STT` and `LLM` reuse the OpenAI plugin's transports with `base_url="https://api.sprag.ai/v1"`.
`TTS` streams speech over Sprag's realtime WebSocket, keeping a warm connection between turns.
LLM, TTS, and REST transcription requests carry an `X-Sprag-Integration` attribution header.
