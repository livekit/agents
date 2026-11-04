# ReadAloud plugin for LiveKit Agents

Support for voice synthesis with the [ReadAloud](https://readaloudai.org) API.

ReadAloud streams audio as it is generated, so an agent can start speaking before a full reply is rendered, and it can output 8 kHz audio for phone calls. `stream()` splits LLM output into sentences and synthesizes each one with its own streaming request; the API returns no word timestamps, so transcripts are not time-aligned. Closing a stream (for example on a barge-in) closes the HTTP connection, which stops generation on the server.

See the [ReadAloud API reference](https://readaloudai.org/developers) for voices and pricing.

## Installation

```bash
pip install livekit-plugins-readaloud
```

## Prerequisites

You'll need an API key from ReadAloud. It can be set as an environment variable: `READALOUD_API_KEY`

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import readaloud

session = AgentSession(
    # stt=..., llm=...,
    tts=readaloud.TTS(sample_rate=24000),  # 24000 | 16000 | 8000
)
```

`engine` is `"live"` (default, lowest latency and price) or `"studio"`.
