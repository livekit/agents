# RTZR plugin for LiveKit Agents

Support for RTZR Streaming STT via WebSocket interface, following the "Streaming STT" guide in the RTZR Developers docs.

- Docs: `https://developers.rtzr.ai/docs/en/` (see Streaming STT)

## Installation

```bash
pip install livekit-plugins-rtzr
```

## Prerequisites

Obtain `client_id` and `client_secret` from the RTZR Developers Console.

Set credentials as environment variables:

```
RTZR_CLIENT_ID=<your_client_id>
RTZR_CLIENT_SECRET=<your_client_secret>
```

```
# Override base HTTP API URL (used for token issuance)
RTZR_API_BASE=https://openapi.vito.ai

# Override WebSocket URL (used for live streaming)
RTZR_WEBSOCKET_URL=wss://openapi.vito.ai
```

If `RTZR_WEBSOCKET_URL` is not set, the plugin will derive it from `RTZR_API_BASE` by replacing the scheme with `wss://`.

## Usage

Use RTZR STT in an `AgentSession` or as a standalone streaming service.

```python
from livekit.agents import AgentSession
from livekit.plugins import rtzr

# Basic usage with env-based credentials
stt = rtzr.STT()

session = AgentSession(
    stt=stt,
    # ... llm, tts, etc.
)
```

Keyword boosting (Streaming STT only, sommers_ko model only):

```python
stt = rtzr.STT(
    model="sommers_ko",
    keywords=[
        "키워드",
        ("부스팅", 3.5),
        "키위드:-1.0",
    ],
)
```

Rules:
- Use list entries as `keyword` or `keyword:score`, or use `(keyword, score)` tuples.
- Score must be between -5.0 and 5.0, up to 100 keywords, each <= 20 chars.
- Keywords must be written in Korean pronunciation (Hangul and spaces only); non-Korean input will error.

Notes:
- The WebSocket streaming endpoint accepts raw PCM frames when `encoding=LINEAR16`.
- The plugin relies on the server-side endpointing (EPD). You do not need to send finalize messages.
- When the pipeline closes the stream, the plugin sends `EOS` to end the session.
