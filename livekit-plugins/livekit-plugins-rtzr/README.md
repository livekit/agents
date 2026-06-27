# RTZR plugin for LiveKit Agents

Support for RTZR Streaming STT via WebSocket interface, following the "Streaming STT" guide in the RTZR Developers docs.

- Docs (root): `https://developers.rtzr.ai/docs/en/`
- Docs (Streaming STT): `https://developers.rtzr.ai/docs/en/stt-streaming/`

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

`STT()` options (commonly used):
- `model`: `sommers_ko` (default, Korean), `sommers_ja` (Japanese), `sommers_en` (English), `whisper` (English, Korean, Japanese, and Others, but not recommended)
- `language`: language hint (only used for `whisper`, default `ko`)
- `sample_rate`: `8000` to `48000` (Hz)
- `encoding`: currently `LINEAR16` only in this plugin
- `domain`: `CALL` (default) or `MEETING`
- `epd_time`: endpoint detection time in seconds (default `0.8`)
- `noise_threshold`: noise threshold (default `0.60`)
- `active_threshold`: active speech threshold (default `0.80`)
- `use_itn`: Inverse Text Normalization. normalize numbers/units/English tokens (default `True`)
- `use_disfluency_filter`: filler-word filtering (default `False`)
- `use_profanity_filter`: profanity filtering (default `False`)
- `use_punctuation`: punctuation output (default `False`)
- `keywords`: keyword boosting list (`sommers_ko` only)

Example with additional decoder parameters:

```python
stt = rtzr.STT(
    model="sommers_ko",
    sample_rate=8000,
    encoding="LINEAR16",
    domain="CALL",
    epd_time=0.8,
    noise_threshold=0.60,
    active_threshold=0.80,
    use_itn=True,
    use_disfluency_filter=False,
    use_profanity_filter=False,
    use_punctuation=False,
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
- Score 0.0 does not boost the keyword, but its presence in the list can still affect other keywords' boosting. Remove keywords that don't need boosting instead of setting score to 0.0.

Notes:
- The WebSocket streaming endpoint accepts raw PCM frames when `encoding=LINEAR16`.
- The plugin relies on the server-side endpointing (EPD). You do not need to send finalize messages.
- When the pipeline closes the stream, the plugin sends `EOS` to end the session.
