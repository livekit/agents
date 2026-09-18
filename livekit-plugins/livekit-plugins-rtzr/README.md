# RTZR plugin for LiveKit Agents

Support for RTZR Streaming STT via WebSocket interface, following the "Streaming STT" guide in the RTZR Developers docs.

- Docs: `https://developers.rtzr.ai/docs/en/`
- Docs: `https://developers.rtzr.ai/docs/en/stt-streaming/`

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

Optional endpoint overrides:

```bash
RTZR_API_BASE=https://openapi.vito.ai
RTZR_WEBSOCKET_URL=wss://openapi.vito.ai
```

Without `RTZR_WEBSOCKET_URL`, the WebSocket base is derived from `RTZR_API_BASE`.
Use credentials issued for the selected environment.

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

Common `STT()` options:

- `model`: `sommers_ko` (default), `sommers_ja`, or `whisper`
- `language`: language hint for `whisper` (default `ko`)
- `sample_rate`: 8000–48000 Hz
- `encoding`: `LINEAR16`
- `domain`: `CALL` (default) or `MEETING`
- `epd_time`: server endpoint detection timeout (default `0.8`)
- `use_itn`, `use_disfluency_filter`, `use_profanity_filter`, `use_punctuation`

Keyword boosting (`sommers_ko`, or `whisper` with `language="ko"`):

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
- A score of `0.0` does not boost a keyword; remove unused keywords instead.

Notes:
- The WebSocket streaming endpoint accepts raw PCM frames when `encoding=LINEAR16`.
- `SpeechStream.flush()` sends `{"type":"Finalize"}` after pending audio. This lets an
  external VAD request a final result without waiting for `epd_time`, while
  keeping the WebSocket open for the next utterance.
- When the pipeline closes the stream, the plugin sends `EOS` to end the session.

## Stream lifecycle

The default `AgentSession` STT node does not call `flush()` on VAD boundaries.
For external VAD integration, see the [RTZR voice agent guide](https://developers.rtzr.ai/docs/en/stt-streaming/#ai-voice-agent).

- `flush()` marks an utterance boundary. It does not wait for a server acknowledgement;
  empty utterances and repeated Finalize commands produce no transcript.
- `end_input()` finishes the input, sends EOS, and drains the final responses.
- `aclose()` cancels the stream immediately. Use `end_input()` and consume the stream
  to completion when final transcripts must be retained.
- A connection with no audio input for 25 seconds is closed and reopened lazily on
  new audio. Finalize itself keeps the connection open.
- Audio chunks use the shared progressive buffer (20 ms initially, growing to 200 ms).
- Connection retries replay up to 60 seconds of an unfinalized utterance and its boundary.
  A later connection failure is terminal rather than replaying a truncated longer utterance;
  final-response failures after input exhaustion are also terminal.
- Recognition usage counts successfully sent PCM once. Failed connections are
  reported through LiveKit's error/retry path without double-counting replayed audio.
