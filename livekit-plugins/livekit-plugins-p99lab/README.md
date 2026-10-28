# p99lab plugin for LiveKit Agents

Support for streaming speech-to-text with **Audrey**, p99lab's speech-to-text model for phone calls, meetings and
accented speech.

## Installation

```bash
pip install livekit-plugins-p99lab
```

## Pre-requisites

Sign up at [console.p99lab.com](https://console.p99lab.com) and create an API key. Set it as an environment variable:

| Variable | Meaning |
|---|---|
| `P99LAB_API_KEY` | API key from the console (required unless passed as `api_key`) |
| `P99LAB_API_URL` | Optional. API base URL; defaults to the public Audrey API, `https://api.p99lab.com`. The stream URL (`wss://…/v1/stream`) is derived from it |

The public API serves 30 languages; English (`language="en"`) is the default. Turn detection and number formatting are English-only.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import p99lab, silero

session = AgentSession(
    stt=p99lab.STT(language="en", prompt="Acme Dental, Dr. Okafor"),
    vad=silero.VAD.load(),
    llm=...,
    tts=...,
)
```

- Streaming: the session's audio goes to Audrey's `/v1/stream` WebSocket as 16-bit mono PCM in 50 ms chunks.
  Partial transcripts arrive as `INTERIM_TRANSCRIPT` events while the user speaks; each final arrives as a
  `FINAL_TRANSCRIPT` followed by `END_OF_SPEECH`. `flush()` asks the server to finalise the current utterance at once.
- `recognize()` posts a buffer as WAV to `/v1/transcribe` and returns one `FINAL_TRANSCRIPT`.
- Audio at another sample rate is resampled to `sample_rate` (8000, 16000, 24000 or 48000); stereo is mixed down.

| Parameter | Default | Meaning |
|---|---|---|
| `language` | `"en"` | ISO code (`en`, `es`, `fr`, `de`, `ja`, ... 30 languages) or `"auto"` (English vs Spanish); region codes such as `"pt-BR"` are accepted |
| `api_key` | `P99LAB_API_KEY` | API key. Keep it server-side |
| `prompt` | none | Context text, up to 500 characters: names, product terms |
| `endpoint_ms` | server default | Silence after which an utterance is finalised, 200 to 3000 |
| `partial_ms` | server default | How often a partial is sent, in ms of new speech, 250 to 5000 |
| `sample_rate` | `16000` | Rate the audio is sent at |
| `base_url` | `P99LAB_API_URL`, else the public API | HTTP base of the API |
| `stream_url` | derived from `base_url` | WebSocket URL |

`update_options(language=..., prompt=..., endpoint_ms=..., partial_ms=...)` changes settings; open streams
reconnect to apply them.

## Errors

Dropped connections and transient server errors ("server busy", 503) reconnect under the session's retry limits.
Account refusals raise `APIStatusError` without retrying; `body` is a dict with `code`:

- 401: unknown or revoked key (`code == "unknown_key"`).
- 402: the account cannot pay. `body["reason"]` is `"no_plan"`, `"no_credit"` or `"payment_failed"` and
  `body["console_url"]` is the billing page. When credit runs out mid-stream, the final of the turn in progress is
  delivered first.
- 429 with `code == "key_daily_limit"`: the key's daily audio limit, which resets at midnight Pacific time. A 429 for
  too many parallel streams on one key is retried.

The first connection after an idle period can take longer than LiveKit's default connect timeout while capacity
starts; pass `conn_options=APIConnectOptions(timeout=...)` if a first call times out.
