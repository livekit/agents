# Saidwire plugin for LiveKit Agents

Support for [Saidwire](https://saidwire.com) real-time English speech-to-text.

The plugin streams the user's audio to Saidwire over a WebSocket and emits interim and
final transcripts. When the agent session's VAD or turn detector ends the user's turn, it
flushes the stream; the plugin sends `Finalize`, and the final transcript comes straight
back.

See [saidwire.com/docs/api](https://saidwire.com/docs/api) for the API reference. This
plugin is built and maintained by Saidwire, the company that runs the service.

## Installation

```bash
pip install livekit-plugins-saidwire
```

Or install with the LiveKit Agents extra:

```bash
uv add "livekit-agents[saidwire]"
```

## Pre-requisites

You'll need an API key from Saidwire ([saidwire.com/dev](https://saidwire.com/dev)). It
can be set as an environment variable: `SAIDWIRE_API_KEY`.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import saidwire, silero

session = AgentSession(
    stt=saidwire.STT(),  # reads SAIDWIRE_API_KEY
    vad=silero.VAD.load(),
    llm=...,
    tts=...,
)
```

## Options

```python
stt = saidwire.STT(
    api_key="...",  # default: $SAIDWIRE_API_KEY
    model="saidwire-en-2026-10-30",
    language="en",  # en, en-US, en-GB, ... (English only)
    sample_rate=16000,  # 8000 for telephony audio
    interim_results=True,
    punctuate=True,
    endpointing_ms=None,  # server default; 0 = leave turn-taking to the session
    utterance_end_ms=None,  # e.g. 1000: END_OF_SPEECH after 1 s of silence
    vad_events=False,  # True: START_OF_SPEECH from the server
    early_words=False,  # True: "early_word" events (below)
    warmup_timeout=300.0,  # seconds to wait while the service starts up
)
```

| option | meaning |
|---|---|
| `endpointing_ms` | pause, in ms, that ends an utterance on the server. `None` uses the server default. `0` turns server endpointing off: finals come when the session flushes and at longer pauses. |
| `utterance_end_ms` | the server sends `UtteranceEnd` (mapped to `END_OF_SPEECH`) after this much silence |
| `vad_events` | the server sends `SpeechStarted` (mapped to `START_OF_SPEECH`) |
| `early_words` | emit an `early_word` event for each word as soon as it is committed |
| `warmup_timeout` | if the service answers "starting up" or "busy" (HTTP 503 / 429), wait for `Retry-After` and try again, for up to this many seconds. Audio pushed while waiting is not transcribed. |

`stt.update_options(...)` changes options at runtime; open streams reconnect with them.
`recognize()` sends pre-recorded audio to `POST /v1/listen`.

### Early words

```python
stt = saidwire.STT(early_words=True)


@stt.on("early_word")
def on_early_word(word: dict) -> None:
    # {"word": "cancel", "punctuated_word": "cancel.", "start": 6.84, "end": 7.29, ...}
    # Sent as soon as the word is committed, usually before the final that contains it.
    # Useful for keyword triggers ("stop", "cancel"). The final transcript is authoritative.
    ...
```

## Behavior

* The session's flush at end of turn sends `Finalize`; the answer is a final transcript
  (empty if everything was already final), so the session never waits on a timeout.
* Words an earlier final already delivered are not repeated in later finals or interims
  (for example a word cut by a flush and decoded again); the plugin checks word timings.
* While audio is not flowing, a `KeepAlive` message holds the stream open. Silence would be
  billed as audio; `KeepAlive` is free.
* Dropped connections are retried per `APIConnectOptions`. A bad or revoked key, or used-up
  credit, fails at once with a non-retryable `APIStatusError`.
* `stream.request_id` is the id of the current connection; include it when you contact
  support.

## Protocol

Saidwire's streaming API uses the widely used `/v1/listen` WebSocket protocol, with
`Finalize`, `KeepAlive` and `CloseStream` control messages.

## Support

Email support@saidwire.com.
