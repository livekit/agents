# LiveKit Plugins Saidwire

**Saidwire** real-time English speech-to-text for [LiveKit Agents](https://docs.livekit.io/agents/).

The plugin streams the user's audio to Saidwire over a WebSocket and emits interim and final transcripts. When the agent session's VAD or turn detector ends the user's turn, it flushes the stream; the plugin sends `Finalize` and the final transcript comes straight back. See [saidwire.com](https://saidwire.com) and the [API docs](https://saidwire.com/developers).

LiveKit docs: [speech-to-text](https://docs.livekit.io/agents/models/stt/saidwire/) (live with the next release).

## Installation

```bash
pip install livekit-plugins-saidwire
```

Set `SAIDWIRE_API_KEY`, or pass `api_key=` to `STT`. Keys come from [saidwire.com/developers](https://saidwire.com/developers).

## Usage

### In an agent

```python
from livekit.agents import AgentSession
from livekit.plugins import openai, saidwire, silero

session = AgentSession(
    stt=saidwire.STT(),  # SAIDWIRE_API_KEY from env
    llm=openai.LLM(),
    tts=openai.TTS(),
    vad=silero.VAD.load(),
)
```

### Speech-to-text

`stream()` sends 16-bit PCM over `wss://api.saidwire.com/v1/listen`: interim transcripts while the user speaks, a final transcript at the end of each utterance or flush, with word timings. `recognize()` transcribes one clip with `POST /v1/listen`.

```python
stt = saidwire.STT()
event = await stt.recognize(frames)  # a list of rtc.AudioFrame
print(event.alternatives[0].text)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | env `SAIDWIRE_API_KEY` | Saidwire API key |
| `model` | `str` | `"saidwire-en-2026-10-30"` | Model id |
| `language` | `str` | `"en"` | `en`, `en-US`, `en-GB`, ... (English only) |
| `sample_rate` | `int` | `16000` | Rate audio is sent at; use `8000` for telephony audio |
| `interim_results` | `bool` | `True` | Emit interim transcripts |
| `punctuate` | `bool` | `True` | Punctuation and capitals |
| `endpointing_ms` | `int \| None` | server default | Pause, in ms, that ends an utterance on the server. `0` leaves turn-taking to the session |
| `endpointing` | `str \| None` | `None` | `"smart"`: the server ends the turn from the words and the pause (overrides `endpointing_ms`) |
| `utterance_end_ms` | `int \| None` | `None` | The server sends `UtteranceEnd` (mapped to `END_OF_SPEECH`) after this much silence |
| `vad_events` | `bool` | `False` | The server sends `SpeechStarted` (mapped to `START_OF_SPEECH`) |
| `keyterms` | `list[str] \| None` | `None` | Names and product terms, spelled this way in final transcripts |
| `early_words` | `bool` | `False` | Emit an `early_word` event on the STT for each word as soon as it is committed |
| `send_frame_ms` | `int` | `100` | Audio is sent in frames of this length (20-200 ms); held audio is sent on flush |
| `warmup_timeout` | `float` | `300.0` | Seconds to wait and retry while the service answers HTTP 503 / 429 |
| `base_url` | `str` | `"wss://api.saidwire.com/v1/listen"` | API endpoint |

`stt.update_options(...)` changes options at runtime; open streams reconnect with them.

### Behavior

- The session's flush at end of turn sends `Finalize`; the answer is a final transcript (empty if everything was already final), so the session never waits on a timeout.
- While audio is not flowing, a `KeepAlive` message holds the stream open.
- Dropped connections are retried per `APIConnectOptions`. A bad or revoked key, or used-up credit, fails at once with a non-retryable `APIStatusError`.
- `stream.request_id` is the id of the current connection; include it when you contact support@saidwire.com.

## License

Apache 2.0
