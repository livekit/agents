# livekit-plugins-yoob

[Yoob](https://yoob.com) speech and transcription for [LiveKit Agents](https://docs.livekit.io/agents/).

- `yoob.TTS`: Overtone expressive speech. Understands inline tags such as `[laughing]`, `[sigh]` and `[whisper] ... [/whisper]`, plus free-text `instructions`. Supports `synthesize()` (one request, audio streamed back) and `stream()` (text pushed as the LLM writes it, over one socket per segment). 48 kHz mono.
- `yoob.STT`: Undertone transcription. Streaming with interim results and start/end-of-speech events, plus batch `recognize()` of up to 60 s.

## Install

```bash
pip install livekit-plugins-yoob
```

Set `YOOB_API_KEY` (get one at [app.yoob.com](https://app.yoob.com)).

## Use

```python
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import yoob

session = AgentSession(
    stt=yoob.STT(),
    llm=...,  # any LiveKit LLM
    tts=yoob.TTS(voice="w1243"),
    turn_handling=TurnHandlingOptions(turn_detection="stt"),
)
```

A full agent is in [`examples/voice_agent.py`](examples/voice_agent.py).

### `yoob.TTS(...)`

| Argument | Default | |
|---|---|---|
| `voice` | `w1243` | A built-in voice id or one of your cloned voices (`GET /v1/voices`). |
| `quality` | `best` | `fast` gives a quicker first sound. |
| `instructions` | none | Delivery, e.g. `"warm voice, softly spoken"`. |
| `speed` | none | 0.5 to 2.0. |
| `seed` | none | Makes renders repeatable. |
| `api_key`, `base_url`, `http_session` | env | `YOOB_API_KEY`, `YOOB_BASE_URL`. |

`update_options(voice=..., quality=..., instructions=..., speed=...)` changes them mid-session.

### `yoob.STT(...)`

| Argument | Default | |
|---|---|---|
| `model` | `yoob-asr-turbo-v1` | For batch `recognize()`: `yoob-asr-turbo-v1` or `yoob-asr-pro-v1`. |
| `interim_results` | `True` | Emit partial transcripts while the user speaks. |
| `api_key`, `base_url`, `http_session` | env | |

## Tests

The tests follow LiveKit's own plugin tests (`tests/test_tts.py`, `tests/test_stt.py` in livekit/agents) with the same assertions and thresholds, run against the live API:

```bash
pip install -e ".[test]"
YOOB_API_KEY=... pytest -q
```

Last run (2026-11-01, livekit-agents 1.8.10): 10 passed. STT WER on LiveKit's 46.7 s test recording: 4.0% (`yoob-asr-turbo-v1`), 3.4% (`yoob-asr-pro-v1`), 4.0% streamed; `flush()` returns a final in about 0.3 s. Spoken audio is checked with Yoob transcription (WER <= 0.2), and timeouts with a local socket that never answers.
