# livekit-plugins-yoob

[Yoob](https://yoob.com) speech and transcription for [LiveKit Agents](https://docs.livekit.io/agents/).

LiveKit docs: [speech-to-text](https://docs.livekit.io/agents/models/stt/yoob/) and [text-to-speech](https://docs.livekit.io/agents/models/tts/yoob/) (live with the next release).

- `yoob.TTS`: Overtone expressive speech. Understands inline tags such as `[laughing]`, `[sigh]` and `[whisper] ... [/whisper]`, plus free-text `instructions`. Supports `synthesize()` (one request, audio streamed back) and `stream()` (text pushed as the LLM writes it, over one socket). 48 kHz mono.
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

Yoob runs in this repository's shared suites (`tests/test_tts.py`, `tests/test_stt.py`) against the live API:

```bash
YOOB_API_KEY=... uv run pytest tests/test_tts.py tests/test_stt.py -k yoob
```
