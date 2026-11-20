# AirCaps AI plugin for LiveKit Agents

Support for streaming speech-to-text with AirCaps Live 1, a low-latency real-time English speech-to-text model for
voice agents from [AirCaps AI](https://aircaps.ai).

See the [AirCaps AI integration docs](https://docs.livekit.io/agents/models/stt/aircaps/) for more information, and the
[AirCaps Live 1 guide](https://aircaps.ai/docs/guides/live-1) for the API.

## Installation

```bash
pip install livekit-plugins-aircaps
```

## Pre-requisites

You'll need an AirCaps AI API key, created in the [AirCaps AI Console](https://console.aircaps.ai). It can be set as the
environment variable `AIRCAPS_API_KEY`.

## Usage

By default the session's own turn handling decides when the user has finished speaking. LiveKit flushes the stream
at the end of each turn (from its VAD or turn detector, or a manual commit), and the final transcript comes back right
away. Without a flush, words stay interim until the stream ends:

```python
from livekit.agents import AgentSession
from livekit.plugins import aircaps

session = AgentSession(
    stt=aircaps.STT(),
    # ... llm, tts, vad, turn_handling, etc.
)
```

To let the model detect the end of each turn instead, turn on `end_of_turn` and use the STT's turn detection. The
model's end of turn has already waited out the user's pause, so set LiveKit's own endpointing delay to 0 (its
default adds 0.5 s to every turn):

```python
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import aircaps

session = AgentSession(
    stt=aircaps.STT(end_of_turn=True, end_of_turn_mode="balanced"),
    turn_handling=TurnHandlingOptions(turn_detection="stt", endpointing={"min_delay": 0}),
    # ... llm, tts, vad, etc.
)
```

### Options

- `model`: defaults to `"aircaps-live-1"`.
- `end_of_turn`: let the model detect the end of each turn (default `False`).
- `end_of_turn_mode`: `"min_latency"`, `"balanced"` (default) or `"max_accuracy"`. A mode sets all the values below;
  any of them given beside it overrides that one.
- `end_of_turn_threshold`: how sure the model must be that the turn ended, 0.2 to 0.9.
- `min_turn_silence_ms`: the shortest pause that can end a turn, 0 to 1000 ms.
- `max_turn_silence_ms`: a pause this long always ends the turn, 500 to 5000 ms, or 0 for no limit.
- `vad_threshold`: how sure the model's voice detector must be that a sound is speech, 0.3 to 0.9.

`stt.update_options(...)` changes the end-of-turn settings, also for streams already open.

Audio is resampled to 16 kHz mono. The model transcribes English and supports streaming recognition only.
