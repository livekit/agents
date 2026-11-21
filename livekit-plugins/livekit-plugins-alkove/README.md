# Alkove plugin for LiveKit Agents

Self-hosted voice isolation, VAD and turn detection for LiveKit voice agents, with [Alkove](https://alkove.ai)'s on-device models.

- **`alkove.Focus()`**: a `FrameProcessor` for `noise_cancellation` that keeps the user's voice and removes the other voices and the noise.
- **`alkove.VAD.load()`**: Cue as the session's VAD.
- **`alkove.TurnDetector()`**: Cue's end of turn head as a streaming turn detector.

All run on the agent's CPU, with no API key and no LiveKit Cloud requirement. Plugin and models are Apache-2.0.

## Installation

```bash
pip install livekit-plugins-alkove
python my_agent.py download-files     # Focus, Timbre and Cue into the cache, checked by their sha256
```

## Usage

```python
from livekit.agents import AgentServer, AgentSession, JobContext, JobProcess, room_io
from livekit.plugins import alkove

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = alkove.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(vad=ctx.proc.userdata["vad"], ...)
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(noise_cancellation=alkove.Focus()),
        ),
    )
```

### `alkove.Focus()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voiceprint` | `alkove.Voiceprint` or a 16 kHz recording | `None` | Keep this person's voice instead of the closest voice |
| `strength` | float | 1.0 | How much of Focus goes out, from 0 to 1, the rest being the raw microphone |
| `normalize` | bool | `False` | Bring the voice to a steady level |
| `models` | `alkove.Models` | the cache | Models to use instead of those `download-files` fetched |

It works at the session's sample rate and gives each frame back at the size it came in, about 37 ms behind. `set_voiceprint()` and `set_strength()` change it during the session. In `console` mode, LiveKit reads the microphone without `RoomIO`, so Focus is not applied there: test it in a room (`dev`).

To keep one person's voice rather than the closest one, give their print, made with Timbre from 1.5 s of their voice or more, or a few seconds of them alone at 16 kHz: Focus then keeps that voice even while someone else talks at the same time.

```python
models = alkove.Models.load(["focus", "cue", "timbre"])
me = alkove.Timbre(models).embed(pcm)
focus = alkove.Focus(models=models, voiceprint=me)  # voiceprint=pcm works too
```

### `alkove.VAD.load()`

It gives a speech probability every 30 ms. `speech=alkove.SpeechParams(confidence=0.5, exit_confidence=0.1, start_secs=0.2, stop_secs=0.6)` sets when speech starts and ends; `stop_secs` is the turn detector's `min_silence_duration`. The end of speech carries its audio from `prefix_padding_duration` (0.5 s) before its start, `max_buffered_speech` (60 s) at most. Behind Focus, it hears only the user.

### `alkove.TurnDetector()`

```python
from livekit.agents import AgentSession, TurnHandlingOptions

session = AgentSession(
    vad=alkove.VAD.load(),
    turn_handling=TurnHandlingOptions(
        turn_detection=alkove.TurnDetector(),  # threshold=0.6, min_pause=0.3, max_pause=1.5
        endpointing={"max_delay": 1.5},
        interruption={"mode": "vad", "min_duration": 0.7},
    ),
    ...
)
```

It hears the user's audio, Focus then Cue on the agent's CPU, and reads Cue's end of turn head every 0.1 s of each pause. The answer comes on the first read over `threshold` once the pause reaches `min_pause`, or at `max_pause` with the head's last read. With the VAD mode and `min_duration=0.7` for interruptions, a "mm" or a "yeah" shorter than that lets the agent go on.

The models are loaded once per process and shared by every session. Pass `models=alkove.Models.load(["focus", "cue"], dir=...)` to read them from your own folder, which then holds `alkove-timbre-1.alk` too.

## Limits

Focus keeps one voice: on a microphone shared by several people, it keeps one of them. Without a print, it keeps the closest voice.
