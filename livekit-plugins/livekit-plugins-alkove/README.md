# Alkove plugin for LiveKit Agents

Self-hosted voice isolation and VAD for LiveKit voice agents, with [Alkove](https://alkove.ai)'s on-device models.

- **`alkove.Focus()`**: a `FrameProcessor` for `noise_cancellation` that keeps the user's voice and removes the other voices and the noise.
- **`alkove.VAD.load()`**: Cue as the session's VAD.

Both run on the agent's CPU, with no API key and no LiveKit Cloud requirement. Plugin and models are Apache-2.0.

## Installation

```bash
pip install livekit-plugins-alkove
python my_agent.py download-files     # Focus and Cue into the cache, checked by their sha256
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
| `voiceprint` | `alkove.Voiceprint` or embedding | `None` | Keep the voice of this print instead of the closest voice |
| `strength` | float | 1.0 | How much of Focus goes out, from 0 to 1, the rest being the raw microphone |
| `normalize` | bool | `False` | Bring the voice to a steady level |
| `models` | `alkove.Models` | the cache | Models to use instead of those `download-files` fetched |

It works at the session's sample rate and gives each frame back at the size it came in, about 37 ms behind. `set_voiceprint()` and `set_strength()` change it during the session. In `console` mode, LiveKit reads the microphone without `RoomIO`, so Focus is not applied there: test it in a room (`dev`).

To keep one person's voice rather than the closest one, make their print with Timbre from 1.5 s of their voice or more, at 16 kHz:

```python
models = alkove.Models.load(["focus", "cue", "timbre"])
me = alkove.Timbre(models).embed(pcm)
focus = alkove.Focus(models=models, voiceprint=me)
```

### `alkove.VAD.load()`

It gives a speech probability every 30 ms. `speech=alkove.SpeechParams(confidence=0.5, exit_confidence=0.1, start_secs=0.2, stop_secs=0.6)` sets when speech starts and ends; `stop_secs` is the turn detector's `min_silence_duration`. The end of speech carries its audio from `prefix_padding_duration` (0.5 s) before its start, `max_buffered_speech` (60 s) at most. Behind Focus, it hears only the user.

The models are loaded once per process and shared by every session. Pass `models=alkove.Models.load(["focus", "cue"], dir=...)` to read them from your own folder.

## Limits

Focus is tuned for a close microphone. Telephony audio at 8 kHz has not been measured yet.
