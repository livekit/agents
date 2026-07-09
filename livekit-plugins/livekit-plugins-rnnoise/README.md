# RNNoise Plugin for LiveKit Agents

OSS, self-hosted noise cancellation for LiveKit Agents (RNNoise) — a free, fully-local alternative
to the Cloud-gated Krisp plugin.

## Features

- **`RNNoise`**: A `FrameProcessor` that runs [RNNoise](https://github.com/xiph/rnnoise) noise
  suppression on the agent's incoming audio, entirely on-device.

## Installation

```bash
pip install livekit-plugins-rnnoise
```

No API key, license, or separate model download is required — the RNNoise model is bundled in the
wheel via [`pyrnnoise`](https://pypi.org/project/pyrnnoise/).

## Quick Start

```python
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, room_io
from livekit.plugins import rnnoise


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(...)

    await session.start(
        agent=Agent(instructions="..."),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=rnnoise.RNNoise(),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
```

**Audio Pipeline:** `Room → RoomIO (with RNNoise) → VAD → STT → LLM`

See `examples/rnnoise_agent_example.py` for a complete, runnable agent.

## Notes

- **Mono only (v1):** `RNNoise` currently supports single-channel audio. A frame with more than one
  channel raises `ValueError`.
- **Latency:** RNNoise operates on fixed 48kHz/10ms frames internally, so incoming audio is resampled
  in and out. Expect roughly 10-70ms of warm-up latency (padded with silence) before denoised audio
  starts flowing, depending on the source sample rate.
- **CPU cost:** small — a resample pass plus RNNoise inference per frame. No GPU required.
- **License:** Apache-2.0.
