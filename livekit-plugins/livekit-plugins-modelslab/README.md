# ModelsLab plugin for LiveKit Agents

Support for voice synthesis with [ModelsLab](https://modelslab.com/) in LiveKit Agents.

## Installation

```bash
pip install livekit-plugins-modelslab
```

## Pre-requisites

You'll need an API key from ModelsLab. It can be set as an environment variable: `MODELSLAB_API_KEY`. You can get it from [ModelsLab Dashboard](https://modelslab.com/dashboard/api-keys).

## Usage

```python
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import modelslab


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        tts=modelslab.TTS(
            voice_id="default",
            output_format="mp3",
        ),
    )
    agent = Agent(instructions="You are a helpful voice assistant.")
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voice_id` | `str` | `"default"` | Voice ID to use. Browse voices at [modelslab.com/voice-cloning](https://modelslab.com/voice-cloning). |
| `output_format` | `str` | `"mp3"` | Audio format: `"mp3"` or `"wav"`. |
| `sample_rate` | `int` | `24000` | Output sample rate in Hz. |
| `api_key` | `str` | env `MODELSLAB_API_KEY` | ModelsLab API key. |

## Links

- [ModelsLab API docs](https://docs.modelslab.com/text-to-speech/community-tts/generate)
- [Available voices](https://modelslab.com/voice-cloning)
- [Get API key](https://modelslab.com/dashboard/api-keys)
