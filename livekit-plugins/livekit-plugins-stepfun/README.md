# LiveKit Plugins StepFun

Agent Framework plugin for StepFun (阶跃星辰) StepAudio Realtime speech-to-speech models.

## Installation

```bash
pip install livekit-plugins-stepfun
```

## Quickstart

```python
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import stepfun

session = AgentSession(
    llm=stepfun.realtime.RealtimeModel(
        model="stepaudio-3-realtime-preview",  # or "stepaudio-2.5-realtime"
        voice="linjiajiejie",                 # or "vibrant-youth", "lively-girl"
    ),
    vad=None,
    turn_handling=TurnHandlingOptions(turn_detection="realtime_llm"),
)
```

## Supported Models

- `stepaudio-3-realtime-preview`: StepFun's flagship duplex reasoning audio model with thinking trace support.
- `stepaudio-2.5-realtime`: Production-ready low-latency realtime audio model.
- `step-1o-audio`: Multimodal speech model.

## Endpoints and Voice Mapping

The plugin supports both international (`stepfun.ai`) and domestic (`stepfun.com`) clusters, with automatic bidirectional voice name normalization:

- **International cluster (default)**: `wss://api.stepfun.ai/v1/realtime` (`stepfun.OVERSEAS_BASE_URL`)
- **Domestic cluster**: `wss://api.stepfun.com/v1/realtime` (`stepfun.DOMESTIC_BASE_URL`)

## Tools & Function Calling

The plugin supports both local Python `@function_tool` and StepFun cloud provider tools:

```python
from livekit.agents import Agent, function_tool
from livekit.plugins import stepfun

@function_tool
async def query_db(key: str) -> str:
    """Query custom database"""
    return "result"

agent = Agent(
    instructions="You are a helpful voice assistant.",
    tools=[
        stepfun.tools.WebSearch(top_k=3),  # Native server-side web search
        query_db,                         # Client-side Python function tool
    ],
)
```

## Environment Variables

- `STEPFUN_API_KEY`: Your StepFun API key from [platform.stepfun.com](https://platform.stepfun.com) or [platform.stepfun.ai](https://platform.stepfun.ai).
- `STEPFUN_BASE_URL`: Optional custom WebSocket base URL.
