# VLM Run plugin for LiveKit Agents

Vision-language models from the [VLM Run gateway](https://docs.vlm.run/gateway/introduction) as an
LLM for LiveKit agents, through the gateway's OpenAI-compatible chat completions API. Images in the
chat context (`ImageContent`, including `rtc.VideoFrame` camera frames) are sent to the model.

`vlmrun.DecisionModel` answers typed questions (probability, choice, score) through the gateway's
[System One](https://docs.vlm.run/gateway/system-one) API.

## Installation

```bash
pip install livekit-plugins-vlmrun
```

## Pre-requisites

Set `VLMRUN_API_KEY` (from [app.vlm.run](https://app.vlm.run)).
Set `VLMRUN_GATEWAY_BASE_URL` (or `VLMRUN_GATEWAY_URL`) to use another gateway.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import vlmrun

session = AgentSession(llm=vlmrun.LLM(model="qwen/qwen3.8-27b"), stt=..., tts=...)
```

Tool calling works on `qwen/qwen3.8-27b` and the provider models (`google/gemini-*`,
`moonshotai/kimi-k3`, `minimax/minimax-m3`, `meta/muse-*`); the provider models need a paid
account. Sampling settings are sent only when set, because the gateway rejects settings a model
doesn't support.

## Seeing the user's camera

`AgentSession` forwards live video only to realtime models, so `RoomOptions(video_input=True)` alone
does not reach this LLM. Keep the latest frame from the user's video track and add it to the user's
message before each reply. Overriding `llm_node` covers both spoken and typed turns:

```python
import asyncio

from livekit import rtc
from livekit.agents import Agent, ModelSettings, llm


class VisionAgent(Agent):
    def __init__(self, room: rtc.Room) -> None:
        super().__init__(instructions="You can see the user's camera.")
        self._latest: rtc.VideoFrame | None = None
        room.on("track_subscribed", self._on_track)

    def _on_track(self, track: rtc.Track, *_) -> None:
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(self._read(rtc.VideoStream(track, capacity=1)))

    async def _read(self, stream: rtc.VideoStream) -> None:
        async for event in stream:
            self._latest = event.frame

    async def llm_node(self, chat_ctx: llm.ChatContext, tools, model_settings: ModelSettings):
        if self._latest is not None:
            for item in reversed(chat_ctx.items):
                if item.type == "message" and item.role == "user":
                    item.content.append(
                        llm.ImageContent(image=self._latest, inference_width=640, inference_height=480)
                    )
                    break
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk
```

Set `inference_width` and `inference_height` to keep frames small; each frame adds image tokens to
the request.

## Decisions

```python
from livekit.agents import llm, utils
from livekit.plugins import vlmrun

async with utils.http_context.open():
    model = vlmrun.DecisionModel()  # qwen/qwen3.8-27b
    context = llm.ChatContext.empty()
    context.add_message(role="user", content="Please let me speak to a person.")
    response = await model.evaluate(
        input=context,
        decisions={
            "handoff": llm.Probability("The caller requests a human."),
            "intent": llm.Choice(
                "What does the caller want?",
                options={"booking": "Make a reservation", "other": "Anything else"},
            ),
            "frustration": llm.Score(
                "How frustrated is the caller?", levels=["Calm", "Annoyed", "Angry"]
            ),
        },
    )
    for name, result in response.results.items():
        print(name, result.value)
```

`input` also accepts a string, which is sent as-is. Each requested decision appears in either
`response.results` or `response.errors`. Request failures and malformed responses raise.
