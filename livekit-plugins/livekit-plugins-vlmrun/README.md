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

Requests time out after 70 s by default, because the gateway allows `qwen/qwen3.8-27b` up to 60 s
to its first token. Set `AgentSession(conn_options=...)` to change it.

See the [LiveKit docs](https://docs.livekit.io/agents/models/llm/vlmrun/) for more.

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
