# Pydantic AI plugin for LiveKit Agents

This plugin runs a [Pydantic AI](https://ai.pydantic.dev) agent as the LLM of a LiveKit
`AgentSession`. Tools, retries, structured output and dependencies stay in Pydantic AI, while
LiveKit handles speech, turn-taking and interruptions.

## Installation

```bash
pip install livekit-plugins-pydantic-ai
```

## Usage

Wrap any Pydantic AI `Agent` with `pydantic_ai.LLMAdapter` and pass it as the `llm` of a LiveKit
`Agent`:

```python
from pydantic_ai import Agent as PydanticAgent
from livekit.agents import Agent, AgentSession, JobContext
from livekit.plugins import pydantic_ai

support_agent = PydanticAgent("openai:gpt-5.2", instructions="You help customers with orders.")


@support_agent.tool_plain
def lookup_order(order_id: str) -> str:
    return f"Order {order_id} ships tomorrow."


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=...,
        stt=...,
        tts=...,
    )

    await session.start(
        agent=Agent(
            instructions="Keep answers short, this is a voice call.",
            llm=pydantic_ai.LLMAdapter(support_agent),
        ),
        room=ctx.room,
    )
```

### How it works

- On every turn the adapter converts the LiveKit `ChatContext` into Pydantic AI
  `ModelMessage` history and runs the agent with `run_stream_events()`. Because the history is
  rebuilt from LiveKit's chat context each time, interruptions and other edits LiveKit makes to
  the conversation are reflected automatically.
- Text output is streamed to LiveKit as it is generated. Token usage from the run is reported
  as LiveKit LLM metrics.
- Tool calls are executed by Pydantic AI. The `tools` registered on the LiveKit `Agent` are not
  sent to the model; define them on the Pydantic AI agent instead.
- The LiveKit `Agent` instructions are forwarded as a system prompt. Pydantic AI
  `system_prompt`s are only added by Pydantic AI when the history is empty, so prefer
  `instructions=` on the Pydantic AI agent for prompts that must apply on every turn.

### Dependencies and model settings

`LLMAdapter` accepts the `deps` and `model_settings` passed through to every run:

```python
pydantic_ai.LLMAdapter(support_agent, deps=MyDeps(db=db), model_settings={"temperature": 0.2})
```
