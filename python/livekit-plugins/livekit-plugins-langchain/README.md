# LangChain plugin for LiveKit Agents

This plugin integrates capabilites from LangChain within LiveKit Agents

## Installation

```bash
pip install livekit-plugins-langchain
```

## Usage

### Using LangGraph workflows

You can bring over any existing workflow in LangGraph as an Agents LLM with `langchain.LLMAdapter`. For example:

```python
from langgraph.graph import StateGraph
from livekit.agents import Agent, AgentServer, JobContext, VoiceSession
from livekit.plugins import langchain

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    graph = StateGraph(...).compile()

    session = VoiceSession(
        vad=...,
        stt=...,
        tts=...,
    )

    await session.start(
        agent=Agent(instructions="", llm=langchain.LLMAdapter(graph)),
    )
    ...
```
