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
from livekit.agents import Agent, AgentSession, JobContext
from livekit.plugins import langchain

...

def entrypoint(ctx: JobContext):
    graph = StateGraph(...).build()

    session = AgentSession(
        vad=...,
        stt=...,
        tts=...,
    )

    await session.start(
        agent=Agent(llm=langchain.LLMAdapter(graph)),
    )
    ...
```
