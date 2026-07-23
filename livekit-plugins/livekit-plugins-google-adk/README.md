# LiveKit Plugins Google ADK

Agent plugin for [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/) integration with [LiveKit Agents](https://github.com/livekit/agents).

This plugin wraps a Google ADK agent as a LiveKit `llm.LLM`, enabling ADK's multi-agent orchestration, tool calling, and session management to be used within a LiveKit voice pipeline.

## Installation

```bash
pip install livekit-plugins-google-adk
```

## Usage

```python
from google.adk.agents import LlmAgent
from livekit.plugins.google_adk import LLMAdapter

adk_agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful voice assistant.",
    tools=[...],
)

llm = LLMAdapter(agent=adk_agent)
```

See `examples/voice_agents/google_adk_agent.py` for a full working example.
