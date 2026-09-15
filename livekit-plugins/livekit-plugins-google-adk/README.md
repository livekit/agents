# Google ADK plugin for LiveKit Agents

This plugin provides a lightweight adapter for using Google ADK runners and
agents inside LiveKit Agents.

## Installation

```bash
pip install livekit-plugins-google-adk google-adk
```

## Usage

```python
from google.adk import Agent
from livekit.agents import AgentSession
from livekit.plugins import google_adk

adk_agent = Agent(name="assistant")

session = AgentSession(
    llm=google_adk.LLMAdapter(agent=adk_agent),
)
```

## Notes

- This initial adapter is text-focused.
- LiveKit function tools are not forwarded into ADK. Tool execution is expected
  to happen inside the ADK runner/agent.
- The adapter reconstructs an ephemeral ADK session from the LiveKit
  `ChatContext` for each `chat()` call.
