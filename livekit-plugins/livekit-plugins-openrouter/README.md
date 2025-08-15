# OpenRouter plugin for LiveKit Agents

Access to 100+ AI models through OpenRouter's intelligent routing platform with support for provider routing, cost optimization, and web search capabilities.

See [https://docs.livekit.io/agents/integrations/llm/](https://docs.livekit.io/agents/integrations/llm/) for more information.

## Installation

```bash
pip install livekit-plugins-openrouter
```

## Pre-requisites

You'll need an API key from OpenRouter. It can be set as an environment variable: `OPENROUTER_API_KEY`

## Usage

```python
from livekit.plugins import openrouter

# Basic usage
llm = openrouter.LLM(model="anthropic/claude-3.5-sonnet")

# With provider preferences for cost optimization
llm = openrouter.LLM(
    model="qwen/qwen3-coder",
    provider_preferences=openrouter.ProviderPreferences(
        sort="price",  # Use cheapest providers
        data_collection="deny"  # Privacy-focused
    )
)

# With web search capabilities
llm = openrouter.LLM(model="openai/gpt-4o:online")  # :online enables web search

# Use in AgentSession
from livekit.agents import AgentSession

session = AgentSession(
    llm=openrouter.LLM(model="anthropic/claude-3.5-sonnet"),
    # ... other components
)
```