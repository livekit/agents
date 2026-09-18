# Perplexity plugin for LiveKit Agents

Support for [Perplexity](https://www.perplexity.ai/) models through the Agent API.

See [https://docs.livekit.io/agents/models/llm/perplexity/](https://docs.livekit.io/agents/models/llm/perplexity/) for more information.

## Installation

```bash
pip install livekit-plugins-perplexity
```

## Pre-requisites

You'll need an API key from Perplexity. It can be passed directly or set as the
`PERPLEXITY_API_KEY` environment variable.

## Usage

```python
from livekit.plugins import perplexity

llm = perplexity.responses.LLM(
    model="perplexity/sonar",
    # api_key picked up from PERPLEXITY_API_KEY if omitted
)
```

The Responses LLM uses `base_url="https://api.perplexity.ai/v1"`, disables
websocket transport, and sends an `X-Pplx-Integration` attribution header on
its OpenAI-compatible client.

## Migrating from Chat Completions

The `perplexity.LLM` class and `openai.LLM.with_perplexity()` use Sonar Chat
Completions and are deprecated. Replace either legacy path with the Responses
LLM:

```python
from livekit.plugins import perplexity

llm = perplexity.responses.LLM(
    model="perplexity/sonar",
    # api_key picked up from PERPLEXITY_API_KEY if omitted
)
```

See Perplexity's [migration guide](https://docs.perplexity.ai/docs/agent-api/migrate-from-sonar/overview)
for request and model changes when moving from Sonar to the Agent API.
