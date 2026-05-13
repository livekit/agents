# Perplexity plugin for LiveKit Agents

Support for [Perplexity](https://www.perplexity.ai/) LLMs via the OpenAI-compatible
chat completions endpoint at `https://api.perplexity.ai`.

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

llm = perplexity.LLM(
    model="sonar-pro",
    # api_key picked up from PERPLEXITY_API_KEY if omitted
)
```

The plugin reuses the OpenAI plugin's chat completions transport with
`base_url="https://api.perplexity.ai"` and forwards an `X-Pplx-Integration`
attribution header on every outgoing request.
