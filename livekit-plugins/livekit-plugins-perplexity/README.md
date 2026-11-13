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

## Agent API usage

Perplexity's Agent API is compatible with OpenAI's Responses API and is
available through the `perplexity.responses` submodule.

```python
from livekit.plugins import perplexity

llm = perplexity.responses.LLM(
    model="perplexity/sonar",
    # api_key picked up from PERPLEXITY_API_KEY if omitted
)
```

The Responses LLM uses `base_url="https://api.perplexity.ai/v1"`, disables
websocket transport, and sends the same `X-Pplx-Integration` attribution header
on its OpenAI-compatible client.

## Decisions API usage

`perplexity.DecisionModel` implements `llm.DecisionModel` with Perplexity's
[Decisions API](https://docs.perplexity.ai/docs/decisions/quickstart). This example
gates each reply in `Agent.on_user_turn_completed` with one short attempt, and fails
open so a decision outage never blocks the conversation:

```python
from livekit.agents import APIConnectOptions, APIError, llm
from livekit.plugins import perplexity

decider = perplexity.DecisionModel()  # pplx-decider-flash-v1.3

async def on_user_turn_completed(self, turn_ctx, new_message):
    conversation = turn_ctx.copy()
    conversation.items.append(new_message)  # not in turn_ctx yet
    try:
        response = await decider.evaluate(
            input=conversation,
            decisions={"handoff": llm.Probability("The caller asks for a human.")},
            conn_options=APIConnectOptions(timeout=1.5, max_retry=0),
        )
    except APIError:
        return
    handoff = response.results.get("handoff")  # failed questions land in response.errors
    if handoff is not None and handoff.value > 0.8:
        ...
```

Outside a job, wrap calls in `async with utils.http_context.open():` or pass
`http_session=`. For [Decider for Mac](https://docs.perplexity.ai/docs/decisions/decider-for-mac),
pass `base_url="http://127.0.0.1:8080/v1"`; it needs no API key.
