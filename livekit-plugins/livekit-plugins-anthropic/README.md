# Anthropic plugin for LiveKit Agents

Support for the Claude family of LLMs from Anthropic.

See [https://docs.livekit.io/agents/integrations/llm/anthropic/](https://docs.livekit.io/agents/integrations/llm/anthropic/) for more information.

## Installation

```bash
pip install livekit-plugins-anthropic
```

## Pre-requisites

You'll need an API key from Anthropic. It can be set as an environment variable: `ANTHROPIC_API_KEY`

## Extended Thinking

This plugin supports Claude's extended thinking feature, which allows Claude to show its internal reasoning process before delivering the final answer. This is particularly useful for complex mathematical, logical, or analytical tasks.

### Usage

```python
from livekit.plugins import anthropic

# Enable thinking with a dict configuration
llm = anthropic.LLM(
    model="claude-3-7-sonnet-20250219",
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    }
)

# Or use the TypedDict style
thinking_config: anthropic.ThinkingConfig = {
    "type": "enabled",
    "budget_tokens": 5000
}

llm = anthropic.LLM(
    model="claude-3-7-sonnet-20250219",
    thinking=thinking_config
)
```

### Supported Models

Extended thinking is supported in the following models:

- `claude-opus-4-20250514`
- `claude-sonnet-4-20250514`
- `claude-3-7-sonnet-20250219`

### Configuration

- **type**: Must be set to `"enabled"` to activate thinking
- **budget_tokens**: The maximum number of tokens Claude can use for its internal reasoning process

For more information about extended thinking, see the [Anthropic documentation](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking).
