# VLM Run plugin for LiveKit Agents

Vision-language models from the [VLM Run gateway](https://docs.vlm.run/gateway/introduction) as an
LLM for LiveKit agents, through the gateway's OpenAI-compatible chat completions API. Images in the
chat context, including camera frames, are sent to the model.

## Installation

```bash
pip install livekit-plugins-vlmrun
```

## Pre-requisites

Set `VLMRUN_API_KEY` (from [app.vlm.run](https://app.vlm.run)). Without a key, the chat route uses
the gateway's anonymous tier, at lower rate limits; some models require a paid account.
Set `VLMRUN_GATEWAY_BASE_URL` to use another gateway.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import vlmrun

session = AgentSession(llm=vlmrun.LLM(model="qwen/qwen3.8-27b"), stt=..., tts=...)
```

Tool calling works on `qwen/qwen3.8-27b` and the provider models (`google/gemini-*`,
`moonshotai/kimi-k3`, `minimax/minimax-m3`); the provider models need a paid account. Sampling
settings are sent only when set, because the gateway rejects settings a model doesn't support.
