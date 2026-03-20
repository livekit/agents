# MiniMax plugin for LiveKit Agents

Support for [MiniMax](https://www.minimaxi.com/)'s TTS and LLM services in LiveKit Agents.

See the [MiniMax TTS integration docs](https://docs.livekit.io/agents/models/tts/minimax/) for more information.

## Installation

```bash
pip install livekit-plugins-minimax-ai
```

## Pre-requisites

You'll need an API key from MiniMax. It can be set as an environment variable: `MINIMAX_API_KEY`

## Usage

### LLM

```python
from livekit.plugins import minimax

llm = minimax.LLM(model="MiniMax-M2.7")
```

Available models: `MiniMax-M2.7`, `MiniMax-M2.5`, `MiniMax-M2.5-highspeed`

### TTS

```python
from livekit.plugins import minimax

tts = minimax.TTS()
```
