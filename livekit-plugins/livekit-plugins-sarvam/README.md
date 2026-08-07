# Sarvam.ai Plugin for LiveKit Agents

Support for [Sarvam.ai](https://sarvam.ai)'s Indian-language voice AI services in LiveKit Agents.

## Features

- **Speech-to-Text (STT)**: Convert audio to text using Sarvam's "Saarika" models. See the [STT docs](https://docs.livekit.io/agents/integrations/stt/sarvam/) for more information.
- **Text-to-Speech (TTS)**: Convert text to audio using Sarvam's "Bulbul" models. See the [TTS docs](https://docs.livekit.io/agents/integrations/tts/sarvam/) for more information.
- **LLM (Chat Completions)**: OpenAI-compatible chat-completions support for `sarvam-30b`, `sarvam-30b-16k`, `sarvam-105b` and `sarvam-105b-32k` including tool calling.

## Installation 

```bash
pip install livekit-plugins-sarvam
```

## Pre-requisites

You'll need an API key from Sarvam.ai. It can be set as an environment variable: `SARVAM_API_KEY`

## LLM Usage

```python
from livekit.plugins import sarvam

llm = sarvam.LLM(
    model="sarvam-30b",  # or `sarvam-30b-16k`, `sarvam-105b`, `sarvam-105b-32k`
)
```

`sarvam-30b`, `sarvam-30b-16k`, `sarvam-105b`, and `sarvam-105b-32k` support tool calling via the standard OpenAI-style `tools` and
`tool_choice` parameters.
