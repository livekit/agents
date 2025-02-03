# LiveKit Plugins OpenAI

Agent Framework plugin for services from OpenAI. Currently supports STT, TTS, and Dalle 3.

## Installation

```bash
pip install livekit-plugins-openai
```

## Pre-requisites

You'll need an API key from OpenAI. It can be set as an environment variable: `OPENAI_API_KEY`

## OpenAI Beta Features

### Assistants API

In addition to LLM, STT, and TTS, this package also supports using [OpenAI's Assistants API](https://platform.openai.com/docs/assistants/overview) as a LLM.

The Assistants API is a stateful API that holds the conversation state on the server-side.

The `AssistantLLM` class gives you a LLM-like interface to interact with the Assistant API.

For examples of using Assistants API with VoicePipelineAssistant, see the [openai assistants API example](https://github.com/livekit/agents/blob/main/examples/voice-pipeline-agent/openai_assistant.py)
