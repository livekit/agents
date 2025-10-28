# Talkscriber plugin for LiveKit Agents

Support for [Talkscriber](https://github.com/Talkscriber/ts-client)'s voice AI services in LiveKit Agents.

This plugin provides both Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities using Talkscriber's WebSocket and REST APIs.

## Features

- **Speech-to-Text (STT)**: Real-time streaming transcription with word-level timestamps
- **Text-to-Speech (TTS)**: High-quality voice synthesis with streaming and chunked modes
- Supports multiple languages and voice options
- Low-latency WebSocket-based communication
- Configurable for local or remote Talkscriber servers

## Installation

```bash
pip install livekit-plugins-talkscriber
```

Or install with LiveKit Agents:

```bash
pip install "livekit-agents[talkscriber]"
```

## Pre-requisites

You'll need an API key from Talkscriber. It can be set as an environment variable: `TALKSCRIBER_API_KEY`

## Usage

```python
from livekit.agents import VoiceAssistant
from livekit.plugins import talkscriber

# Create STT instance
stt = talkscriber.STT()

# Create TTS instance
tts = talkscriber.TTS()

# Use in your agent
assistant = VoiceAssistant(
    stt=stt,
    tts=tts,
    # ... other configuration
)
```