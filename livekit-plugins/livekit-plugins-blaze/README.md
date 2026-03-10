# livekit-plugins-blaze

Agent Framework plugin for [Blaze AI](https://blaze.ai) services:

- **STT** (Speech-to-Text) via `POST /v1/stt/transcribe`
- **TTS** (Text-to-Speech) via `POST /v1/tts/realtime` (streaming PCM)
- **LLM** (Conversational AI) via `POST /voicebot/{bot_id}/chat-conversion?stream=true` (SSE)

## Installation

```bash
pip install livekit-plugins-blaze
```

## Usage

```python
from livekit.plugins import blaze

stt = blaze.STT(language="vi")
tts = blaze.TTS(speaker_id="speaker-1")
llm = blaze.LLM(bot_id="my-chatbot")
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `BLAZE_API_URL` | Base URL for Blaze API gateway | `https://api.blaze.vn` |
| `BLAZE_API_TOKEN` | Bearer token for authentication | |
