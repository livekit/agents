# livekit-plugins-blaze

Agent Framework plugin for [Blaze AI](https://blaze.vn) services:

- **STT** (Speech-to-Text) via `POST /v1/stt/transcribe` (batch, default model `v2.0`) and WS `/v1/stt/realtime` (streaming, default model `stt-stream-1.5`)
- **TTS** (Text-to-Speech) via `WS /v1/tts/realtime` (default model `2.0-realtime`; one-shot `synthesize()` and streaming `stream()` both use WebSocket — there is no HTTP POST for realtime TTS)
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
| `BLAZE_TTS_TIMEOUT` | TTS idle timeout per WebSocket recv (seconds) | `60` |
| `BLAZE_TTS_STREAM_TIMEOUT` | Max duration for one streaming TTS turn (seconds) | `300` |
