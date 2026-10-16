# Bland plugin for LiveKit Agents

Support for voice synthesis with the [Bland](https://bland.ai/) API.

Voice agents can stream text through Bland's realtime WebSocket. Use `synthesize()` for complete strings over HTTP, or pass `streaming=False` to use HTTP for everything. See the [Bland realtime TTS reference](https://docs.bland.ai/api-v2/post/tts-ws).

## Installation

```bash
pip install livekit-plugins-bland
```

## Prerequisites

You'll need an API key from Bland. It can be set as an environment variable: `BLAND_API_KEY`
