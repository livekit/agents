# ElevenLabs plugin for LiveKit Agents

Support for voice synthesis with [ElevenLabs](https://elevenlabs.io/).

See [https://docs.livekit.io/agents/integrations/tts/elevenlabs/](https://docs.livekit.io/agents/integrations/tts/elevenlabs/) for more information.

## Installation

```bash
pip install livekit-plugins-elevenlabs
```

## Pre-requisites

You'll need an API key from ElevenLabs. It can be set as an environment variable: `ELEVEN_API_KEY`

## Supported Models

All ElevenLabs TTS models are supported, including:
- `eleven_v3` - Most expressive model with emotion and delivery control
- `eleven_turbo_v2_5` - Fast, high-quality multilingual model (default)
- `eleven_flash_v2_5`, `eleven_flash_v2` - Ultra-fast models
- And more...

**Note:** The `eleven_v3` model uses HTTP streaming instead of WebSocket for compatibility, as it doesn't support WebSocket connections. The plugin automatically handles this difference. Aligned transcripts are not yet supported for `eleven_v3`.
