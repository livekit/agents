# ElevenLabs plugin for LiveKit Agents

Support for voice synthesis with [ElevenLabs](https://elevenlabs.io/).

See [https://docs.livekit.io/agents/integrations/tts/elevenlabs/](https://docs.livekit.io/agents/integrations/tts/elevenlabs/) for more information.

## Installation

```bash
pip install livekit-plugins-elevenlabs
```

## Pre-requisites

You'll need an API key from ElevenLabs. It can be set as an environment variable: `ELEVEN_API_KEY`

Note: this plugin reads `ELEVEN_API_KEY`, while the official ElevenLabs Python SDK reads
`ELEVENLABS_API_KEY`. If you use both in the same project (for example, the plugin in your
agent and the SDK in a separate script), set both variables to the same key.
