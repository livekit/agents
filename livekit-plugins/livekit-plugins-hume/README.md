# Hume AI TTS plugin for LiveKit Agents

Support for text-to-speech with [Hume](https://www.hume.ai/).

See [https://docs.livekit.io/agents/integrations/tts/hume/](https://docs.livekit.io/agents/integrations/tts/hume/) for more information.

## Installation

```bash
pip install livekit-plugins-hume
```

You will need an API Key from Hume, it can be set as an environment variable: `HUME_API_KEY`. You can get it from [here](https://platform.hume.ai/settings/keys)

The plugin advertises Hume's websocket input streaming endpoint by default when used with LiveKit's streaming TTS interface. To preserve the previous sentence-adapter behavior in LiveKit's default TTS node, initialize `TTS` with `streaming=False`.
