# Gradium plugin for LiveKit Agents

Support for [Gradium](https://gradium.ai/)-hosted models in LiveKit Agents.

More information is available in the docs for the
[TTS](https://docs.livekit.io/agents/integrations/tts/gradium/) and [STT](https://docs.livekit.io/agents/integrations/stt/gradium/) integrations.

## Installation

```bash
pip install livekit-plugins-gradium
```

## Pre-requisites

You'll need an API key from Gradium. It can be set as an environment variable: `GRADIUM_API_KEY`.

You also need to deploy a model to Gradium and will need your model endpoint to configure the plugin.
