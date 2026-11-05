# Respeecher plugin for LiveKit Agents

Support for [Respeecher](https://respeecher.com/)'s TTS in LiveKit Agents.

More information is available in the docs for the [Respeecher](https://docs.livekit.io/agents/integrations/tts/respeecher/) integration.

## Installation

```bash
pip install livekit-plugins-respeecher
```

## Pre-requisites

You'll need an API key from Respeecher. It can be set as an environment variable: `RESPEECHER_API_KEY` or passed to the `respeecher.TTS()` constructor.

To get the key, log in to [Respeecher Space](https://space.respeecher.com/).

## Models

Supported public models are English (`/public/tts/en-rt`), Ukrainian (`/public/tts/ua-rt`),
Swedish (`/public/tts/sv-rt`), Brazilian Portuguese (`/public/tts/pt-br-rt`),
German (`/public/tts/de-rt`), and Arabic (`/public/tts/ar-rt`).

English and Ukrainian have default voices. For the other models, select a voice from
`list_voices()` and pass its ID explicitly:

```python
from livekit.plugins import respeecher

voices = await respeecher.list_voices(model="/public/tts/sv-rt")
tts = respeecher.TTS(model="/public/tts/sv-rt", voice_id=voices[0].id)
```
