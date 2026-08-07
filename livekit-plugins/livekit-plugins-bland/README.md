# Bland plugin for LiveKit Agents

Support for voice synthesis with the [Bland](https://bland.ai/) API.

Voice agents stream text into a single realtime session; `synthesize()` also works over HTTP for complete strings. See [https://docs.bland.ai/api-v2/post/tts-ws](https://docs.bland.ai/api-v2/post/tts-ws) for more information.

## Installation

```bash
pip install livekit-plugins-bland
```

## Pre-requisites

You'll need an API key from Bland. It can be set as an environment variable: `BLAND_API_KEY`
