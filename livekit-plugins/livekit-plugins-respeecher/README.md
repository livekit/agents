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

## Example

To try out the Respeecher plugin, run the example:

```bash
uv run python examples/other/text-to-speech/respeecher_tts.py start
```

Check [`examples/other/text-to-speech/README.md`](../../examples/other/text-to-speech/README.md) for running details.