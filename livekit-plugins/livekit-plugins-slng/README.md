# SLNG plugin for LiveKit Agents

Support for [SLNG](https://slng.ai/)'s voice AI gateway in LiveKit Agents, providing access to multiple STT and TTS providers through a unified API.

See [https://docs.slng.ai/](https://docs.slng.ai/) for more information.

## Installation

```bash
pip install livekit-plugins-slng
```

## Pre-requisites

You'll need an API key from SLNG. It can be set as an environment variable: `SLNG_API_KEY`

## Region override

The plugin supports gateway region routing via the `region_override` option on both `STT` and `TTS`.
This maps directly to the gateway's `X-Region-Override` header.
See the available regions at [docs.slng.ai/region-override](https://docs.slng.ai/region-override).

You can pass either a single region:

```python
stt = slng.STT(
    api_key="your-slng-api-key",
    model="deepgram/nova:3",
    region_override="eu-west-1",
)
```

Or multiple preferred regions in priority order:

```python
tts = slng.TTS(
    api_key="your-slng-api-key",
    model="deepgram/aura:2",
    voice="aura-2-thalia-en",
    region_override=["eu-west-1", "us-east-1"],
)
```
