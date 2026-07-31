# BytePlus TTS plugin for LiveKit Agents

Support for [BytePlus Voice](https://docs.byteplus.com/en/docs/byteplusvoice/docs-overview)
text-to-speech services in LiveKit Agents.

See the
[BytePlus unidirectional TTS documentation](https://docs.byteplus.com/en/docs/byteplusvoice/unidirectional_tts_http),
[BytePlus Voice List](https://docs.byteplus.com/en/docs/byteplusvoice/voicelist), and
[LiveKit TTS documentation](https://docs.livekit.io/agents/models/tts/) for more information.

## Installation

```bash
pip install livekit-plugins-byteplus
```

## Pre-requisites

Create a BytePlus API key and expose it as an environment variable:

```bash
export BYTEPLUS_API_KEY="your-api-key"
```

The API key must have access to the selected resource ID and voice.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import byteplus

session = AgentSession(
    tts=byteplus.TTS(
        model="seed-tts-2.0",
        voice="zh_female_vv_uranus_bigtts",
        audio_format="pcm",
        sample_rate=24000,
    )
)
```

The plugin supports LiveKit's `synthesize()` and streaming `stream()` interfaces.
Available voices and resource IDs depend on the BytePlus account configuration.
