# livekit-plugins-typecast

Typecast TTS plugin for [LiveKit Agents](https://github.com/livekit/agents).

## Installation

```bash
pip install livekit-plugins-typecast
```

## Usage

```python
from livekit.plugins.typecast import TTS

tts = TTS(voice_id="tc_672c5f5ce59fac2a48faeaee")
```

Set `TYPECAST_API_KEY` environment variable with your API key from [typecast.ai/developers](https://typecast.ai/developers).
