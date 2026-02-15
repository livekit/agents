# Pocket TTS plugin for LiveKit Agents

Support for local voice synthesis with [Pocket TTS](https://github.com/kyutai-labs/pocket-tts).

## Installation

```bash
pip install livekit-plugins-pocket
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import pocket

session = AgentSession(
    tts=pocket.TTS(voice="alba"),
)
```

## Notes

- Pocket TTS output is emitted as mono PCM at native `24000` Hz.
- If `sample_rate` is passed with a different value, the plugin keeps `24000` Hz.
