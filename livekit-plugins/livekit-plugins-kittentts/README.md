# LiveKit Plugins KittenTTS

Support for local KittenTTS synthesis in LiveKit Agents.

## Installation

```bash
pip install livekit-plugins-kittentts
```

## Usage

```python
from livekit.plugins import kittentts

tts = kittentts.TTS(
    model="KittenML/kitten-tts-nano-0.8",
    voice="expr-voice-5-m",
)
```
