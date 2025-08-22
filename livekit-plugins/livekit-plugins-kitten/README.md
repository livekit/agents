# Kitten plugin for LiveKit Agents

Support for voice synthesis with [KittenTTS](https://github.com/KittenML/KittenTTS), a local ONNX-based text-to-speech engine.

## Installation

```bash
pip install livekit-plugins-kitten
```

## Usage

Before using the plugin, you need to download the model files:

```bash
python myagent.py download-files
```

Then you can use the TTS in your agent:

```python
from livekit.plugins import kitten

tts = kitten.TTS(
    model_name="KittenML/kitten-tts-nano-0.2",
    voice="expr-voice-5-m",
    speed=1.0
)
```
