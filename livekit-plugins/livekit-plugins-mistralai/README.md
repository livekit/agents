# MistralAI Plugin for LiveKit Agents

Support for MistralAI services:

- **LLM** — Chat completion with Mistral models
- **STT** — Speech-to-text with Voxtral
- **TTS** — Text-to-speech with Voxtral (supports saved voices and zero-shot voice cloning via `ref_audio`)

See [https://docs.livekit.io/agents/integrations/mistral/](https://docs.livekit.io/agents/integrations/mistral/) for more information.

## Installation

```bash
pip install livekit-plugins-mistralai
```

## Pre-requisites

You'll need an API key from MistralAI. It can be set as an environment variable:

```bash
export MISTRAL_API_KEY=your_api_key_here
```

## Usage

### TTS

```python
from livekit.plugins import mistralai

# Using a built-in voice
tts = mistralai.TTS(voice="en_paul_neutral")

# Using zero-shot voice cloning
import base64
ref_audio_b64 = base64.b64encode(open("sample.mp3", "rb").read()).decode()
tts = mistralai.TTS(ref_audio=ref_audio_b64)
```
