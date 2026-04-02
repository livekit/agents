# Smallest AI plugin for LiveKit Agents

Support for speech synthesis (TTS) and speech recognition (STT) with the [Smallest AI](https://smallest.ai/) API.

## Installation

```bash
pip install livekit-plugins-smallestai
```

## Pre-requisites

You'll need an API key from Smallest AI. It can be set as an environment variable: `SMALLEST_API_KEY`

## Usage

### Speech-to-Text (Pulse STT)

```python
from livekit.plugins import smallestai

# Streaming transcription
stt = smallestai.STT(language="en")

# Automatic language detection across 39 languages
stt = smallestai.STT(language="multi")
```

### Text-to-Speech (Lightning TTS)

```python
from livekit.plugins import smallestai

tts = smallestai.TTS(voice_id="emily")
```
