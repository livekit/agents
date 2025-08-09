# LiveKit Reverie Plugin

This plugin provides Indic language speech-to-text capabilities using [Reverie's STT API](https://docs.reverieinc.com/api-reference/speech-to-text-streaming).

Signup [here](https://revup.reverieinc.com/login) to create API keys.

## Installation

```bash
pip install livekit-plugins-reverie
```

## Configuration

You need to set the following environment variables:

- `REVERIE_API_KEY`: Your Reverie API key
- `REVERIE_APP_ID`: Your Reverie App ID


## Usage

```python
from livekit.plugins import reverie
from livekit.agents import stt

# Create STT instance with all available options
reverie_stt = reverie.STT(
    language="hi_en",       # Language code (default: "hi_en")
    domain="generic",       # Domain context (default: "generic")
    continuous=True,        # Continuous streaming (default: True)
    silence=0.5,           # Return final after silence in seconds (default: 0.5, max: 30)
    format="16k_int16",    # Audio format (default: "16k_int16")
    punctuate=False,       # Enable punctuation (default: False)
)

```

## Supported Languages

Reverie supports multiple Indian languages. Common language codes include:

- `hi_en` - Hinglish
- `hi` - Hindi
- `en` - English
- `bn` - Bengali
- `ta` - Tamil
- `te` - Telugu
- `mr` - Marathi
- `gu` - Gujarati
- `kn` - Kannada
- `ml` - Malayalam
- `pa` - Punjabi
- `or` - Odia
- `as` - Assamese

## Configuration Parameters

### Core Parameters
- `language`: Language code for recognition (e.g., "hi_en", "hi", "en", "bn") - Default: "hi_en"
- `domain`: Context domain for transcription (e.g., "generic") - Default: "generic"

### Audio & Format
- `format`: Audio format specification - Default: "16k_int16"
- `sample_rate`: Audio sample rate in Hz - Default: 16000

