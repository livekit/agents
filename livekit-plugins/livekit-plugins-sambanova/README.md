# LiveKit Plugins Sambanova

LiveKit Agent Framework plugin for services from [Sambanova](https://sambanova.ai/).

## Installation

```bash
pip install livekit-plugins-sambanova
```

## Pre-requisites

You'll need an API key from Sambanova. It can be set as an environment variable:
`SAMBANOVA_API_KEY`.

## Usage

```python
from livekit.plugins import sambanova

stt = sambanova.STT(
    model="Whisper-Large-v3",
    language="en",
)
```

By default, STT requests are sent to:
`https://api.sambanova.ai/v1/audio/transcriptions`
