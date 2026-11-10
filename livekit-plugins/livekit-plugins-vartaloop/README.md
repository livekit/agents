# Vartaloop plugin for LiveKit Agents

Support for [Vartaloop](https://vartaloop.ai/)'s Hinglish and Indian-language text-to-speech in LiveKit Agents.

See the [Vartaloop integration docs](https://docs.livekit.io/agents/models/tts/vartaloop/) for more information, and the [Vartaloop API reference](https://docs.vartaloop.ai) for the API and voices.

## Installation

```bash
pip install livekit-plugins-vartaloop
```

## Pre-requisites

You'll need an API key from Vartaloop. It can be set as an environment variable: `VARTALOOP_API_KEY`

## Usage

```python
from livekit.plugins import vartaloop

tts = vartaloop.TTS(voice="saanvi", expression="neutral", digits="auto")
```

Every option (`voice`, `expression`, `digits`, `speed`, `temperature`, `seed`, `postfx`, `profile`, `raw`) can be changed mid-session with `tts.update_options(...)`; it applies from the next sentence without reconnecting.
