# Vartalaap plugin for LiveKit Agents

Support for [Vartalaap](https://vartalaap.si/)'s Hinglish and Indian-language text-to-speech in LiveKit Agents.

More information is available in the docs for the [TTS](https://docs.livekit.io/agents/integrations/tts/vartalaap/) integration and in the [Vartalaap API reference](https://docs.vartalaap.si).

## Installation

```bash
pip install livekit-plugins-vartalaap
```

## Pre-requisites

You'll need an API key from Vartalaap. It can be set as an environment variable: `VARTALAAP_API_KEY`

## Usage

```python
from livekit.plugins import vartalaap

tts = vartalaap.TTS(voice="saanvi", expression="neutral", digits="auto")
```

Every option (`voice`, `expression`, `digits`, `speed`, `temperature`, `seed`, `postfx`, `profile`, `raw`) can be changed mid-session with `tts.update_options(...)`; it applies from the next sentence without reconnecting.
