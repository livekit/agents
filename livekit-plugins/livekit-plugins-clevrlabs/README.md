# Clevr Labs plugin for LiveKit Agents

Support for [Clevr Labs](https://theclevr.com/)' conversational speech model in LiveKit Agents.

More information is available in the docs for the [TTS](https://docs.livekit.io/agents/integrations/tts/) integration.

## Installation

```bash
pip install livekit-plugins-clevrlabs
```

## Pre-requisites

You'll need an API key from Clevr Labs, available at [theclevr.com](https://theclevr.com/). Pass it to the plugin via the `api_key` argument:

```python
from livekit.plugins import clevrlabs

session = AgentSession(tts=clevrlabs.TTS(api_key="clevr_..."), ...)
```
