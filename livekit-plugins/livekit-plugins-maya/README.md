# Maya Research plugin for LiveKit Agents

Support for voice synthesis with the [Maya Research](https://www.mayaresearch.ai/) API.

Ten Indian languages plus Indian English, with every voice speaking all eleven.
A conversation runs over one persistent websocket with turn-level barge-in.

See [https://www.mayaresearch.ai/llm.txt](https://www.mayaresearch.ai/llm.txt) for more information.

## Installation

```bash
pip install livekit-plugins-maya
```

## Pre-requisites

You'll need an API key from Maya Research. It can be set as an environment variable: `MAYA_API_KEY`

`MAYA_BASE_URL` overrides the endpoint for a self-hosted deployment.

## Usage

```python
from livekit.plugins import maya

tts = maya.TTS(voice="Ananya", language="hi")  # see Maya's docs for voices
```

Omit `language` for text that switches languages mid-sentence, so each part is
pronounced with its own script's rules.

### Streaming Indic text

The default sentence tokenizer breaks on western punctuation, not on the danda
(`।`), so a reply written in Devanagari and most other Indic scripts reaches the
socket as one sentence once the LLM has finished rather than sentence by
sentence as it is written. Pass a tokenizer that breaks on the danda to stream
those replies as they are generated:

```python
tts = maya.TTS(voice="Ananya", language="hi", tokenizer=my_indic_tokenizer)
```
