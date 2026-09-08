# Maya Research plugin for LiveKit Agents

Support for voice synthesis with the [Maya Research](https://www.mayaresearch.ai/) API.

The plugin integrates Maya Research voice models, rather than being named after
one model generation. As of 8 September 2026, the public API documents **Maya Calyx**,
with ten Indian languages plus Indian English. The default model is `Maya Calyx`
and the default voice is `Aarav`; both are selected explicitly on the connection.
A conversation uses a persistent websocket with turn-level cancellation.

See [https://www.mayaresearch.ai/llm.txt](https://www.mayaresearch.ai/llm.txt) for more information.
The public [Maya Research Cookbook](https://github.com/MayaResearch/maya-cookbook)
has runnable TTS and voice-agent examples, API references, and coding-agent instructions.

## Installation

This plugin is proposed in [LiveKit PR 6899](https://github.com/livekit/agents/pull/6899)
and is not yet an upstream release. To test it now, use the cookbook's
[pinned LiveKit example](https://github.com/MayaResearch/maya-cookbook/tree/main/integrations/livekit).
Once published, it will be installable as `livekit-plugins-maya`.

## Pre-requisites

You'll need an API key from Maya Research. It can be set as an environment variable: `MAYA_API_KEY`

`MAYA_BASE_URL` overrides the endpoint for a self-hosted deployment.

## Usage

```python
from livekit.plugins import maya

tts = maya.TTS(model="Maya Calyx", voice="Aarav", language="hi")
```

Omit `language` for text that switches languages mid-sentence, so each part is
pronounced with its own script's rules.

Model and voice strings are passed through to Maya's API, so newly supported
models do not need a renamed plugin. Consult the current API reference before
changing them. This implementation emits 24 kHz mono signed 16-bit little-endian
PCM and rejects incompatible startup metadata before sending text.

### Streaming Indic text

The default sentence tokenizer breaks on western punctuation, not on the danda
(`।`), so a reply written in Devanagari and most other Indic scripts reaches the
socket as one sentence once the LLM has finished rather than sentence by
sentence as it is written. Pass a tokenizer that breaks on the danda to stream
those replies as they are generated:

```python
tts = maya.TTS(model="Maya Calyx", voice="Aarav", language="hi", tokenizer=my_indic_tokenizer)
```
