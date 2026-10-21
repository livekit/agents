# AssemblyAI plugin for LiveKit Agents

Support for Streaming Speech-to-Text and Streaming Text-to-Speech from AssemblyAI.

More information is available in the docs for the [STT](https://docs.livekit.io/agents/integrations/stt/assemblyai/) integration and for AssemblyAI's [Streaming TTS](https://www.assemblyai.com/docs/tts/getting-started/quickstart).

## Installation

```bash
pip install livekit-plugins-assemblyai
```

## Pre-requisites

You'll need to specify an AssemblyAI API Key. It can be set as environment variable: `ASSEMBLYAI_API_KEY`.

## Usage

```python
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import assemblyai

session = AgentSession(
    stt=assemblyai.STT(model="universal-3-6-pro", mode="balanced"),
    tts=assemblyai.TTS(model="assemblyai-tts-1", voice="jane"),
    # let AssemblyAI decide when the user has finished speaking
    turn_handling=TurnHandlingOptions(turn_detection="stt"),
    # ... llm, vad, etc.
)
```

### Speech-to-text

`assemblyai.STT` streams Universal-3.6 Pro by default (`model`). Useful options:

- `mode`: `"min_latency"`, `"balanced"` or `"max_accuracy"`.
- `language_codes`: steer transcription toward one or more expected languages (`"es"` or
  `["en", "es"]`).
- `prompt`: context about the audio, such as its domain or scenario, to help recognize likely
  vocabulary (up to 1,750 characters).
- `keyterms_prompt`: names and terms the model should get right.

### Text-to-speech

`assemblyai.TTS` streams speech over a pooled WebSocket and cancels in-flight audio when the user
interrupts. Useful options:

- `voice` and `language` (a code such as `"es"`): set independently, since every voice speaks
  every language. See the [voices](https://www.assemblyai.com/docs/tts/voices).
- `model`: defaults to `assemblyai-tts-1`.
- `voice_id`: a custom (cloned) voice, instead of `voice`.
- `base_url`: the US data zone by default; `wss://tts.eu.assemblyai.com` keeps data in the EU.
- `word_timestamps`: per-word timings for synchronized transcripts (default `True`).

Call `tts.prewarm()` to open the session before the first reply.
