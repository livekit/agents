# Chariot plugin for LiveKit Agents

Support for speech-to-text and text-to-speech with [Chariot](https://chariot.in/) in LiveKit Agents: English and Indian languages, including code-mixed speech such as Hinglish.

See the [Chariot docs](https://docs.chariot.in/api-reference/introduction) for the API, models and voices.

## Installation

```bash
pip install livekit-plugins-chariot
```

## Pre-requisites

You'll need an API key from Chariot. It can be set as an environment variable: `CHARIOT_API_KEY`. You can create one [here](https://platform.chariot.in).

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import chariot

session = AgentSession(
    stt=chariot.STT(language="hi"),  # omit the language to detect it
    tts=chariot.TTS(voice="bac7d666-094d-4698-91fa-741d60fce662"),  # Matt
    # ... llm, etc.
)
```

The STT transcribes English, Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Sanskrit, Tamil and Telugu. By default each transcript is finalized when the agent's turn detection ends the user's turn; pass `server_vad={}` to let Chariot's voice detection split utterances instead.

Any TTS voice speaks every supported language: the language of the audio follows the text. List the voices with `GET https://api.chariot.in/v1/voices`. An interrupted reply is cancelled on the server and its connection is kept for the next one.
