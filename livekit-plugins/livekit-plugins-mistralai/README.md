# Mistral AI Plugin for LiveKit Agents

Support for Mistral AI STT, TTS, and LLM services.

## Installation

```bash
pip install livekit-plugins-mistralai
```

## Pre-requisites

You'll need an API key from Mistral AI. It can be set as an environment variable:

```bash
export MISTRAL_API_KEY=your_api_key_here
```

## Usage

### Speech-to-Text (STT)

#### Offline transcription

```python
from livekit.plugins import mistralai

stt = mistralai.STT()

# With context biasing
stt = mistralai.STT(
    model="voxtral-mini-latest",
    context_bias=["LiveKit", "Voxtral", "Mistral"]
)
```

#### Realtime streaming transcription

Voxtral Realtime streams interim transcripts over a WebSocket connection. Since this
model has no server-side endpointing, the voice pipeline (`AgentSession`) tells the plugin when
the user stopped speaking, based on its own VAD and turn detection, and the plugin flushes the
audio to get the final transcript.

```python
from livekit.agents import AgentSession
from livekit.plugins import mistralai

session = AgentSession(
    stt=mistralai.STT(model="voxtral-mini-transcribe-realtime-2602"),
    ...
)
```

To use the STT on its own, outside of an `AgentSession`, flush the stream yourself with
`stream.flush()` when the user stops speaking to get the final transcript.

### Text-to-Speech (TTS)

```python
from livekit.plugins import mistralai

# Using a built-in voice
tts = mistralai.TTS(voice="en_paul_neutral")

# Using zero-shot voice cloning
import base64
ref_audio_b64 = base64.b64encode(open("sample.mp3", "rb").read()).decode()
tts = mistralai.TTS(ref_audio=ref_audio_b64)
```

### LLM

```python
from livekit.plugins import mistralai

llm = mistralai.LLM()

# With all available options
llm = mistralai.LLM(
    model="mistral-large-latest",
    temperature=0.7,
    top_p=0.9,
    max_completion_tokens=150,
    presence_penalty=0.1,
    frequency_penalty=0.1,
    random_seed=42,
    tool_choice="auto",
)

# With provider tools
agent = Agent(
    llm=llm,
    tools=[
        mistralai.tools.WebSearch(),
        mistralai.tools.CodeInterpreter(),
        mistralai.tools.DocumentLibrary(library_ids=["<your-library-id>"]),
        mistralai.tools.Connector(connector_id="<your_connector_id>")
    ]
)
```
