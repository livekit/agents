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

# the default model is the realtime voxtral-mini-transcribe-realtime-3, pass a batch model for offline transcription
stt = mistralai.STT(model="voxtral-mini-transcribe-3")

# With context biasing
stt = mistralai.STT(
    model="voxtral-mini-transcribe-3",
    context_bias=["LiveKit", "Voxtral", "Mistral"]
)

# The larger model
stt = mistralai.STT(model="voxtral-small-transcribe-3", language="en")
```

#### Realtime streaming transcription

Voxtral Realtime streams interim transcripts over a WebSocket connection.

**Voxtral 3** (`voxtral-mini-transcribe-realtime-3`) has server-side turn detection: the
server tells the plugin when the user starts and stops speaking and the plugin finalizes the
transcript of each turn. It is enabled by default, and can be tuned with `turn_detection`.

```python
from livekit.plugins import mistralai

# server-side turn detection with the server defaults (the default model)
stt = mistralai.STT(language="en")

# tuned turn detection
stt = mistralai.STT(
    model="voxtral-mini-transcribe-realtime-3",
    turn_detection={"min_silence_ms": 400, "end_timeout_ms": 1500},
)
```

To let the voice pipeline decide when a turn ends instead of the server, pass `turn_detection=None`.
To use the end of the turns detected by the server as the end of the user turns of an
`AgentSession`, set `turn_detection="stt"` on the session or the agent.

**Earlier realtime models** (e.g. `voxtral-mini-transcribe-realtime-2602`) have no server-side
endpointing, and neither does Voxtral 3 with `turn_detection=None` or `diarize=True`. The voice pipeline
(`AgentSession`) tells the plugin when the user stopped speaking, based on its own VAD and turn
detection, and the plugin flushes the audio to get the final transcript.

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

**With a VAD.** Pass a `vad` to have the plugin detect the end of the speech itself: its end of speech
flushes the audio, and its start and end of speech are emitted as speech events. That is what
`turn_detection="stt"` of an `AgentSession` relies on with the models that have no server-side turn
detection, and it also makes a stream used on its own flush without `stream.flush()`. With Voxtral 3, a
`vad` replaces the server-side turn detection, which can't be used together with it.

```python
from livekit.plugins import mistralai, silero

stt = mistralai.STT(
    model="voxtral-mini-transcribe-realtime-2602",
    vad=silero.VAD.load(),
)
```

#### Speaker diarization

`diarize=True` labels each piece of text with its speaker. The speaker is exposed as `speaker_id` on the
transcripts, and on each segment (`words`).

```python
# batch models (Voxtral 2 and 3)
stt = mistralai.STT(model="voxtral-small-transcribe-3", diarize=True)

# Voxtral 3 realtime
stt = mistralai.STT(model="voxtral-mini-transcribe-realtime-3", diarize=True)
```

Earlier realtime models don't support it. On the Voxtral 3 realtime model, diarization can't be
used together with server-side turn detection, so `turn_detection` is disabled by default when
`diarize=True` and passing both raises a `ValueError`. The voice pipeline then decides when a turn
ends, like with `turn_detection=None`.

Without the server-side turn detection, the plugin emits no start and end of speech events, so the
`AgentSession` can't use `turn_detection="stt"`: no user turn would ever end. To use it with diarization,
give the STT a `vad`, which provides the events:

```python
stt = mistralai.STT(
    model="voxtral-mini-transcribe-realtime-3",
    diarize=True,
    vad=silero.VAD.load(),
)
session = AgentSession(stt=stt, turn_detection="stt", ...)
```

The realtime model numbers the speakers within one transcript only: the numbering restarts after each
flush, so the speakers of different transcripts can't be compared. The transcripts carry no timestamps.

#### Updating the options

`stt.update_options(...)` accepts `model`, `language`, `context_bias`, `target_streaming_delay_ms`,
`diarize` and `turn_detection`. Batch models apply them to the next transcription. The settings of a realtime
session can't change once audio has started, so changing one reconnects the active streams, and a
transcript that was not finalized yet is lost. The model can only be changed to one of the same
kind (batch or realtime).

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

#### Voxtral 3

The Voxtral 3 model (`voxtral-mini-tts-3`) is the default. By default the text is synthesized over http, sentence
by sentence. The speech starts as soon as the first sentence of the LLM reply is complete.

```python
from livekit.plugins import mistralai

tts = mistralai.TTS(model="voxtral-mini-tts-3", voice="en_paul_neutral")

# Voxtral 3 adds multilingual voices (`voice_<n>`, see `TTSVoxtral3Voices`) and still accepts the older ones.
# The older models only accept the older voices, anything else raises a ValueError.
tts = mistralai.TTS(model="voxtral-mini-tts-3", voice="voice_447")  # Margaret, en-US

# How expressive the speech is, between 1 and 5 (Voxtral 3 only)
tts = mistralai.TTS(model="voxtral-mini-tts-3", expressiveness=3)

# Pronunciation hints, as IPA phones per word (Voxtral 3 only, http only)
tts = mistralai.TTS(phoneme_hints={"tomato": ["t", "ə", "m", "ɑ", "t", "oʊ"]})
```

##### Realtime websocket

With `use_realtime=True` the speech is streamed over the realtime websocket api while the LLM generates the text,
using a pool of connections reused between the requests. `phoneme_hints` isn't supported in this mode.

```python
tts = mistralai.TTS(use_realtime=True)
```

The streamed speech is always PCM, `response_format` only applies to the http mode. The realtime engine can't
handle a single request of more than about 3,500 characters, so a longer reply is automatically split in several
requests.

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
    store=False,
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
