# Reson8 plugin for LiveKit Agents

Support for [Reson8](https://reson8.dev) speech-to-text, with server-side turn detection.

More information is available in the docs for the [STT](https://docs.livekit.io/agents/integrations/stt/reson8/) integration.

## Installation

```bash
pip install livekit-plugins-reson8
```

## Pre-requisites

You'll need an API key from Reson8. It can be set as an environment variable: `RESON8_API_KEY`

## Usage

`reson8.STT` adapts to how LiveKit uses it:

- **Streaming** (`stream()`, used by voice agents) connects to the turn-aware
  endpoint. Reson8 detects conversational turn boundaries server-side: it emits
  a *preflight* transcript (an eager guess that the turn is over) that your agent
  can start responding to. A later guess replaces it, and the last one becomes
  the final transcript when the turn ends.
- **Batch** (`recognize()`) transcribes pre-recorded audio and returns the full
  transcript.

```python
from livekit.agents import AgentSession
from livekit.plugins import openai, reson8

session = AgentSession(
    stt=reson8.STT(),          # streaming + turn detection, language auto-detected
    llm=openai.LLM(),
    tts=openai.TTS(),
    # "stt" hands turn detection to Reson8 and lets the agent start
    # generating on the preflight transcript instead of the confirmation.
    turn_handling={
        "turn_detection": "stt",
        "preemptive_generation": {"enabled": True},
    },
)
```

By default LiveKit runs its own turn detector. Set `turn_handling` as above to
hand turn-taking to Reson8 instead.

### Transcribing a file

```python
event = await reson8.STT().recognize(audio_buffer)
print(event.alternatives[0].text)
```

## Turn detection

Reson8 decides turn boundaries by confidence: it emits the preflight transcript
at `turn.eager_probability` and commits the turn at `turn.final_probability`.
The server default of `0.92` is tuned for conversational speech and is slow to
commit a one-word answer; lower it to commit sooner, at the risk of cutting off
longer utterances.

```python
stt = reson8.STT(turn=reson8.TurnOptions(final_probability=0.7))
```

`SpeechStream.flush()` commits the current turn immediately, keeping
`final_probability` intact. LiveKit never calls it for you.

See [Turns](https://docs.reson8.dev/speech-to-text/turns/) for how turn events
work server-side.

## Languages

Leave `language` unset to **auto-detect** the spoken language, or pin recognition
to one or more supported codes. You can pass a single code, a comma-string, or a
list — a list is normalized to Reson8's comma-joined form and any unsupported code
raises `ValueError` locally, before a request is made.

```python
reson8.STT()                        # auto-detects the spoken language
reson8.STT(language="en")           # English only
reson8.STT(language="nl,de")        # Dutch or German
reson8.STT(language=["nl", "de"])   # same, as a list
```

Reson8 supports `de`, `en`, `es`, `fr`, `fy` (Frisian), `it`, `nl`, `pl`, `pt`
and `sv` — available as the `reson8.SupportedLanguage` type and the
`reson8.SUPPORTED_LANGUAGES` tuple. See
[Languages](https://docs.reson8.dev/speech-to-text/features/languages/).

## Configuration

Settings are grouped into sections, each of which validates itself on
construction:

```python
stt = reson8.STT(
    api_key="your-api-key",   # or set RESON8_API_KEY
    language="nl",
    turn=reson8.TurnOptions(final_probability=0.7),
    audio=reson8.AudioOptions(sample_rate=16000),
    transcript=reson8.TranscriptOptions(words=True),
    biasing=reson8.BiasingOptions(custom_model_id="my-model"),
)
```

### `TurnOptions`

The main lever on end-of-turn latency. `None` leaves the server's default.

| Field | Default | |
|---|---|---|
| `eager_probability` | `None` (server: `0.5`) | confidence at which the preflight transcript is emitted |
| `final_probability` | `None` (server: `0.92`) | confidence at which the turn commits |

### `AudioOptions`

Describes the audio sent to Reson8; it does not convert it. Streaming input is
resampled to `sample_rate`, but nothing remixes channels or transcodes samples,
so a pushed frame whose channel count disagrees with `num_channels` raises
rather than being relabelled.

| Field | Default | |
|---|---|---|
| `sample_rate` | `16000` | streaming input is resampled to this |
| `encoding` | `"pcm_s16le"` | the only value; `rtc.AudioFrame` is signed 16-bit PCM and is forwarded unchanged |
| `num_channels` | `1` | channel count of the frames you push, 1 to 10 |

### `TranscriptOptions`

| Field | Default | |
|---|---|---|
| `words` | `False` | word-level results, each with its own timing |
| `language` | `True` | the detected language code |
| `confidence` | `False` | per-word confidence, batch recognition only |
| `filler_mode` | `None` (server: `natural`) | `clean` removes filler words, `natural` lets the model decide, `verbatim` preserves them |

### `BiasingOptions`

Use `phrases` for a handful of terms on a single request, a `custom_model_id`
for a vocabulary that is larger or reused across requests, and `patterns` for
structured tokens whose shape you know up front.

Biasing is not free: phrases and patterns can *degrade* transcription of audio
that does not contain them, and stronger biasing introduces irrelevant terms.

| Field | Default | |
|---|---|---|
| `custom_model_id` | `None` | a custom model to bias toward, for a vocabulary too large for `phrases` or reused across requests |
| `phrases` | `None` | terms to bias toward, at most 250; needs no custom model |
| `strength` | `None` (server: `0.45`) | additive boost on the model's trained calibration. Raise only when expected terminology is not being recovered |
| `patterns` | `None` | shapes for short alphanumeric tokens to recover, e.g. `"AMZ[0-9]{6}"` or `"[0-9]{4,6}"` |

```python
# bias toward vocabulary the model would otherwise miss
stt = reson8.STT(biasing=reson8.BiasingOptions(phrases=["Reson8", "LiveKit"]))

# or recover a structured token, so its digits are not heard as words
stt = reson8.STT(biasing=reson8.BiasingOptions(patterns=["AMZ[0-9]{6}", "[0-9]{4,6}"]))
```

See [custom models](https://docs.reson8.dev/speech-to-text/features/custom-models/)
and [patterns](https://docs.reson8.dev/speech-to-text/features/patterns/).

`STT.update_options(...)` takes the same sections and changes them at runtime;
active streaming sessions reconnect automatically to apply them. `AudioOptions`
is fixed for the life of a stream, since the input resampler is built when the
stream opens.

### Self-hosted deployments

`base_url` (or `RESON8_BASE_URL`) points the plugin at a Reson8 deployment other
than `https://api.reson8.dev`.
