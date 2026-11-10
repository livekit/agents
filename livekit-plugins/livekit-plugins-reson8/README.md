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
    stt=reson8.STT(model="rumble-2"),  # streaming + turn detection, language auto-detected
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
event = await reson8.STT(model="rumble-2").recognize(audio_buffer)
print(event.alternatives[0].text)
```

## Model

Use `rumble-2`, Reson8's latest model:

```python
reson8.STT(model="rumble-2")
```

See [Models](https://docs.reson8.dev/speech-to-text/models/).

## Turn detection

`SpeechStream.flush()` commits the current turn immediately. LiveKit never calls
it for you.

See [Turns](https://docs.reson8.dev/speech-to-text/turns/) for how turn events
work server-side.

## Languages

Leave `language` unset to **auto-detect** the spoken language, or pin recognition
to one or more supported codes. You can pass a single code, a comma-string, or a
list - a list is normalized to Reson8's comma-joined form and any code the
selected `model` does not support raises `ValueError` locally, before a request
is made.

```python
reson8.STT(model="rumble-2")                         # auto-detects the spoken language
reson8.STT(model="rumble-2", language="en")          # English only
reson8.STT(model="rumble-2", language="nl,de")       # Dutch or German
reson8.STT(model="rumble-2", language=["nl", "de"])  # same, as a list
```

`rumble-2` supports `af` `ar` `az` `be` `bg` `bn` `bs` `ca` `ceb` `cs` `da` `de`
`el` `en` `es` `et` `eu` `fa` `fi` `fil` `fr` `fy` `gl` `hi` `hr` `hu` `id` `is`
`it` `ja` `jv` `ko` `lt` `lv` `mk` `mr` `ms` `mt` `nb` `nl` `pl` `pt` `ro` `ru`
`sk` `sl` `sq` `sr` `sv` `ta` `th` `tr` `uk` `vi` `yue` `zh` - available as the
`reson8.SupportedLanguage` type and the `reson8.SUPPORTED_LANGUAGES` tuple. See
[Languages](https://docs.reson8.dev/speech-to-text/features/languages/).

## Configuration

Settings are grouped into sections, each of which validates itself on
construction:

```python
stt = reson8.STT(
    api_key="your-api-key",   # or set RESON8_API_KEY
    model="rumble-2",
    language="nl",
    audio=reson8.AudioOptions(sample_rate=16000),
    transcript=reson8.TranscriptOptions(words=True),
    biasing=reson8.BiasingOptions(custom_model_id="my-model"),
)
```

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
structured tokens whose shape you know up front. `rumble-2` does not support
`patterns` and caps `strength` at `1`; either raises `ValueError` locally.

Biasing is not free: phrases and patterns can *degrade* transcription of audio
that does not contain them, and stronger biasing introduces irrelevant terms.

| Field | Default | |
|---|---|---|
| `custom_model_id` | `None` | a custom model to bias toward, for a vocabulary too large for `phrases` or reused across requests |
| `phrases` | `None` | terms to bias toward, at most 250; needs no custom model |
| `strength` | `None` (server: `0.45`) | additive boost on the model's trained calibration, at most `1` on `rumble-2`. Raise only when expected terminology is not being recovered |
| `patterns` | `None` | shapes for short alphanumeric tokens to recover, e.g. `"AMZ[0-9]{6}"` or `"[0-9]{4,6}"` |

```python
# bias toward vocabulary the model would otherwise miss
stt = reson8.STT(biasing=reson8.BiasingOptions(phrases=["Reson8", "LiveKit"]))

# or recover a structured token, so its digits are not heard as words
stt = reson8.STT(biasing=reson8.BiasingOptions(patterns=["AMZ[0-9]{6}", "[0-9]{4,6}"]))
```

See [custom models](https://docs.reson8.dev/speech-to-text/features/custom-models/)
and [patterns](https://docs.reson8.dev/speech-to-text/features/patterns/).

`STT.update_options(...)` takes the same sections and changes them at runtime.
Reson8 reads its configuration from the query string, so a live stream applies
new settings by reconnecting immediately — which abandons the audio already
sent for the turn in progress, since Reson8 holds turn state server-side. Change
options between turns, or accept losing the one in progress. `AudioOptions` is
fixed for the life of a stream, since the input resampler is built when the
stream opens.

### Self-hosted deployments

`base_url` (or `RESON8_BASE_URL`) points the plugin at a Reson8 deployment other
than `https://api.reson8.dev`.

## Documentation

- [Reson8 API reference](https://docs.reson8.dev/api/speech-to-text/turns/)
- [Turns and turn detection](https://docs.reson8.dev/speech-to-text/turns/)
- [LiveKit Agents docs](https://docs.livekit.io/agents/)
- [Reson8 STT plugin guide](https://docs.livekit.io/agents/integrations/stt/reson8/)
