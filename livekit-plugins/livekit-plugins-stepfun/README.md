# StepFun plugin for LiveKit Agents

Support for [StepFun](https://docs.livekit.io/agents/models/stt/stepfun/) speech
recognition and synthesis.

## Installation

```bash
pip install livekit-plugins-stepfun
```

## Configuration

Set `STEPFUN_API_KEY`, or pass `api_key=` to the constructors. StepFun runs two
regional clusters with separate endpoints and separate credentials:

| Cluster | Endpoint | Value |
| --- | --- | --- |
| Global (default) | `api.stepfun.ai` | `"global"` |
| Mainland China | `api.stepfun.com` | `"china"` |

Select the cluster with `STEPFUN_CLUSTER` or `cluster=`. The two clusters use
separate endpoints and separate credentials, so a key from one will not work on
the other.

- `stepfun.STT()` streams recognition with server endpointing.
  `recognize()` uses the batch route; `use_streaming=False` selects batch
  recognition for sessions and requires local VAD through `stt.StreamAdapter`.
  Streaming input is intended for paced conversational audio, not file uploads;
  use `recognize()` for recorded audio. At end of input, server-VAD streams send
  endpointing silence and then a best-effort `input_audio_buffer.commit` (the
  gateway currently ignores it with server VAD), and keep receiving: while a
  turn is pending this waits up to `conn_options.timeout` and fails on expiry,
  otherwise it ends after a short quiet period. Once speech stops, if every
  unresolved transcription item produced usable interim text but no final, the
  stream preserves those interims and ends without inventing a final transcript.
  A pending item with no usable text still raises a timeout. `aclose()` cancels
  the stream without waiting or draining it.
  A connection lost after any audio was consumed, or while a turn is still
  pending, is reported as non-retryable: the SDK cannot replay that audio, and
  guessing which part the server finalized would risk dropping or repeating
  speech. Reliable recovery needs a server signal that identifies what a final
  covers; until then, recorded audio belongs on the batch `recognize()` route.
- `stepfun.TTS()` supports streaming and one-shot synthesis. Discover voice IDs
  using `await tts.list_voices()`. One-shot synthesis streams PCM over SSE, so
  audio starts arriving before the whole request is synthesized; inputs longer
  than 1,000 characters are split into sequential requests at sentence or word
  boundaries where possible and emitted as one audio segment. Text without a
  usable boundary is hard-split to respect the provider's limit. Independent
  requests do not share pronunciation/prosody context. Each request has an
  idle-read timeout plus a total deadline that scales with its text length.
- For chat, install `livekit-plugins-openai` and use `openai.LLM` with StepFun's
  OpenAI-compatible endpoint, for example:
  `openai.LLM(model="step-3.7-flash", base_url="https://api.stepfun.ai/v1", api_key=os.environ["STEPFUN_API_KEY"])`.

Pass `language="zh"` for Chinese; the speech services default to English.
Streaming recognition defaults to `stepaudio-2.5-asr-stream`, batch recognition
to `stepaudio-3-asr-max`, and synthesis to `stepaudio-3-tts`. Custom model IDs
are accepted.
