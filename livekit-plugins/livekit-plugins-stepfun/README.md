# StepFun plugin for LiveKit Agents

Support for [StepFun](https://platform.stepfun.ai/docs) speech recognition,
speech synthesis, and OpenAI-compatible chat models.

## Installation

```bash
pip install livekit-plugins-stepfun
# Include the OpenAI-compatible chat client:
pip install 'livekit-plugins-stepfun[llm]'
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
  gateway currently ignores it with server VAD, but it will finalize the last
  segment once commits are forwarded), and keep receiving: while a turn is pending this waits up
  to `conn_options.timeout` and fails on expiry, otherwise it ends after a short
  quiet period. The API has no acknowledgement that trailing silence was
  processed, so completion is best-effort: it succeeds only after a final was
  delivered, and otherwise reports that completion cannot be confirmed once a
  bounded interval elapses. A turn first reported later than the quiet period is
  not recovered. `aclose()` cancels without waiting.
  Failures after consuming audio are non-retryable because a fresh session
  cannot recover that input. `aclose()` cancels the stream without draining it.
- `stepfun.TTS()` supports streaming and one-shot synthesis. Discover voice IDs
  using `await tts.list_voices()`. One-shot synthesis streams PCM over SSE, so
  audio starts arriving before the whole request is synthesized; inputs longer
  than 1,000 characters are split into sequential requests and emitted as one
  audio segment. Each request has an idle-read timeout plus a total deadline
  that scales with its text length.
- `stepfun.LLM()` uses the optional OpenAI plugin for chat and tool calling.

Pass `language="zh"` for Chinese; the speech services default to English.
Streaming recognition defaults to `stepaudio-2.5-asr-stream`, batch recognition
to `stepaudio-3-asr-max`, synthesis to `stepaudio-3-tts`, and chat to
`step-3.7-flash`. Custom model IDs are accepted.
