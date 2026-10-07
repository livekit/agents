# Oruk hosted STT plugin

This plugin sends final utterances to the authenticated [Oruk speech API](https://oruk.ai/docs). It defaults to stable Spectra-2 (`oruk-spectra-2`), using your existing Oruk API key and shared speech understanding plan minutes. This is separate from local Orukeet inference: audio leaves the device. Your plan's allowance, overage rate and spending cap apply.

Before a package release, install from this checkout:

```bash
uv sync --package livekit-agents --extra oruk --extra silero --dev
export ORUK_API_KEY='your-own-api-key'
uv run python examples/other/oruk_transcribe.py recording.wav
```

The example accepts PCM16 WAV and makes a real authenticated request. It does not require a LiveKit room, OpenAI credentials, or a local model download. Never put your key in source code or a browser client.

For a voice agent, use LiveKit's existing VAD adapter:

```python
from livekit.agents import AgentSession, stt
from livekit.plugins import oruk, silero

session = AgentSession(
    stt=stt.StreamAdapter(stt=oruk.STT(), vad=silero.VAD.load()),
    # Supply the LLM and TTS providers for your application.
)
```

This is final-utterance recognition, not native partial streaming. The adapter waits for VAD to end an utterance. Keep utterances between 45 ms and 60 seconds; the plugin downmixes PCM to mono and resamples to 16 kHz. Spectra-2 returns automatic transcription across 25 languages and clip-level scores, but this STT interface returns the transcript only. It does not invent a language identifier, word timestamps, confidence, or diarization. Language forcing is unsupported. Check the current API docs for other model limits before selecting a different `model`.

Transport retries keep the same request ID and exact WAV bytes. A `model_busy` rejection gets a new ID as required by the API; `upload_busy` retains its ID. The plugin honors `Retry-After`, rejects redirects, does not retry authentication or 409 completion/uncertainty errors, and never retries a malformed successful response. A lost completed response cannot be retrieved by replay: reconcile its request ID before manually creating another recognition call.

Call `await recognizer.aclose()` when using it outside an agent session. A supplied `httpx.AsyncClient` stays owned by the caller. This integration makes no latency or quality guarantee; use your own held-out recordings to measure the complete VAD, network, and inference path.

## Native Realtime (proposed)

`RealtimeSTT` is an additive candidate for the separate `oruk-realtime` WebSocket API. `STT` remains the batch default. The candidate is qualified only in offline tests using synthetic PCM, sockets and model stubs; it has not been published or qualified against a live provider.

Direct stream callers push mono PCM16 at 16 kHz and call `flush()` at a real turn boundary, or `end_input()` at EOF. Audio is sent as it arrives, with interim transcripts before commit. Each nonempty segment gets one socket and one request ID. A final transcript is released only after final text, usage and clean close, so it can carry late phrase events in `alternatives[0].metadata["oruk"]`. Usage alone is not success. Phrase data means observed events only: failed or missing phrase analysis is not a complete affect result. The SDK's default confidence `0.0` is not measured ASR confidence; detected language and measured confidence remain unavailable.

For a LiveKit agent, override `stt_node` and use `vad_stream_node(recognizer, audio, detector=your_vad)`. The stock node does not commit on VAD silence. This bridge targets the inspected Silero VAD processed-window contract, validates its PCM sequence, streams speech incrementally, commits on END, trims padding already sent in a previous turn, and preserves an active EOF tail. It does not resend END's whole-utterance audio. Other VAD implementations require matching contract tests. The caller creates/owns the detector; the bridge does not download or initialize a VAD model.

The candidate bounds each turn to 60 audio seconds and a 20-second completion wait, queued PCM to ten seconds, the pending VAD PCM buffer to one second, and queued boundaries to eight. The VAD bridge lazily splits large frames and waits up to one second for capacity instead of expanding its pending PCM buffer. After input ends, it allows one second for VAD drain; a stalled VAD fails explicitly. These limits bound that pending buffer, not caller-owned frames or temporary downmix/resampler allocations. It retains at most 256 phrase events/256 KiB, 64 KiB of usage data and 32,768 transcript characters. Interim notifications are capped at 128 per turn; final text is not truncated. A stream permits at most 64 unconsumed events across its lifetime, including the framework metrics tee, and an instance permits four unclosed streams. An exceeded limit is an explicit error. Upgrade retries are allowed only before any audio send attempt; framework retries cannot replay consumed input. Cancellation closes the socket and discards unfinished results. Audio can already have been processed or billed before commit or cancellation. Boundary timestamps are captured when the caller flushes, rather than when a queued turn is eventually transmitted; they are not acoustic speech-end measurements.

### Turn-completion hook candidate and core dependency

LiveKit 1.8.3 and 1.8.5 remove the STT request ID and alternative metadata on the normal path to `on_user_turn_completed`. `take_turn(request_id)` provides a single-use receipt, bounded to four cached turns with 30-second lookup validity per recognizer. Physical expiry is lazy on read/write; `aclose()` clears the cache. This is not a timed-deletion guarantee. It deliberately has no "latest emotion" or transcript-text lookup. Use one recognizer per agent session.

`examples/other/oruk_bound_turn_hook.py` supplies the VAD streaming node and a bounded context-injection hook. It requires the **proposed STT turn-identity core changes in this checkout**, which propagate ordered distinct IDs from accepted final transcripts to `ChatMessage.extra["stt_request_ids"]`, with explicit identity completeness. The adapter package alone does not supply this core change. Stock LiveKit 1.8.3/1.8.5 lack that binding, so the hook adds no affect context there. The combined candidate has offline tests invoking the real AgentSession hook against its pinned core source; this does not establish live-provider acceptance or availability in a released package.

The hook atomically consumes all exact receipts for the message (up to four provider turns), then checks the already-bound transcript for consistency. Missing, evicted, expired, incomplete, corrected or oversized results add no affect context. Receipt IDs never come from speaker IDs, language codes, transcript tokens, text matching or "latest result" state. The injected payload keeps observed phrase IDs, timings, scores and completion/failure types; it omits raw phrase text and arbitrary error strings, and remains bounded to 8 KiB. One recognizer belongs to one session. The application controls the resulting chat history; lookup expiry is not a promise to delete messages or model context.
