---
"livekit-plugins-cartesia": minor
---

Adds the Ink 2 STT model, which supports turn detection.

- The default model is now `ink-2` when `language="en"`, and `ink-whisper` otherwise.
  - Note that `ink-2` does not support aligned transcripts yet
- `STT.update_options()`: the `model` kwarg is now a no-op (emits a `DeprecationWarning`); construct a new `STT` instance to change the model.
- `SpeechStream` is now an abstract base class (alias of `CartesiaRecognizeStream`) — concrete streams are split into `LegacyRecognizeStream` (Ink Whisper) and `TurnsRecognizeStream` (Ink 2).
- `APIConnectionError(retryable=True)` is now raised on terminal server-side STT errors and on websocket connect failures, so the stream reconnects instead of silently dropping.
- Ink Whisper now respects flush sentinels so the server gets a `"finalize"` command and emits a transcript for each completed utterance, then sends `close` on stream end for a clean session teardown. Previously it sent a single `finalize` at stream end and never sent `close`.
- New `audio_chunk_duration_ms` constructor argument (defaults to 160 ms) controls the size of audio chunks sent to the Cartesia STT websocket.
- `STTLanguages` is now `Literal["en"]`. Ink Whisper continues to support all previous languages and `STT` type hints continue to accept any `str` value.
- `STTOptions` is deprecated and no longer used internally; kept as a re-export for backward compatibility.
