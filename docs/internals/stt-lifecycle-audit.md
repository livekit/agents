# STT lifecycle audit

Audit of the STT adapters present in this checkout against
[agents PR 8147](https://github.com/livekit/agents/pull/8147) and public provider documentation.
The documentation check covers stream completion, utterance completion, and input EOF.
It does not constitute a live integration test of each provider.

A final transcript or turn boundary does not generally finish a session. There are three
different completion contracts: an explicit terminal response, a server close after flushing,
or a bounded drain when no terminal guarantee is documented. Providers without a documented
guarantee must not inherit a guarantee from another provider or model.

| Provider / API | Official reference | Completion evidence |
| --- | --- | --- |
| AssemblyAI | [Message sequence](https://www.assemblyai.com/docs/streaming/message-sequence) | `Terminate` produces `Termination`, the last message. Turn finals are separate. |
| AWS Transcribe | [Streaming setup](https://docs.aws.amazon.com/transcribe/latest/dg/streaming-setting-up.html) | Empty audio event ends input. Continue reading the result event stream; a segment final is not stream completion. |
| Azure Speech | [Continuous recognition](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-recognize-speech) | SDK stopping/session events documented. Public documentation does not establish the adapter's raw WebSocket terminal ordering. |
| Baseten Whisper / Qwen | [Streaming transcription](https://docs.baseten.co/reference/inference-api/predict-endpoints/streaming-transcription-api) | New API acknowledges `end_audio`, then flushes and reports finished. Legacy deployed model protocols need compatibility handling. Qwen flush acknowledgements are distinct. |
| Cartesia Manual | [WebSocket](https://docs.cartesia.ai/api-reference/stt/websocket) | `finalize` produces `flush_done`; `close` produces `done`. Only the latter terminates the session. |
| Cartesia Turns | [Turns WebSocket](https://docs.cartesia.ai/api-reference/stt/turns/websocket) | JSON close flushes remaining events and closes the session. A turn end is not a session end. |
| Clova | [Recognition API](https://api.ncloud-docs.com/docs/en/ai-naver-clovaspeechrecognition-stt) | Batch HTTP response; no persistent STT stream to drain. |
| Deepgram Nova | [CloseStream](https://developers.deepgram.com/docs/close-stream) | Flushes remaining results and summary metadata before closing. `Finalize` is a separate operation. |
| Deepgram Flux | [Close stream](https://developers.deepgram.com/docs/flux/close-stream) | CloseStream emits remaining interim Updates, then closes without a status code. Latest Update is the final stream transcript; no EndOfTurn or summary metadata is promised. |
| ElevenLabs | [Realtime STT](https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime) | Commits finish segments, with alternate timestamp-bearing response variants. No global terminal message established. |
| fal | [Wizper API](https://fal.ai/models/fal-ai/wizper/api) | Job/HTTP result; no persistent STT stream. |
| Fireworks | [Streaming reference](https://fireworks.ai/docs/api-reference/audio-streaming-transcriptions) | Reference unavailable during audit (404). Terminal ordering remains unverified. |
| Gemini | [Live API](https://ai.google.dev/api/live), [transcription](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe) | `audioStreamEnd` ends audio input; input can resume. Generation/turn completion does not acknowledge all transcription work. |
| Gladia | [Live API](https://docs.gladia.io/api-reference/v2/live/websocket) | Transcription and optional postprocessing precede session completion. Do not equate a turn transcript with completion. |
| Gnani | [STT WebSocket](https://docs.gnani.ai/api/STT/stt-websocket) | Commit ends an utterance on a persistent socket. No whole-session terminal ordering established. |
| Google Cloud | [Streaming recognition](https://cloud.google.com/speech-to-text/docs/speech-to-text-requests#streaming_responses) | Request EOF and response RPC completion are distinct. `is_final` applies to a portion of audio. |
| Gradium | [WebSocket lifecycle](https://docs.gradium.ai/guides/websocket-lifecycle) | Send end-of-stream, read until server end-of-stream or error. Buffered text must be finalized first. |
| Inworld | [Best practices](https://docs.inworld.ai/stt/best-practices) | `closeStream` includes the final turn. Do not send `endTurn` immediately before closing. |
| Meta | [Speech to text](https://dev.meta.ai/docs/speech-to-text) | `endStream` ends input; keep reading until clean server close. Turn finals may overlap and carry separate IDs. |
| Microsoft MAI | [Realtime transcription](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe-2-streaming-realtime) | Commit produces completed transcription; the connection remains usable. Track pending work, not an invented session marker. |
| Mistral | [Realtime transcription](https://docs.mistral.ai/studio/audio/speech-to-text/transcribe-in-realtime) | Input end precedes remaining deltas and `transcription.done`. Public docs do not establish one acknowledgement per manual flush. |
| Nabrah | [Streaming STT](https://www.nabrah.ai/en/docs/api-reference/speech-to-text/stream-speech-to-text) | Partial and final transcripts documented; whole-session terminal ordering not established. |
| NVIDIA Riva | [ASR customization](https://docs.nvidia.com/nim/speech/26.10.0/asr/customization/customization.html) | Segment finals are not RPC completion. Streaming gRPC response termination is a separate boundary. |
| OpenAI | [Realtime transcription](https://platform.openai.com/docs/guides/realtime-transcription) | Completions may arrive out of order. Correlate item IDs and resolve every committed item; clear acknowledgement is not a transcript. |
| Palabra | [Realtime STT](https://docs.palabra.ai/docs/streaming_api/realtime_stt) | Empty binary frame requests final transcription/translations, then `stream_finished` and close. Client close can discard pending output. |
| RTZR | [Streaming STT](https://developers.rtzr.ai/docs/en/stt-streaming/) | `FINALIZE` commits an utterance; `EOS` ends the session. Finalization can produce no result for empty audio. |
| Sarvam | [Realtime STT](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/realtime-streaming) | Realtime `end` closes gracefully and `session.end` reports billed audio. Legacy flush protocol is separate. |
| Simplismart | [Whisper API](https://docs.simplismart.ai/api-reference/inference/whisper-v3) | Public reference covers HTTP inference; adapter WebSocket terminal ordering remains unverified. |
| SLNG | [Nova WebSocket](https://docs.slng.ai/api-reference/speech-to-text/deepgram/deepgram-nova-3/nova-3-websocket) | Backend-specific passthrough. No universal completion guarantee established across supported backends. |
| Smallest | [Response format](https://docs.smallest.ai/models/speech-to-text/realtime-web-socket/response-format) | `close_stream` produces terminal `is_last`; ordinary `is_final` is only an utterance boundary. |
| Soniox | [WebSocket API](https://docs.soniox.com/api-reference/stt/websocket-api) | Empty audio ends input, then remaining results and `finished` precede close. Manual-finalization markers are not session completion. |
| Speechmatics | [Agent STT](https://docs.speechmatics.com/api-ref/agent-stt-websocket) | EndOfStream produces EndOfTranscript after transcripts. Turn boundaries are separate. |
| Spitch | [Transcription](https://docs.spitch.app/features/transcription) | Batch response; no persistent STT stream. |
| Telnyx | [WebSocket lifecycle](https://developers.telnyx.com/docs/voice/stt/websocket-streaming) | Deepgram, Speechmatics and Soniox engines accept CloseStream, flush results, then close. Other engines require client close; no universal drain control. |
| xAI | [Speech to text](https://docs.x.ai/developers/rest-api-reference/inference/speech-to-text) | `audio.done` produces remaining transcripts, then `transcript.done` and close. |

ResON8 appears in the upstream PR but has no plugin in this checkout. Adding a new plugin
is separate from this audit. Native LiveKit Inference is audited separately from the Python
plugins: it needs heartbeat failure detection and a fixed 30-second absolute drain cap.

## Shared lifecycle changes

The Rust WebSocket executor must distinguish terminal-message completion from a clean
server close and from a best-effort bounded drain. A final transcript alone cannot satisfy
a promised terminal message. A timeout or early close must remain an error when the required
completion has not arrived. Existing plugins default to bounded settlement until their
actual protocol guarantee has been established.

Pending-item accounting is independent of this policy: it prevents an earlier final transcript
from settling later committed input. EOF must prevent a retry from opening an empty stream.
Failures sending finalization must not become success merely because the peer later closes.

## Regression coverage and limits

Provider tests exercise OpenAI item correlation and failures, Baseten EOF and overlapping
Qwen commits, ElevenLabs outstanding commits and documented rejections, Gemini transcripts
after turn completion, Gradium buffered text at EOF, Inworld's close command, Palabra's
trailing transcripts and required marker, and Telnyx Deepgram shutdown. Shared WebSocket
tests cover missing terminal responses after a final transcript. Rust tests cover terminal
policy, failed writes, EOF retries, native heartbeat failure and the absolute drain cap.

Unknown protocol ordering remains explicitly unknown. In particular, no minimum ElevenLabs
commit duration or one-response-per-Mistral-flush guarantee is inferred from upstream code.
The installed Palabra SDK exposes the `stream_finished` event type but drops its top-level
audio accounting fields; checking `final_audio_proc_ms` against `total_audio_proc_ms` requires
an SDK change or a transport replacement. This audit does not claim that check exists.
