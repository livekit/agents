# Rust file cleanup inventory

The inventory below records the working tree at `bab13261d`, after the Rust architecture
refactor and storage changes. The cleanup has now consolidated **62 files net**: all 32
recommended groups and 22 additional source files from the conditional set. Types, behavior,
scoped effects/events, machine namespaces and integration targets are preserved.

Original source paths and line counts remain below as the audit trail. They describe the
pre-cleanup snapshot; merge destinations describe the resulting layout.

The scan covers every tracked `.rs` file under `rust/`, including build tools and architecture
fixtures. Of 1,097 Rust files, 29 are generated and excluded from handwritten cleanup. There
were **1,068 handwritten files**, including **438 at 150 lines or fewer** (246 at 80 or fewer;
152 at 50 or fewer). Line counts include comments, blank lines and local tests. Larger related
files were inspected as destinations; the appendix exhaustively classifies the 438 small files.
Python and console file layout are outside this inventory.

The recommended set comprised **32 merge groups, removing 40 files net**. Its 44
participating small-file entries include a few destination files that remain. Another **30
entries were conditional**, and **364 were retained boundaries**. These are navigation improvements,
not a minimum file size or a request to merge every small module.

## Execution decisions

- Completed every recommended merge. The media archive-lookup helper moved into
  `media/build/cache.rs` rather than `build/mod.rs`: production and the isolated cache test
  target share that helper without importing the full native build script.
- Completed the conditional ABI declaration merges while preserving the generated speech-default
  and worker-reset-code destinations, serde gating and public Python exports.
- Completed the smaller same-owner core, voice, inference and runtime consolidations. Cursor
  drop admission, cancellation acknowledgement, avatar wakeups, storage receipts and provider
  retirement still use the same types and lifetime rules.
- Retained AgentRun's `model_step.rs` and the worker's `load_driver.rs` for file headroom and
  readable coordination; voice's `video_sampler.rs` remains a named sampling algorithm.
- Retained distinct realtime machine/driver fixtures and media's build settings. Combining the
  latter with cache code would add compiler-toolchain dependencies to cache integration tests.
- Retained runtime `run_stream/input.rs` as the named runner-input contract shared by direct
  and session execution. Its private consumer cursor moved beside the stream owner.

This is a layout cleanup: no compatibility forwarding modules, replacement trackers or new
behavioral policies were introduced. Bindings were regenerated from declarations.
The reduction counts the inventory's 65 retired source files and three new leaf modules;
concurrent A2A/storage changes are preserved and excluded from that count. Validation uses
the shared working tree, including those changes as present when each check runs.

## Merge rule and deliberate exceptions

Merge files when a developer learning one component currently has to open several files for
its ordinary data, private helpers, adapters or constants. Keep the existing types and policy
owners; reduce navigation without adding a new abstraction or changing behavior.

Keep these boundaries even when tiny:

- Category `effects.rs` declarations and crate effect indexes. Preserve the producer/executor
  links and domain ownership map.
- Scoped event/input protocols, `machine` namespaces, and recognizable domain entry modules
  with real children. A four-line ownership map can be useful; a forwarding directory with
  just one trivial child often is not.
- Pure decisions versus async execution, ABI declarations versus implementations, and actual
  fakeable port contracts. An `impl` of an owner's ABI trait can usually join that owner;
  a shared port trait should remain easy to find.
- Independent resource, cancellation, shutdown and drop lifetimes. Physical consolidation
  must not fuse their state or invent another tracker.
- Cargo integration targets, allocation-counter tests, distinct behavioral suites and shared
  fixtures. Small local tests may join their subject when they have no process-isolation
  requirement; the voice tool-result suite below is one concrete candidate.
- DST rule files: `fuzz/src/rules/mod.rs` explicitly defines one guarantee per file, and
  `build.rs` discovers them by filename. Keep distinct model/room/provider fakes and actors.
- Generated output, Cargo build scripts, architecture counterexamples and enforced process
  service boundaries such as `net::dns`, `net::blocking_io` and `ProcessEnv`.

Every proposed destination remains within the 1,250-line handwritten Rust limit. Near-limit
optional merges are explicitly lower priority. Keep `name/mod.rs` when children remain; use
`name.rs` for a leaf. Update internal paths without compatibility forwarding files. Preserve
Python export names and generated destinations when moving ABI declarations.

## Recommended merge groups

Paths in each table are relative to the named crate. Sizes are estimates from current raw
line counts, before duplicate imports, headers and module declarations are removed. They are
not a promise of the final formatted size. Each source is moved into the destination; existing
destination files remain.


### lk-agents-abi

| Existing files | Destination | Resulting lines, approx. | Why |
| --- | --- | ---: | --- |
| `src/a2a/handler.rs` (40) | `src/a2a/turn.rs` | 189 | HandlerOutcome and HandlerRun only describe normalization of the HandlerOutput already defined by managed-turn contracts; colocate (~189 lines). |
| `src/errors/stream.rs` (15) | `src/errors/provider.rs` | 115 | PullError is a tiny stream failure wrapper over ProviderError; colocate errors and retain root errors::PullError re-export (~115 lines). |
| `src/llm/chat_context/plans.rs` (49), `src/llm/ctx_diff.rs` (19), `src/llm/persistence.rs` (17) | `src/llm/chat_context/mod.rs` | 909 | ChatContext edit plans, diff contract and media persistence codec belong beside their conversation records; combined parent ~909 lines. Retain serde gating on codec implementations. |
| `src/llm/provider_format/error.rs` (23), `src/llm/provider_format/strict.rs` (23) | `src/llm/provider_format/mod.rs` | 93 | Small error declarations belong beside the provider-format category declarations; move both enums into the existing parent (~93 lines total). |
| `src/llm/schema/defaults.rs` (25), `src/llm/schema/mod.rs` (3) | `src/llm/schema.rs` | 28 | The entire schema module is one 25-line error enum plus a 3-line forwarding shell; collapse to one leaf (~28 lines). |
| `src/tts/markup/bindings.rs` (66), `src/tts/markup/expr.rs` (20), `src/tts/markup/mod.rs` (40), `src/tts/markup/mood.rs` (20), `src/tts/markup/provider.rs` (34), `src/tts/markup/steering.rs` (92) | `src/tts/markup.rs` | 272 | One markup ABI category is fragmented across six files (272 lines total); consolidate records, vocabulary, configuration and host contracts in one leaf, retaining exported names and generated Python destinations. |

### lk-agents-cli

| Existing files | Destination | Resulting lines, approx. | Why |
| --- | --- | ---: | --- |
| `src/signals.rs` (54) | `src/shutdown.rs` | 220 | Signal ownership, watchdog begin/release and shutdown completion are one process shutdown component (~220 lines); preserve singleton signal ownership and isolated shutdown integration tests. |

### lk-agents-core

| Existing files | Destination | Resulting lines, approx. | Why |
| --- | --- | ---: | --- |
| `src/agent_run/result.rs` (23) | `src/agent_run/mod.rs` | 37 | Five correlation ID declarations belong at the AgentRun public entry; result.rs contains no result implementation. |
| `src/llm/chat_plans.rs` (73) | `src/llm/history.rs` | 252 | Host copy/merge/truncate adapter directly calls the history planners; keep planning and its ABI adapter together. |
| `src/llm/provider_format/image.rs` (42) | `src/llm/provider_format/mod.rs` | 192 | Pre-resolved lookup and byte projections are shared provider-format data/helpers already re-exported at this parent. |
| `src/text_session/history.rs` (75) | `src/text_session/machine/mod.rs` | 796 | Private frame-history selection/delivery helpers serve only the TextSession owner; keep ordinary helper functions local without making another owner. |
| `src/tts/markup/xml.rs` (49) | `src/tts/markup/host.rs` | 126 | Only caller is the host markup utility; put XML rendering and its inline regression beside that call. |
| `src/tts/transcript.rs` (90) | `src/tts/mod.rs` | 94 | The TTS root is four lines and the file is one aligned-transcript normalizer; colocate at the TTS entry while retaining markup as child domain. |

### lk-agents-inference

| Existing files | Destination | Resulting lines, approx. | Why |
| --- | --- | ---: | --- |
| `src/amd/detector.rs` (78), `src/amd/mod.rs` (11) | `src/amd.rs` | 89 | Only child of amd; model selection and constants can live together in amd.rs (≈89). |
| `src/eot/eot.rs` (56) | `src/eot/mod.rs` | 98 | 56-line types file loaded through #[path] adds naming/navigation indirection. Put domain values/constants in eot/mod.rs (42+56≈98). |
| `src/fallback/tts/ports.rs` (101) | `src/fallback/tts/mod.rs` | 698 | Only an ABI trait impl for adjacent FallbackTts/TokenLog; colocate with native owner (597+101≈698). |
| `src/tts/ports.rs` (96) | `src/tts/mod.rs` | 767 | GatewayTts ABI impl only wraps its adjacent constructor/methods; colocate (671+96≈767). |
| `src/tts/adapter.rs` (56) | `src/tts/sentence_adapter.rs` | 611 | PreSplit/AdapterStreams are entry adapters of the sentence adapter; colocate (555+56≈611). |

### lk-agents-media

| Existing files | Destination | Resulting lines, approx. | Why |
| --- | --- | ---: | --- |
| `build/libraries.rs` (10) | `build/cache.rs` | 143 | Archive lookup is shared by the build entry and isolated cache tests; colocate with cache code to preserve their dependency boundary. |
| `src/images/prepare.rs` (90), `src/images/resolve.rs` (83) | `src/images/prepare.rs` | 173 | Destination of resolve.rs merge: image preparation, bounds, and resolver in ≈173 lines. |

### lk-agents-runtime

| Existing files | Destination | Resulting lines, approx. | Why |
| --- | --- | ---: | --- |
| `src/a2a/turn_handle.rs` (51) | `src/a2a/mod.rs` | 596 | The 51-line ABI impl only adapts the adjacent Turn owner; keep its implementation beside Turn (545+51≈596). |
| `src/host/cancellation.rs` (31) | `src/host/mod.rs` | 201 | The concrete cancellation adapter belongs next to call_cancellable, not in a one-type child (170+31≈201). |
| `src/room_io/output/segment_ids.rs` (38) | `src/room_io/output/transcription.rs` | 488 | Transcript segment IDs are already re-exported by transcription; colocate its tiny ID helper/tests with that owner (450+38≈488). |
| `src/voice/remote_session/host.rs` (29) | `src/voice/remote_session/mod.rs` | 94 | SessionHost is a 29-line wrapper of reply in the parent; put encoding and dispatch together (65+29≈94). |
| `src/voice/storage.rs` (73) | `src/voice/session/storage.rs` | 215 | SessionStorage startup and StorageReporter runtime receipts are one voice storage feature; combine at session/storage.rs, keeping decisions in voice (142+73≈215). |

### lk-agents-voice

| Existing files | Destination | Resulting lines, approx. | Why |
| --- | --- | ---: | --- |
| `src/agent_activity/tool_results_tests.rs` (90) | `src/agent_activity/tool_results.rs` | 303 | Small local two-test suite can be an inline cfg(test) module beside the private policy; retain assertions unchanged. |
| `src/agent_session/nodes.rs` (11), `src/recorder_io.rs` (31) | `src/agent_session/mod.rs` | 747 | One-field OutputNodes snapshot is plain session data; put beside SessionParts/Outputs, keeping runtime live nodes separate. |
| `src/room_io/ingress.rs` (46) | `src/room_io/mod.rs` | 51 | Small StreamBudget and three bounds are ordinary domain-level ingress data; parent currently five-line namespace. |
| `src/run_view.rs` (147) | `src/run_result.rs` | 654 | Voice run recording/output projection and live run settlement are the same run-result feature; colocate distinct types and preserve each contract. |
| `src/tool_body.rs` (14) | `src/tool_executor.rs` | 762 | Two shared completion-policy functions belong to the existing sans-I/O tool owner; runtime imports new path, never duplicates policy. |
| `src/workflows/prompts.rs` (55) | `src/workflows/dtmf.rs` + `src/workflows/warm_transfer.rs` | ≤732 / ≤546 | Move each private prompt constant to its DTMF or warm-transfer owner. Preserve the runtime import; do not duplicate prompts. |
| `src/workflows/instructions.rs` (17) | `src/workflows/mod.rs` | 249 | Two shared template helpers belong at common workflow entry; rendering algorithm remains generation::safe_render. |

## Conditional decisions

These had a defensible existing boundary or needed a concrete metadata/lifetime check. They
were reviewed during execution; the disposition column records the result. Sources that share a destination are one possible consolidation;
check their joint size rather than treating each estimate independently. In particular:

- Voice driver + initial setup + recognition dispatch would be about 871 lines.
- AgentRun's optional model-step move would take its machine root to about 1,117 lines;
  worker load-driver consolidation would take its root to about 1,078. Both have limited
  growth room and should be last choices.
- Combining core markup's host adapter with its parent after the recommended XML merge
  would be about 252 lines. Sentence-adapter + entry adapter + pacing would be about 711.
- ABI speech defaults and worker reset codes currently select separate generated modules.
  Verify declaration metadata can preserve those destinations before moving anything.
- The two realtime fixture modules construct different owners. Share only identical helpers;
  do not replace the pure-machine fixture with a driver fixture.


### lk-agents-abi

| File | Lines | Proposed destination | Condition / tradeoff  Disposition |
| --- | ---: | --- | ---  --- |
| `src/telemetry/observation.rs` | 40 | `src/telemetry/report/mod.rs` | Report projection and bounded report event retention are one reporting-boundary category; collect their declarations with report records (~389 lines). Keep upload and provider metrics independent.  Merged |
| `src/telemetry/report/projection.rs` | 73 | `src/telemetry/report/mod.rs` | Report projection and bounded report event retention are one reporting-boundary category; collect their declarations with report records (~389 lines). Keep upload and provider metrics independent.  Merged |
| `src/voice/rtc/playout.rs` | 18 | `src/voice/rtc/media.rs` | The 18-line playout ledger and 48-line source/synchronizer contracts can form one RTC media-source category (~66 lines). Optional: ledger is a distinct capability and its existing direct module path is also defensible.  Merged |
| `src/voice/speech.rs` | 21 | `src/voice/speech_handle.rs` | Speech priorities and default modality are read by the SpeechHandle boundary; colocate (~147 lines), but preserve speech_defaults generated-module metadata. Verify codegen file metadata before implementing.  Merged |
| `src/worker/http/reset_codes.rs` | 26 | `src/worker/http/mod.rs` | Reset constants are part of the same worker HTTP contract (~188 lines); preserve generated worker_reset_codes destination and compile-time upstream value assertions. Verify file metadata before implementing.  Merged |

### lk-agents-core

| File | Lines | Proposed destination | Condition / tradeoff  Disposition |
| --- | ---: | --- | ---  --- |
| `src/agent_run/machine/model_step.rs` | 71 | `src/agent_run/machine/mod.rs` | Private model-request accumulator can be local, but parent already1046 lines; resulting~1117 leaves little growth room. Keep if coordinator readability suffers.  Retained |
| `src/tokenize/host.rs` | 87 | `src/tokenize/mod.rs` | 87-line native host adapter can join27-line parent; optional because host.rs is a meaningful adapter navigation seam.  Merged |
| `src/tts/markup/host.rs` | 77 | `src/tts/markup/mod.rs` | After absorbing xml49, host adapter126 can join parent126 (~252); optional, dedicated host adapter is also a useful seam.  Merged |
| `src/tts/markup/steering.rs` | 130 | `src/tts/markup/instructions.rs` | Steering vocabulary/prompt construction and instructions share expressive-instruction feature (~326), but keep separate if provider-independent steering is navigated on its own.  Merged |

### lk-agents-inference

| File | Lines | Proposed destination | Condition / tradeoff  Disposition |
| --- | ---: | --- | ---  --- |
| `src/eot/options.rs` | 63 | `src/eot/detector.rs` | 63 lines of constructor selection can live with AudioEot creation (206+63≈269); retain module if externally consumed as its own API.  Merged |
| `src/interruption/cache.rs` | 92 | `src/interruption/machine.rs` | Single-owner bounded FIFO used only by InterruptionMachine; inline private cache and local tests (549+92≈641) if reducing navigation is preferred.  Merged |
| `src/interruption/detector.rs` | 69 | `src/interruption/mod.rs` | Native detector is a small host-facing facade (46+69≈115); inline in domain root if keeping this layer explicit.  Merged |
| `src/interruption/retry.rs` | 75 | `src/interruption/transport.rs` | 75-line transport error mapping can join its only executor (846+75≈921), retaining private pure helper functions.  Merged |
| `src/realtime/testfix.rs` | 22 | `src/realtime/openai/testfix.rs` | Two fixture modules define msg/machine. Can share common helpers or relocate into one test fixture owner, but one produces pure RealtimeMachine and the other a driver wrapper: do not collapse these distinct return semantics blindly.  Retained |
| `src/tts/checks.rs` | 39 | `src/tts/stream.rs` | Host synthesis validation and host stream wrapper can share one file (119+39≈158); keep validation as its declared trait implementation.  Merged |
| `src/tts/pacer.rs` | 100 | `src/tts/sentence_adapter.rs` | SentencePacer belongs to sentence batching (555+56+100≈711 with AdapterStreams); keep separate if its public algorithm is used independently.  Merged |
| `src/tts/stream.rs` | 119 | `src/tts/stream.rs` | Destination of optional checks.rs merge, ≈158 total; no need to merge the stream with gateway wire/machine.  Destination retained |

### lk-agents-media

| File | Lines | Proposed destination | Condition / tradeoff  Disposition |
| --- | ---: | --- | ---  --- |
| `build/cache.rs` | 133 | `build/cache.rs` | Destination of optional settings merge; preserve shared cache ownership and process lock responsibilities.  Retained |
| `build/settings.rs` | 99 | `build/cache.rs` | Native compiler/cache key setup may join native cache utilities (133+99≈232); keep compiler identity and cache artifact code clearly named.  Retained |

### lk-agents-runtime

| File | Lines | Proposed destination | Condition / tradeoff  Disposition |
| --- | ---: | --- | ---  --- |
| `src/a2a/door/admission.rs` | 70 | `src/a2a/door/mod.rs` | Private budget reservation is used by this door only; can inline private Budget/Admission types into door (417+70≈487), though keeping admission test visibility has value.  Merged |
| `src/avatar/queue.rs` | 90 | `src/avatar/output.rs` | Wakeup wrapper for one avatar audio output can live beside its consumer (600+90≈690); retain a distinct QueueAudio type/lifetime.  Merged |
| `src/rtc/playout.rs` | 27 | `src/rtc/mod.rs` | 27-line ABI wrapper could live in RTC facade (10+27≈37); keep type distinct and never merge async wrapper with voice ledger policy.  Merged |
| `src/run_stream/cursor.rs` | 86 | `src/run_stream/mod.rs` | Cursor is private to stream-consumer admission; may inline beside RunStream (296+86≈382), but preserve its drop-before-first-poll lifetime rule.  Merged |
| `src/run_stream/input.rs` | 99 | `src/run_stream/mod.rs` | Two runner input adapters can sit alongside run-stream orchestration; combined with cursor group ≈481; retain Input trait as executor test boundary.  Retained |
| `src/voice/session/providers.rs` | 62 | `src/voice/session/options.rs` | Provider retention and options updates are related (198+62≈260); merge only if lock/destructor boundary remains explicit. Do not add to session/mod.rs (1222).  Merged |

### lk-agents-voice

| File | Lines | Proposed destination | Condition / tradeoff  Disposition |
| --- | ---: | --- | ---  --- |
| `src/agent_activity/video_sampler.rs` | 50 | `src/agent_activity/io.rs` | 50-line video-admission predicate can join449-line activity I/O decisions; optional because sampling is independently reusable and tested through feature paths.  Retained |
| `src/agent_session/machine/initial.rs` | 140 | `src/agent_session/machine/driver.rs` | Same VoiceDriver construction impl; optional flattening within existing machine boundary (~788 alone;871 with recognition).  Merged |
| `src/agent_session/machine/recognition.rs` | 83 | `src/agent_session/machine/driver.rs` | 83-line recognition-effect dispatch is VoiceDriver coordination, not a new owner; optional flatten within machine (~731 alone;871 with initial).  Merged |
| `src/session_start.rs` | 85 | `src/agent_session/lifecycle.rs` | Startup selection/lifetime facts can sit with lifecycle contracts (~226); retain distinct concepts and core OwnerLifecycle re-export, no new policy fusion.  Merged |

### lk-agents-worker

| File | Lines | Proposed destination | Condition / tradeoff  Disposition |
| --- | ---: | --- | ---  --- |
| `src/worker/load_driver.rs` | 114 | `src/worker/mod.rs` | This is only impl Worker methods using parent-private load fields and completion variants; merging removes one owner-navigation hop (~1078 lines). Optional because only ~172 lines headroom remain under1250. Keep load.rs decision model separate.  Retained |


## Complete small-file inventory

Every handwritten Rust file of 150 lines or fewer in the pre-cleanup snapshot appears exactly
once below. These are the original inventory decisions; the execution decisions above record
which conditional entries were merged or retained. “Merge” means
participation in a recommended group, including destinations where noted; “Conditional” means
optional consolidation; “Keep” means a deliberate boundary or no useful same-owner merge
identified. “Keep” does not prohibit absorbing a helper into that file.

A small file is not automatically an API primitive: grouping related private code can reduce
navigation without reducing the number of public types. No type removals or naming changes
are proposed here.


### architecture fixtures inventory

2 small files: 0 merge participants, 0 conditional, 2 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `fixtures.rs` | 76 | Keep | Deliberately failing purity-check fixture; compiled under its own policy. |
| `io/fixtures.rs` | 18 | Keep | Deliberately failing I/O-check fixture; must remain separate from purity fixtures. |

### lk-agents-abi inventory

122 small files: 15 merge participants, 5 conditional, 102 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `src/a2a/card.rs` | 23 | Keep | Distinct A2A public/protocol category (card, metadata, arrival options, door, remote interaction); keep endpoint/client/protocol roles visible. |
| `src/a2a/door.rs` | 86 | Keep | Distinct A2A public/protocol category (card, metadata, arrival options, door, remote interaction); keep endpoint/client/protocol roles visible. |
| `src/a2a/handler.rs` | 40 | Merge | `src/a2a/turn.rs`. HandlerOutcome and HandlerRun only describe normalization of the HandlerOutput already defined by managed-turn contracts; colocate (~189 lines). |
| `src/a2a/metadata.rs` | 29 | Keep | Distinct A2A public/protocol category (card, metadata, arrival options, door, remote interaction); keep endpoint/client/protocol roles visible. |
| `src/a2a/mod.rs` | 12 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/a2a/options.rs` | 79 | Keep | Distinct A2A public/protocol category (card, metadata, arrival options, door, remote interaction); keep endpoint/client/protocol roles visible. |
| `src/a2a/remote.rs` | 88 | Keep | Distinct A2A public/protocol category (card, metadata, arrival options, door, remote interaction); keep endpoint/client/protocol roles visible. |
| `src/a2a/turn.rs` | 149 | Keep | Distinct A2A public/protocol category (card, metadata, arrival options, door, remote interaction); keep endpoint/client/protocol roles visible. |
| `src/agent/assertions.rs` | 73 | Keep | Distinct Agent run resource lifecycle or assertion API; do not merge configuration, invocation resources and test assertions. |
| `src/agent/authoring.rs` | 113 | Keep | Distinct Agent run resource lifecycle or assertion API; do not merge configuration, invocation resources and test assertions. |
| `src/agent/resources.rs` | 74 | Keep | Distinct Agent run resource lifecycle or assertion API; do not merge configuration, invocation resources and test assertions. |
| `src/cli.rs` | 125 | Keep | Distinct shared ABI contract or authoring/registration entry point; no same-owner recipient justifies merging solely for size. |
| `src/errors/host.rs` | 94 | Keep | Shared exception taxonomy, host exception ownership or reference registry; independent lifetime/semantic contract. |
| `src/errors/host_capture.rs` | 104 | Keep | Shared exception taxonomy, host exception ownership or reference registry; independent lifetime/semantic contract. |
| `src/errors/mod.rs` | 59 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/errors/provider.rs` | 100 | Keep | Shared exception taxonomy, host exception ownership or reference registry; independent lifetime/semantic contract. |
| `src/errors/side_table.rs` | 145 | Keep | Shared exception taxonomy, host exception ownership or reference registry; independent lifetime/semantic contract. |
| `src/errors/stream.rs` | 15 | Merge | `src/errors/provider.rs`. PullError is a tiny stream failure wrapper over ProviderError; colocate errors and retain root errors::PullError re-export (~115 lines). |
| `src/inference/avatar_session.rs` | 119 | Keep | Distinct shared ABI contract or authoring/registration entry point; no same-owner recipient justifies merging solely for size. |
| `src/inference/fallback.rs` | 48 | Keep | Distinct shared ABI contract or authoring/registration entry point; no same-owner recipient justifies merging solely for size. |
| `src/invoke.rs` | 126 | Keep | Distinct shared ABI contract or authoring/registration entry point; no same-owner recipient justifies merging solely for size. |
| `src/lib.rs` | 136 | Keep | Required crate entry point and public composition surface; keep compact. |
| `src/llm/chat_context/plans.rs` | 49 | Merge | `src/llm/chat_context/mod.rs`. ChatContext edit plans, diff contract and media persistence codec belong beside their conversation records; combined parent ~909 lines. Retain serde gating on codec implementations. |
| `src/llm/ctx_diff.rs` | 19 | Merge | `src/llm/chat_context/mod.rs`. ChatContext edit plans, diff contract and media persistence codec belong beside their conversation records; combined parent ~909 lines. Retain serde gating on codec implementations. |
| `src/llm/events.rs` | 148 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/llm/mod.rs` | 21 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/llm/persistence.rs` | 17 | Merge | `src/llm/chat_context/mod.rs`. ChatContext edit plans, diff contract and media persistence codec belong beside their conversation records; combined parent ~909 lines. Retain serde gating on codec implementations. |
| `src/llm/provider_format/error.rs` | 23 | Merge | `src/llm/provider_format/mod.rs`. Small error declarations belong beside the provider-format category declarations; move both enums into the existing parent (~93 lines total). |
| `src/llm/provider_format/host.rs` | 75 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/llm/provider_format/image.rs` | 68 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/llm/provider_format/mod.rs` | 47 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/llm/provider_format/strict.rs` | 23 | Merge | `src/llm/provider_format/mod.rs`. Small error declarations belong beside the provider-format category declarations; move both enums into the existing parent (~93 lines total). |
| `src/llm/schema/defaults.rs` | 25 | Merge | `src/llm/schema.rs`. The entire schema module is one 25-line error enum plus a 3-line forwarding shell; collapse to one leaf (~28 lines). |
| `src/llm/schema/mod.rs` | 3 | Merge | `src/llm/schema.rs`. The entire schema module is one 25-line error enum plus a 3-line forwarding shell; collapse to one leaf (~28 lines). |
| `src/llm/toolsets.rs` | 119 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/logging.rs` | 41 | Keep | Distinct shared ABI contract or authoring/registration entry point; no same-owner recipient justifies merging solely for size. |
| `src/media/codecs.rs` | 74 | Keep | Distinct codec/image/emitter capability boundary shared with media implementations. |
| `src/media/emitter.rs` | 132 | Keep | Distinct codec/image/emitter capability boundary shared with media implementations. |
| `src/media/images.rs` | 120 | Keep | Distinct codec/image/emitter capability boundary shared with media implementations. |
| `src/media/mod.rs` | 7 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/metrics/mod.rs` | 56 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/metrics/testkit.rs` | 139 | Keep | Distinct shared ABI contract or authoring/registration entry point; no same-owner recipient justifies merging solely for size. |
| `src/recorder.rs` | 96 | Keep | Distinct shared ABI contract or authoring/registration entry point; no same-owner recipient justifies merging solely for size. |
| `src/resources.rs` | 80 | Keep | Distinct shared ABI contract or authoring/registration entry point; no same-owner recipient justifies merging solely for size. |
| `src/storage/context.rs` | 142 | Keep | Distinct storage context/handle/service/voice contract, with its own lifetime and consumer boundary. |
| `src/storage/handles.rs` | 104 | Keep | Distinct storage context/handle/service/voice contract, with its own lifetime and consumer boundary. |
| `src/storage/http.rs` | 9 | Keep | Distinct storage context/handle/service/voice contract, with its own lifetime and consumer boundary. |
| `src/storage/voice.rs` | 51 | Keep | Distinct storage context/handle/service/voice contract, with its own lifetime and consumer boundary. |
| `src/stt/fallback.rs` | 66 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/stt/mod.rs` | 12 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/stt/stream_adapter.rs` | 46 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/telemetry/attributes.rs` | 101 | Keep | Separate telemetry contract (provider metrics, transport parent, upload, attributes); provider metrics and Cloud reports have distinct owners. |
| `src/telemetry/mod.rs` | 26 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/telemetry/observation.rs` | 40 | Conditional | `src/telemetry/report/mod.rs`. Report projection and bounded report event retention are one reporting-boundary category; collect their declarations with report records (~389 lines). Keep upload and provider metrics independent. |
| `src/telemetry/provider.rs` | 74 | Keep | Separate telemetry contract (provider metrics, transport parent, upload, attributes); provider metrics and Cloud reports have distinct owners. |
| `src/telemetry/report/projection.rs` | 73 | Conditional | `src/telemetry/report/mod.rs`. Report projection and bounded report event retention are one reporting-boundary category; collect their declarations with report records (~389 lines). Keep upload and provider metrics independent. |
| `src/telemetry/transport.rs` | 16 | Keep | Separate telemetry contract (provider metrics, transport parent, upload, attributes); provider metrics and Cloud reports have distinct owners. |
| `src/telemetry/upload.rs` | 83 | Keep | Separate telemetry contract (provider metrics, transport parent, upload, attributes); provider metrics and Cloud reports have distinct owners. |
| `src/testkit.rs` | 37 | Keep | Distinct shared ABI contract or authoring/registration entry point; no same-owner recipient justifies merging solely for size. |
| `src/tokenize/bindings.rs` | 134 | Keep | Distinct tokenizer host binding/search/language-data category; keep data and exported operations separate. |
| `src/tokenize/language/mod.rs` | 13 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/tokenize/mod.rs` | 15 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/tokenize/search.rs` | 59 | Keep | Distinct tokenizer host binding/search/language-data category; keep data and exported operations separate. |
| `src/tts/markup/bindings.rs` | 66 | Merge | `src/tts/markup.rs`. One markup ABI category is fragmented across six files (272 lines total); consolidate records, vocabulary, configuration and host contracts in one leaf, retaining exported names and generated Python destinations. |
| `src/tts/markup/expr.rs` | 20 | Merge | `src/tts/markup.rs`. One markup ABI category is fragmented across six files (272 lines total); consolidate records, vocabulary, configuration and host contracts in one leaf, retaining exported names and generated Python destinations. |
| `src/tts/markup/mod.rs` | 40 | Merge | `src/tts/markup.rs`. One markup ABI category is fragmented across six files (272 lines total); consolidate records, vocabulary, configuration and host contracts in one leaf, retaining exported names and generated Python destinations. |
| `src/tts/markup/mood.rs` | 20 | Merge | `src/tts/markup.rs`. One markup ABI category is fragmented across six files (272 lines total); consolidate records, vocabulary, configuration and host contracts in one leaf, retaining exported names and generated Python destinations. |
| `src/tts/markup/provider.rs` | 34 | Merge | `src/tts/markup.rs`. One markup ABI category is fragmented across six files (272 lines total); consolidate records, vocabulary, configuration and host contracts in one leaf, retaining exported names and generated Python destinations. |
| `src/tts/markup/steering.rs` | 92 | Merge | `src/tts/markup.rs`. One markup ABI category is fragmented across six files (272 lines total); consolidate records, vocabulary, configuration and host contracts in one leaf, retaining exported names and generated Python destinations. |
| `src/tts/mod.rs` | 10 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/tts/stream_adapter.rs` | 48 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/tts/stream_pacer.rs` | 20 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/tts/streams.rs` | 33 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/tts/transcript.rs` | 38 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/turn/detector.rs` | 69 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/turn/eot.rs` | 111 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/turn/events.rs` | 35 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/turn/interruption.rs` | 77 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/turn/keyterms.rs` | 127 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/turn/mod.rs` | 9 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/types/identity.rs` | 43 | Keep | Shared value contract with a distinct semantic subject (identity, provider options or retry); do not create a generic mixed types file. |
| `src/types/mod.rs` | 22 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/types/provider.rs` | 59 | Keep | Shared value contract with a distinct semantic subject (identity, provider options or retry); do not create a generic mixed types file. |
| `src/types/retry.rs` | 40 | Keep | Shared value contract with a distinct semantic subject (identity, provider options or retry); do not create a generic mixed types file. |
| `src/utils/audio.rs` | 59 | Keep | Distinct cross-crate utility contract or media representation; shared by multiple consumers, with no single more specific owner. |
| `src/utils/byte_stream.rs` | 40 | Keep | Distinct cross-crate utility contract or media representation; shared by multiple consumers, with no single more specific owner. |
| `src/utils/connection_pool.rs` | 31 | Keep | Distinct cross-crate utility contract or media representation; shared by multiple consumers, with no single more specific owner. |
| `src/utils/dates.rs` | 67 | Keep | Distinct cross-crate utility contract or media representation; shared by multiple consumers, with no single more specific owner. |
| `src/utils/filters.rs` | 68 | Keep | Distinct cross-crate utility contract or media representation; shared by multiple consumers, with no single more specific owner. |
| `src/utils/host_ref.rs` | 124 | Keep | Distinct cross-crate utility contract or media representation; shared by multiple consumers, with no single more specific owner. |
| `src/utils/maybe.rs` | 39 | Keep | Distinct cross-crate utility contract or media representation; shared by multiple consumers, with no single more specific owner. |
| `src/utils/mod.rs` | 16 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/utils/resample.rs` | 24 | Keep | Distinct cross-crate utility contract or media representation; shared by multiple consumers, with no single more specific owner. |
| `src/utils/strings.rs` | 9 | Keep | Distinct cross-crate utility contract or media representation; shared by multiple consumers, with no single more specific owner. |
| `src/utils/timer.rs` | 132 | Keep | Distinct cross-crate utility contract or media representation; shared by multiple consumers, with no single more specific owner. |
| `src/utils/video.rs` | 60 | Keep | Distinct cross-crate utility contract or media representation; shared by multiple consumers, with no single more specific owner. |
| `src/utils/vocabulary.rs` | 71 | Keep | Distinct cross-crate utility contract or media representation; shared by multiple consumers, with no single more specific owner. |
| `src/vad/events.rs` | 110 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/vad/inference.rs` | 70 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/vad/mod.rs` | 6 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/vad/vad.rs` | 60 | Keep | Distinct model/event/adapter contract or domain category; preserve provider seam and event discoverability. |
| `src/voice/avatar_queue.rs` | 60 | Keep | Named voice boundary with its own owner or host API (session, room, avatar, console, workflow, run view); small size is not evidence to merge it into session. |
| `src/voice/background_audio.rs` | 141 | Keep | Named voice boundary with its own owner or host API (session, room, avatar, console, workflow, run view); small size is not evidence to merge it into session. |
| `src/voice/console.rs` | 56 | Keep | Named voice boundary with its own owner or host API (session, room, avatar, console, workflow, run view); small size is not evidence to merge it into session. |
| `src/voice/mod.rs` | 26 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/voice/room_owner.rs` | 149 | Keep | Named voice boundary with its own owner or host API (session, room, avatar, console, workflow, run view); small size is not evidence to merge it into session. |
| `src/voice/rtc/error.rs` | 82 | Keep | Distinct room/RTC capability or port contract, kept visible at the runtime/fake boundary. |
| `src/voice/rtc/media.rs` | 48 | Keep | Distinct room/RTC capability or port contract, kept visible at the runtime/fake boundary. |
| `src/voice/rtc/playout.rs` | 18 | Conditional | `src/voice/rtc/media.rs`. The 18-line playout ledger and 48-line source/synchronizer contracts can form one RTC media-source category (~66 lines). Optional: ledger is a distinct capability and its existing direct module path is also defensible. |
| `src/voice/rtc/ports.rs` | 100 | Keep | Distinct room/RTC capability or port contract, kept visible at the runtime/fake boundary. |
| `src/voice/run_view.rs` | 31 | Keep | Named voice boundary with its own owner or host API (session, room, avatar, console, workflow, run view); small size is not evidence to merge it into session. |
| `src/voice/session_owner.rs` | 149 | Keep | Named voice boundary with its own owner or host API (session, room, avatar, console, workflow, run view); small size is not evidence to merge it into session. |
| `src/voice/session_storage.rs` | 44 | Keep | Named voice boundary with its own owner or host API (session, room, avatar, console, workflow, run view); small size is not evidence to merge it into session. |
| `src/voice/speech.rs` | 21 | Conditional | `src/voice/speech_handle.rs`. Speech priorities and default modality are read by the SpeechHandle boundary; colocate (~147 lines), but preserve speech_defaults generated-module metadata. Verify codegen file metadata before implementing. |
| `src/voice/speech_handle.rs` | 126 | Keep | Named voice boundary with its own owner or host API (session, room, avatar, console, workflow, run view); small size is not evidence to merge it into session. |
| `src/voice/workflows/dtmf.rs` | 76 | Keep | Named voice boundary with its own owner or host API (session, room, avatar, console, workflow, run view); small size is not evidence to merge it into session. |
| `src/voice/workflows/prompts.rs` | 40 | Keep | Named voice boundary with its own owner or host API (session, room, avatar, console, workflow, run view); small size is not evidence to merge it into session. |
| `src/worker/http/reset_codes.rs` | 26 | Conditional | `src/worker/http/mod.rs`. Reset constants are part of the same worker HTTP contract (~188 lines); preserve generated worker_reset_codes destination and compile-time upstream value assertions. Verify file metadata before implementing. |
| `src/worker/ipc.rs` | 33 | Keep | Distinct worker job, process host or server host boundary; retain protocol/service discoverability. |
| `src/worker/job.rs` | 117 | Keep | Distinct worker job, process host or server host boundary; retain protocol/service discoverability. |
| `src/worker/mod.rs` | 44 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/worker/server_host.rs` | 150 | Keep | Distinct worker job, process host or server host boundary; retain protocol/service discoverability. |

### lk-agents-bindgen inventory

1 small files: 0 merge participants, 0 conditional, 1 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `src/lib.rs` | 137 | Keep | Crate entry and shared attribute-language API; keep its independent meta/error subjects. |

### lk-agents-cli inventory

5 small files: 1 merge participants, 0 conditional, 4 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `src/lib.rs` | 39 | Keep | Required crate entry point and public composition surface; keep compact. |
| `src/logging.rs` | 50 | Keep | Distinct CLI feature: structured logging, remote dev transport, or isolated shutdown tests; keep separately navigable. |
| `src/remote_cli.rs` | 119 | Keep | Distinct CLI feature: structured logging, remote dev transport, or isolated shutdown tests; keep separately navigable. |
| `src/shutdown_tests.rs` | 86 | Keep | Sibling behavioral test suite or explicit integration target; preserve process isolation and subject naming. |
| `src/signals.rs` | 54 | Merge | `src/shutdown.rs`. Signal ownership, watchdog begin/release and shutdown completion are one process shutdown component (~220 lines); preserve singleton signal ownership and isolated shutdown integration tests. |

### lk-agents-codegen inventory

7 small files: 0 merge participants, 0 conditional, 7 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `src/decl/resolve/exposed.rs` | 147 | Keep | Named resolution responsibility for errors, exported functions and constants; parent already 1,096 lines. |
| `src/decl/resolve/native.rs` | 98 | Keep | Native-trait resolution is a distinct boundary contract; parent already 1,096 lines. |
| `src/decl/resolve/stream.rs` | 68 | Keep | Stream/duplex resolution is a distinct contract; merging into the parent leaves little headroom. |
| `src/python.rs` | 145 | Keep | Separate output phase for authored Python bases and constants. |
| `src/records/emit.rs` | 95 | Keep | Shared Rust-to-host emission used by records, classes and proxies; not an owner-local fragment. |
| `src/shape_error.rs` | 76 | Keep | Declared shape diagnostics shared by several emitters; adapter.rs is already 1,150 lines. |
| `src/surface.rs` | 122 | Keep | Separate aggregate export-table generation with its own assertions. |

### lk-agents-core inventory

57 small files: 6 merge participants, 4 conditional, 47 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `src/a2a/attachments.rs` | 56 | Keep | Distinct A2A responsibility/public protocol adapter; size alone does not justify combining auth, cards, policy, handler normalization or client tools. |
| `src/a2a/attachments_tests.rs` | 50 | Keep | Cargo-declared integration target; preserve external-crate contract and explicit process/test isolation. |
| `src/a2a/auth.rs` | 131 | Keep | Distinct A2A responsibility/public protocol adapter; size alone does not justify combining auth, cards, policy, handler normalization or client tools. |
| `src/a2a/card.rs` | 82 | Keep | Distinct A2A responsibility/public protocol adapter; size alone does not justify combining auth, cards, policy, handler normalization or client tools. |
| `src/a2a/channels_tests.rs` | 37 | Keep | Cargo-declared integration target; preserve external-crate contract and explicit process/test isolation. |
| `src/a2a/client/tool.rs` | 87 | Keep | Distinct A2A responsibility/public protocol adapter; size alone does not justify combining auth, cards, policy, handler normalization or client tools. |
| `src/a2a/handler.rs` | 87 | Keep | Distinct A2A responsibility/public protocol adapter; size alone does not justify combining auth, cards, policy, handler normalization or client tools. |
| `src/a2a/host.rs` | 122 | Keep | Distinct A2A responsibility/public protocol adapter; size alone does not justify combining auth, cards, policy, handler normalization or client tools. |
| `src/a2a/metadata_tests.rs` | 54 | Keep | Cargo-declared integration target; preserve external-crate contract and explicit process/test isolation. |
| `src/a2a/policy.rs` | 84 | Keep | Distinct A2A responsibility/public protocol adapter; size alone does not justify combining auth, cards, policy, handler normalization or client tools. |
| `src/a2a/task/effects.rs` | 11 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/a2a/task/mod.rs` | 11 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/agent.rs` | 124 | Keep | Recognizable shared Agent validation/output-schema feature used by voice and text. |
| `src/agent_run/effects.rs` | 125 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/agent_run/execution.rs` | 62 | Keep | Shared canonical identity/lifetime policy used across coordinators; keep discoverable with one semantic owner. |
| `src/agent_run/machine/checkpoint.rs` | 79 | Keep | Distinct managed-turn gate, coordinated-run contract or delegation composition seam; adjacent fragments implement different invariants. Parent1046 lines also limits safe absorption. |
| `src/agent_run/machine/coordinated.rs` | 108 | Keep | Distinct managed-turn gate, coordinated-run contract or delegation composition seam; adjacent fragments implement different invariants. Parent1046 lines also limits safe absorption. |
| `src/agent_run/machine/model_step.rs` | 71 | Conditional | `src/agent_run/machine/mod.rs`. Private model-request accumulator can be local, but parent already1046 lines; resulting~1117 leaves little growth room. Keep if coordinator readability suffers. |
| `src/agent_run/machine/text_session.rs` | 50 | Keep | Distinct managed-turn gate, coordinated-run contract or delegation composition seam; adjacent fragments implement different invariants. Parent1046 lines also limits safe absorption. |
| `src/agent_run/mod.rs` | 14 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/agent_run/result.rs` | 23 | Merge | `src/agent_run/mod.rs`. Five correlation ID declarations belong at the AgentRun public entry; result.rs contains no result implementation. |
| `src/assertions_tests.rs` | 104 | Keep | Meaningful sibling behavioral suite; retain test navigation and existing registration. |
| `src/effects.rs` | 17 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/lib.rs` | 48 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/llm/chat_dict.rs` | 69 | Keep | Serialization/filtering ABI adapter is distinct from history mutation/planning; do not collapse merely because both use ChatContext. |
| `src/llm/chat_plans.rs` | 73 | Merge | `src/llm/history.rs`. Host copy/merge/truncate adapter directly calls the history planners; keep planning and its ABI adapter together. |
| `src/llm/mod.rs` | 9 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/llm/provider_format/image.rs` | 42 | Merge | `src/llm/provider_format/mod.rs`. Pre-resolved lookup and byte projections are shared provider-format data/helpers already re-exported at this parent. |
| `src/llm/provider_format/mod.rs` | 150 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/llm/provider_format/testkit.rs` | 101 | Keep | Shared provider-format fixture module serves multiple provider suites. |
| `src/llm/provider_format/turns.rs` | 117 | Keep | Shared provider turn-merging algorithm is separately reusable from tool-call grouping. |
| `src/llm/schema/mod.rs` | 10 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/resource_lifecycle.rs` | 95 | Keep | Shared canonical identity/lifetime policy used across coordinators; keep discoverable with one semantic owner. |
| `src/telemetry/gen_ai.rs` | 79 | Keep | Distinct telemetry policy/record boundary: semantic naming, metric cardinality admission, records projection or W3C propagation. |
| `src/telemetry/metrics.rs` | 91 | Keep | Distinct telemetry policy/record boundary: semantic naming, metric cardinality admission, records projection or W3C propagation. |
| `src/telemetry/mod.rs` | 14 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/telemetry/records/mod.rs` | 42 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/telemetry/transport.rs` | 95 | Keep | Distinct telemetry policy/record boundary: semantic naming, metric cardinality admission, records projection or W3C propagation. |
| `src/text.rs` | 61 | Keep | Shared Python-compatible string rules used by several domains; no single component parent owns them. |
| `src/text_session/effects.rs` | 56 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/text_session/history.rs` | 75 | Merge | `src/text_session/machine/mod.rs`. Private frame-history selection/delivery helpers serve only the TextSession owner; keep ordinary helper functions local without making another owner. |
| `src/text_session/mod.rs` | 15 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/tokenize/host.rs` | 87 | Conditional | `src/tokenize/mod.rs`. 87-line native host adapter can join27-line parent; optional because host.rs is a meaningful adapter navigation seam. |
| `src/tokenize/mod.rs` | 27 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/tool_calls/effects.rs` | 60 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/tool_calls/mod.rs` | 10 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/toolsets/effects.rs` | 12 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/toolsets/mod.rs` | 7 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/tts/markup/chat.rs` | 62 | Keep | Recognizable expressive-markup subfeature (history edits, mood matching or tag conversion); keep algorithms independently navigable. |
| `src/tts/markup/host.rs` | 77 | Conditional | `src/tts/markup/mod.rs`. After absorbing xml49, host adapter126 can join parent126 (~252); optional, dedicated host adapter is also a useful seam. |
| `src/tts/markup/mod.rs` | 126 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/tts/markup/mood.rs` | 60 | Keep | Recognizable expressive-markup subfeature (history edits, mood matching or tag conversion); keep algorithms independently navigable. |
| `src/tts/markup/steering.rs` | 130 | Conditional | `src/tts/markup/instructions.rs`. Steering vocabulary/prompt construction and instructions share expressive-instruction feature (~326), but keep separate if provider-independent steering is navigated on its own. |
| `src/tts/markup/strip.rs` | 118 | Keep | Recognizable expressive-markup subfeature (history edits, mood matching or tag conversion); keep algorithms independently navigable. |
| `src/tts/markup/xml.rs` | 49 | Merge | `src/tts/markup/host.rs`. Only caller is the host markup utility; put XML rendering and its inline regression beside that call. |
| `src/tts/mod.rs` | 4 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/tts/transcript.rs` | 90 | Merge | `src/tts/mod.rs`. The TTS root is four lines and the file is one aligned-transcript normalizer; colocate at the TTS entry while retaining markup as child domain. |

### lk-agents-fuzz inventory

53 small files: 0 merge participants, 0 conditional, 53 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `build.rs` | 46 | Keep | Cargo build entry point and rule discovery; keep separate from runtime simulation code. |
| `build_registration.rs` | 118 | Keep | Build-time test registration checker, separately testable from rule-generation Cargo entry point. |
| `src/boundary.rs` | 87 | Keep | Explicit DST boundary/time/schedule contracts; a useful entry point independent of runner implementation. |
| `src/fakes/eot.rs` | 94 | Keep | A distinct fake provider boundary; EOT, interruption, VAD and vocabulary emulate different ports. Keep explicit fake subjects. |
| `src/fakes/interruption.rs` | 142 | Keep | A distinct fake provider boundary; EOT, interruption, VAD and vocabulary emulate different ports. Keep explicit fake subjects. |
| `src/fakes/vad.rs` | 137 | Keep | A distinct fake provider boundary; EOT, interruption, VAD and vocabulary emulate different ports. Keep explicit fake subjects. |
| `src/fakes/vocabulary.rs` | 100 | Keep | A distinct fake provider boundary; EOT, interruption, VAD and vocabulary emulate different ports. Keep explicit fake subjects. |
| `src/lib.rs` | 38 | Keep | Required crate entry point and public composition surface; keep compact. |
| `src/record/resources.rs` | 136 | Keep | Resource-lifetime view of recordings; separate from unrelated record views and observations. |
| `src/regressions_tests.rs` | 72 | Keep | Named behavioral suite with independent Cargo/registry discovery; keep sibling test boundary, do not inline by line count. |
| `src/rules/api_calls_return.rs` | 130 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/audio_conservation.rs` | 133 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/avatar_av_in_sync.rs` | 110 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/avatar_interrupt_clears_within_a_frame.rs` | 74 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/avatar_leaves_nothing.rs` | 75 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/avatar_playback_finished_once.rs` | 126 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/close_completes_at_its_bound.rs` | 114 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/close_finishes.rs` | 68 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/driver_exits_only_on_close.rs` | 63 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/eot_judged_by_the_detector.rs` | 65 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/every_resource_released.rs` | 63 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/every_task_finishes.rs` | 145 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/idle_is_cheap.rs` | 58 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/idle_respects_holds.rs` | 147 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/interrupted_handle_is_journalled.rs` | 79 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/mod.rs` | 83 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/no_premature_completion.rs` | 102 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/nothing_left_live.rs` | 92 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/on_exit_once_per_frame.rs` | 113 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/output_replacement_is_clean.rs` | 148 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/playback_requires_authorization.rs` | 131 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/provider_failure_is_accounted.rs` | 113 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/provider_streams_closed.rs` | 52 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/realtime_reply_cancelled_with_cause.rs` | 118 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/realtime_reply_reads_tool_output.rs` | 100 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/realtime_server_accepts_requests.rs` | 52 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/realtime_server_holds_what_was_heard.rs` | 77 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/refused_interrupt_is_no_interruption.rs` | 84 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/released_provider_unused.rs` | 91 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/reply_reaches_its_model.rs` | 135 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/shown_transcript_is_conserved.rs` | 140 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/speech_done_once.rs` | 130 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/speech_keeps_the_hosts_setting.rs` | 100 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/tool_call_arguments_are_json.rs` | 71 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/tool_lifecycle.rs` | 131 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/tool_updates_are_told.rs` | 127 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/turn_hook_is_awaited.rs` | 69 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/virtual_time_advances.rs` | 56 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/rules/voice_task_skips_parent_hooks.rs` | 119 | Keep | Deliberate one-guarantee-per-file registry: build.rs derives module and Rule type from filename; each rule owns its own examples/counterexamples. Preserve registration and contract discoverability. |
| `src/search/conversation/speaker.rs` | 134 | Keep | Separate simulated person/speaker actor with its own lifecycle; do not combine with unrelated actors. |
| `src/search/mod.rs` | 124 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/search/uniform.rs` | 113 | Keep | Uniform legal-move sampling is a distinct search algorithm used by conversation actors; parent is search facade and generator tests. |
| `src/soak_tests.rs` | 121 | Keep | Named behavioral suite with independent Cargo/registry discovery; keep sibling test boundary, do not inline by line count. |

### lk-agents-host-derive inventory

1 small files: 0 merge participants, 0 conditional, 1 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `src/table.rs` | 126 | Keep | Distinct native interface declaration-table generation; size alone does not justify mixing macro expansion concerns. |

### lk-agents-inference inventory

48 small files: 6 merge participants, 8 conditional, 34 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `src/amd/detector.rs` | 78 | Merge | `src/amd.rs`. Only child of amd; model selection and constants can live together in amd.rs (≈89). |
| `src/amd/mod.rs` | 11 | Merge | `src/amd.rs`. Only a facade over a single 78-line file. Collapse both to the leaf amd.rs (≈89), preserving amd public API. |
| `src/credentials.rs` | 90 | Keep | Credential resolution shared by inference constructors; do not bury in one provider. |
| `src/duplex/mod.rs` | 88 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/eot/eot.rs` | 56 | Merge | `src/eot/mod.rs`. 56-line types file loaded through #[path] adds naming/navigation indirection. Put domain values/constants in eot/mod.rs (42+56≈98). |
| `src/eot/mod.rs` | 42 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/eot/options.rs` | 63 | Conditional | `src/eot/detector.rs`. 63 lines of constructor selection can live with AudioEot creation (206+63≈269); retain module if externally consumed as its own API. |
| `src/eot/shell.rs` | 150 | Keep | Mounted transport owner and channel lifetime are separate from transport contract and EotSession decisions. |
| `src/eot/transport.rs` | 63 | Keep | Public fakeable transport/listener contract; deliberately separate from concrete shell/cloud/local implementations. |
| `src/evals/transcript.rs` | 79 | Keep | Transcript formatting shared by judge prompts, not a private incidental helper. |
| `src/extra.rs` | 14 | Keep | 14-line connection payload is shared by STT and TTS; keep one helper rather than duplicate or put under one provider (rename is separate scope). |
| `src/fallback/host_tests.rs` | 115 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |
| `src/fallback/mod.rs` | 100 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/fallback/realtime/backlog.rs` | 70 | Keep | Bounded replay queue with explicit byte accounting; clear component, not spare code in session. |
| `src/fallback/realtime/capabilities.rs` | 103 | Keep | Standalone capability compatibility contract with tests; separate from transport and fallback selection. |
| `src/fallback/tts/ports.rs` | 101 | Merge | `src/fallback/tts/mod.rs`. Only an ABI trait impl for adjacent FallbackTts/TokenLog; colocate with native owner (597+101≈698). |
| `src/gateway_tests.rs` | 124 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |
| `src/ids.rs` | 69 | Keep | Request ID creation shared across provider domains; independent utility. |
| `src/inference_tests.rs` | 47 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |
| `src/interruption/cache.rs` | 92 | Conditional | `src/interruption/machine.rs`. Single-owner bounded FIFO used only by InterruptionMachine; inline private cache and local tests (549+92≈641) if reducing navigation is preferred. |
| `src/interruption/detector.rs` | 69 | Conditional | `src/interruption/mod.rs`. Native detector is a small host-facing facade (46+69≈115); inline in domain root if keeping this layer explicit. |
| `src/interruption/mod.rs` | 46 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/interruption/proto.rs` | 121 | Keep | Wire protocol definitions/encoding are a meaningful boundary; keep out of transport lifecycle. |
| `src/interruption/retry.rs` | 75 | Conditional | `src/interruption/transport.rs`. 75-line transport error mapping can join its only executor (846+75≈921), retaining private pure helper functions. |
| `src/lib.rs` | 52 | Keep | Crate root/export map; required entrypoint, not fragmentation. |
| `src/llm/live_tests.rs` | 105 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |
| `src/llm/machine/mod.rs` | 6 | Keep | Domain machine namespace and ownership map; preserve visibility of synchronous effect-producing owners. |
| `src/llm/mod.rs` | 25 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/llm/thinking.rs` | 140 | Keep | Independent streaming thinking-token filter; preserve recognizable algorithm and tests. |
| `src/local_models.rs` | 114 | Keep | Native local-provider link registry/factory boundary shared by VAD and EOT. |
| `src/models.rs` | 137 | Keep | Shared ABI/model registry facade; distinct from model-specific lookup tables. |
| `src/realtime/context_tests.rs` | 95 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |
| `src/realtime/machine/context/server_items.rs` | 138 | Keep | Parent context/mod.rs already 1019 lines and helper is bounded incremental reconciliation; retain focused split. |
| `src/realtime/machine/effects.rs` | 23 | Keep | Explicit effects declaration/category boundary: deliberate exception to consolidation. |
| `src/realtime/machine/options.rs` | 138 | Keep | Shared realtime option data and patch contract, separate from live machine state. |
| `src/realtime/mod.rs` | 23 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/realtime/openai/encode.rs` | 138 | Keep | Typed effect-to-wire conversion at transport boundary, separate from machine and writer. |
| `src/realtime/openai/mod.rs` | 17 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/realtime/openai/session/retirement.rs` | 82 | Keep | Disposal owner outlives callers; keeping the shutdown/lifetime boundary explicit is valuable even at 82 lines. |
| `src/realtime/openai/testfix.rs` | 140 | Keep | Driver fixture serves many behavioral suites; preserve unless common primitive extraction avoids duplicated helpers. |
| `src/realtime/protocol.rs` | 110 | Keep | Typed inbound protocol; do not hide inside one decoder or machine. |
| `src/realtime/testfix.rs` | 22 | Conditional | `src/realtime/openai/testfix.rs`. Two fixture modules define msg/machine. Can share common helpers or relocate into one test fixture owner, but one produces pure RealtimeMachine and the other a driver wrapper: do not collapse these distinct return semantics blindly. |
| `src/stt/machine/mod.rs` | 7 | Keep | Domain machine namespace and ownership map; preserve visibility of synchronous effect-producing owners. |
| `src/tts/adapter.rs` | 56 | Merge | `src/tts/sentence_adapter.rs`. PreSplit/AdapterStreams are entry adapters of the sentence adapter; colocate (555+56≈611). |
| `src/tts/checks.rs` | 39 | Conditional | `src/tts/stream.rs`. Host synthesis validation and host stream wrapper can share one file (119+39≈158); keep validation as its declared trait implementation. |
| `src/tts/pacer.rs` | 100 | Conditional | `src/tts/sentence_adapter.rs`. SentencePacer belongs to sentence batching (555+56+100≈711 with AdapterStreams); keep separate if its public algorithm is used independently. |
| `src/tts/ports.rs` | 96 | Merge | `src/tts/mod.rs`. GatewayTts ABI impl only wraps its adjacent constructor/methods; colocate (671+96≈767). |
| `src/tts/stream.rs` | 119 | Conditional | `src/tts/stream.rs`. Destination of optional checks.rs merge, ≈158 total; no need to merge the stream with gateway wire/machine. |

### lk-agents-media inventory

16 small files: 3 merge participants, 2 conditional, 11 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `build/blingfire.rs` | 117 | Keep | Separate native toolchain/build feature from FFmpeg and cache utilities. |
| `build/cache.rs` | 133 | Conditional | `build/cache.rs`. Destination of optional settings merge; preserve shared cache ownership and process lock responsibilities. |
| `build/libraries.rs` | 10 | Merge | `build/mod.rs`. One 10-line archive lookup is used by this build entry/tests; inline beside FFmpeg linking (244+10≈254). |
| `build/native_cache_tests.rs` | 146 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |
| `build/settings.rs` | 99 | Conditional | `build/cache.rs`. Native compiler/cache key setup may join native cache utilities (133+99≈232); keep compiler identity and cache artifact code clearly named. |
| `src/emitter/effects.rs` | 16 | Keep | Explicit effects declaration/category boundary: deliberate exception to consolidation. |
| `src/emitter/mod.rs` | 8 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/error.rs` | 49 | Keep | Shared codec error/result contract used across codecs. |
| `src/ffmpeg/audio_encoder.rs` | 130 | Keep | Ogg/Opus encoding is separate from demux/decoder FFI lifetime; keep named implementation. |
| `src/ffmpeg/mod.rs` | 122 | Keep | FFI declarations/common error buffer plus codec module facade; meaningful unsafe boundary. |
| `src/images/mod.rs` | 19 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/images/prepare.rs` | 90 | Merge | `src/images/prepare.rs`. Destination of resolve.rs merge: image preparation, bounds, and resolver in ≈173 lines. |
| `src/images/resolve.rs` | 83 | Merge | `src/images/prepare.rs`. Images resolver is the host-frame entry of the same preparation pipeline; combine with prepare bounds/conversion (90+83≈173). |
| `src/lib.rs` | 80 | Keep | Crate root/export map; required entrypoint, not fragmentation. |
| `src/mime.rs` | 141 | Keep | Shared MIME/decoder-format lookup contract; separate from any one codec. |
| `src/pool_tests.rs` | 114 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |

### lk-agents-net inventory

8 small files: 0 merge participants, 0 conditional, 8 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `src/a2a_client/authority.rs` | 119 | Keep | Request authority validation is a distinct security boundary, with sibling authority tests. |
| `src/blocking_io.rs` | 21 | Keep | Process-wide blocking-I/O pool is an explicit architecture boundary, separate from DNS and media work. |
| `src/dns.rs` | 38 | Keep | Shared DNS/TCP resolver policy is an architecture-enforced entry point; do not bury in generic pool. |
| `src/env.rs` | 51 | Keep | ProcessEnv is the single injected process-services contract. |
| `src/lib.rs` | 35 | Keep | Required crate entry point and public composition surface; keep compact. |
| `src/session_report/admission.rs` | 93 | Keep | Process-wide preparation/upload/event quotas are shared across sessions; keep distinct from per-session projection. |
| `src/session_report/tagger.rs` | 140 | Keep | Standalone host-facing Tagger owner and clock adapter, distinct from session report upload. |
| `src/strings.rs` | 92 | Keep | Shared transport URL classification and ephemeral identifiers serve multiple network components. |

### lk-agents-py inventory

4 small files: 0 merge participants, 0 conditional, 4 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `build.rs` | 29 | Keep | Cargo build-script entry point, separate process and dependencies. |
| `src/lib.rs` | 70 | Keep | PyO3 module registration shell and crate root. |
| `src/py_bindings_tests.rs` | 53 | Keep | Declared Cargo integration target; keeps embedded Python contracts isolated. |
| `src/py_test_support.rs` | 94 | Keep | Shared loader for Cargo-linked Python tests; used by multiple suites. |

### lk-agents-runtime inventory

43 small files: 5 merge participants, 6 conditional, 32 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `src/a2a/attachments_tests.rs` | 125 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |
| `src/a2a/door/admission.rs` | 70 | Conditional | `src/a2a/door/mod.rs`. Private budget reservation is used by this door only; can inline private Budget/Admission types into door (417+70≈487), though keeping admission test visibility has value. |
| `src/a2a/door/stream.rs` | 98 | Keep | Durable publication/live observer backpressure has independent lifetime from door admission. |
| `src/a2a/turn_handle.rs` | 51 | Merge | `src/a2a/mod.rs`. The 51-line ABI impl only adapts the adjacent Turn owner; keep its implementation beside Turn (545+51≈596). |
| `src/avatar/media.rs` | 148 | Keep | Avatar track publication executor is separate from queue/playback flow. |
| `src/avatar/mod.rs` | 6 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/avatar/queue.rs` | 90 | Conditional | `src/avatar/output.rs`. Wakeup wrapper for one avatar audio output can live beside its consumer (600+90≈690); retain a distinct QueueAudio type/lifetime. |
| `src/host/cancellation.rs` | 31 | Merge | `src/host/mod.rs`. The concrete cancellation adapter belongs next to call_cancellable, not in a one-type child (170+31≈201). |
| `src/images_tests.rs` | 85 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |
| `src/lib.rs` | 82 | Keep | Crate root/export map; required entrypoint, not fragmentation. |
| `src/logging.rs` | 84 | Keep | Process-wide bounded log forwarding service, distinct from operation HostInvoker. |
| `src/mcp/lifecycle_tests.rs` | 133 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |
| `src/mcp/mod.rs` | 18 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/monotone.rs` | 66 | Keep | Small reusable level-triggered primitive with local tests; cross-feature utility, no single feature parent. |
| `src/output_validation.rs` | 79 | Keep | Shared projection used by run coordinators; preserve one policy projection owner across callers. |
| `src/room_io/detached.rs` | 72 | Keep | Distinct fakeable RoomIo implementation for detached sessions. |
| `src/room_io/output/segment_ids.rs` | 38 | Merge | `src/room_io/output/transcription.rs`. Transcript segment IDs are already re-exported by transcription; colocate its tiny ID helper/tests with that owner (450+38≈488). |
| `src/rtc/mod.rs` | 10 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/rtc/playout.rs` | 27 | Conditional | `src/rtc/mod.rs`. 27-line ABI wrapper could live in RTC facade (10+27≈37); keep type distinct and never merge async wrapper with voice ledger policy. |
| `src/rtc/synchronizer_tests.rs` | 144 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |
| `src/run_stream/cursor.rs` | 86 | Conditional | `src/run_stream/mod.rs`. Cursor is private to stream-consumer admission; may inline beside RunStream (296+86≈382), but preserve its drop-before-first-poll lifetime rule. |
| `src/run_stream/input.rs` | 99 | Conditional | `src/run_stream/mod.rs`. Two runner input adapters can sit alongside run-stream orchestration; combined with cursor group ≈481; retain Input trait as executor test boundary. |
| `src/tcp.rs` | 126 | Keep | Real reactor/socket boundary replaced by fakes; keep independently recognizable. |
| `src/toolsets/mod.rs` | 8 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/toolsets/scopes.rs` | 123 | Keep | Scope activation retains leases with a separate lock/drop contract; separate from group/lifetime executor. |
| `src/tts_live_tests.rs` | 74 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |
| `src/tts_tests.rs` | 41 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |
| `src/voice/audio.rs` | 140 | Keep | Audio/provider drain and camera forwarding feature; not the same owner as output composition. |
| `src/voice/events.rs` | 83 | Keep | Explicit resource-carrying executor input protocol; merging handlers would hide the boundary. |
| `src/voice/generation/mod.rs` | 75 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/voice/io.rs` | 55 | Keep | Owns output composition and synchronizer shutdown; do not merge with audio input/drain merely because both are small. |
| `src/voice/ivr.rs` | 46 | Keep | Feature-specific IVR effect executor, useful for finding what executes the machine. |
| `src/voice/keyterm_detection.rs` | 37 | Keep | Feature-specific inference executor; merging into AMD or generic audio would obscure ownership. |
| `src/voice/recorder.rs` | 101 | Keep | Distinct recording observer lifecycle; do not merge with general output composition. |
| `src/voice/remote_session/host.rs` | 29 | Merge | `src/voice/remote_session/mod.rs`. SessionHost is a 29-line wrapper of reply in the parent; put encoding and dispatch together (65+29≈94). |
| `src/voice/remote_session/mod.rs` | 65 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |
| `src/voice/session/owner_tests.rs` | 149 | Keep | Sibling behavioral suite or explicit Cargo integration target; retain separate tests/process isolation. |
| `src/voice/session/providers.rs` | 62 | Conditional | `src/voice/session/options.rs`. Provider retention and options updates are related (198+62≈260); merge only if lock/destructor boundary remains explicit. Do not add to session/mod.rs (1222). |
| `src/voice/session/storage.rs` | 142 | Keep | Distinct component or shared boundary; no same-owner merge that improves clarity was identified. |
| `src/voice/session/time.rs` | 92 | Keep | Clock/timeouts are a named injected boundary used across the runtime. |
| `src/voice/storage.rs` | 73 | Merge | `src/voice/session/storage.rs`. SessionStorage startup and StorageReporter runtime receipts are one voice storage feature; combine at session/storage.rs, keeping decisions in voice (142+73≈215). |
| `src/voice/workflows/dtmf.rs` | 88 | Keep | Distinct component or shared boundary; no same-owner merge that improves clarity was identified. |
| `src/voice/workflows/mod.rs` | 4 | Keep | Domain/component module facade with real children; retain the navigation and public export boundary. |

### lk-agents-storage inventory

4 small files: 0 merge participants, 0 conditional, 4 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `src/backends/agentdb/tasks_tests.rs` | 92 | Keep | Sibling behavioral test suite or explicit integration target; preserve process isolation and subject naming. |
| `src/backends/mod.rs` | 79 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/lib.rs` | 28 | Keep | Required crate entry point and public composition surface; keep compact. |
| `src/voice_tests.rs` | 107 | Keep | Sibling behavioral test suite or explicit integration target; preserve process isolation and subject naming. |

### lk-agents-voice inventory

57 small files: 8 merge participants, 4 conditional, 45 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `src/agent_activity/idle.rs` | 83 | Keep | Idle/waiter policy spans tools, turns and speech; do not fold into already1024-line speech scheduler. |
| `src/agent_activity/tool_results_tests.rs` | 90 | Merge | `src/agent_activity/tool_results.rs`. Small local two-test suite can be an inline cfg(test) module beside the private policy; retain assertions unchanged. |
| `src/agent_activity/video_sampler.rs` | 50 | Conditional | `src/agent_activity/io.rs`. 50-line video-admission predicate can join449-line activity I/O decisions; optional because sampling is independently reusable and tested through feature paths. |
| `src/agent_session/lifecycle.rs` | 141 | Keep | Lifecycle policy owner remains natural target for optional startup consolidation. |
| `src/agent_session/machine/initial.rs` | 140 | Conditional | `src/agent_session/machine/driver.rs`. Same VoiceDriver construction impl; optional flattening within existing machine boundary (~788 alone;871 with recognition). |
| `src/agent_session/machine/mod.rs` | 12 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/agent_session/machine/recognition.rs` | 83 | Conditional | `src/agent_session/machine/driver.rs`. 83-line recognition-effect dispatch is VoiceDriver coordination, not a new owner; optional flatten within machine (~731 alone;871 with initial). |
| `src/agent_session/nodes.rs` | 11 | Merge | `src/agent_session/mod.rs`. One-field OutputNodes snapshot is plain session data; put beside SessionParts/Outputs, keeping runtime live nodes separate. |
| `src/agent_session/room.rs` | 125 | Keep | Distinct session room-link/departure/deletion lifecycle owner. |
| `src/agent_session/storage.rs` | 123 | Keep | Distinct lane receipt/lease/commit policy; keep independent of room or startup mechanics. |
| `src/agent_session/time.rs` | 63 | Keep | Coherent supplied-clock/stamping policy shared by driver consumers; no ambient clock access. |
| `src/amd/effects.rs` | 23 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/amd/machine/mod.rs` | 2 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/amd/mod.rs` | 108 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/amd_tests.rs` | 95 | Keep | Cargo-declared integration target; preserve external-crate contract and explicit process/test isolation. |
| `src/audio.rs` | 145 | Keep | Small but complete independent audio gating, audio/video pacing or provider-overlay component; not a one-use helper fragment. |
| `src/audio_recognition/effects.rs` | 32 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/audio_recognition/observation.rs` | 23 | Keep | Explicit scoped event/diagnostic record boundary; keep distinct from effects and execution, as part of exceptions. |
| `src/av_sync.rs` | 138 | Keep | Small but complete independent audio gating, audio/video pacing or provider-overlay component; not a one-use helper fragment. |
| `src/avatar/effects.rs` | 14 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/avatar/machine/mod.rs` | 6 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/avatar/mod.rs` | 64 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/data_plane/effects.rs` | 18 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/data_plane/mod.rs` | 12 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/effects.rs` | 38 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/generation/effects.rs` | 145 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/generation/events.rs` | 12 | Keep | Explicit scoped event/diagnostic record boundary; keep distinct from effects and execution, as part of exceptions. |
| `src/generation/machine/llm.rs` | 120 | Keep | Different synchronous owners: streaming response accumulation versus reply startup/completion; keep visible within machine. |
| `src/generation/machine/mod.rs` | 15 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/generation/machine/pipeline.rs` | 136 | Keep | Different synchronous owners: streaming response accumulation versus reply startup/completion; keep visible within machine. |
| `src/ivr/effects.rs` | 8 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/ivr/mod.rs` | 3 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/ivr_tests.rs` | 39 | Keep | Cargo-declared integration target; preserve external-crate contract and explicit process/test isolation. |
| `src/keyterm_detection/effects.rs` | 9 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/keyterm_detection/mod.rs` | 4 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/lib.rs` | 43 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/provider_layers.rs` | 137 | Keep | Small but complete independent audio gating, audio/video pacing or provider-overlay component; not a one-use helper fragment. |
| `src/recorder_io.rs` | 31 | Merge | `src/agent_session/mod.rs`. Private 31-line recorder attachment state belongs beside other VoiceSessionState fields; no I/O or effect owner is moved. |
| `src/remote_session/console.rs` | 85 | Keep | Console-specific audio/playback protocol projection; preserve recognizable protocol seam. |
| `src/room_io/effects.rs` | 27 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/room_io/ingress.rs` | 46 | Merge | `src/room_io/mod.rs`. Small StreamBudget and three bounds are ordinary domain-level ingress data; parent currently five-line namespace. |
| `src/room_io/machine/mod.rs` | 12 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/room_io/machine/playout.rs` | 129 | Keep | Canonical PlayoutLedger shared by real SDK source adapter and DST fake; it is not merely an output.rs helper. |
| `src/room_io/mod.rs` | 5 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/run_view.rs` | 147 | Merge | `src/run_result.rs`. Voice run recording/output projection and live run settlement are the same run-result feature; colocate distinct types and preserve each contract. |
| `src/session_start.rs` | 85 | Conditional | `src/agent_session/lifecycle.rs`. Startup selection/lifetime facts can sit with lifecycle contracts (~226); retain distinct concepts and core OwnerLifecycle re-export, no new policy fusion. |
| `src/speech_handle_tests.rs` | 103 | Keep | Meaningful sibling behavioral suite; retain test navigation and existing registration. |
| `src/tokenize.rs` | 33 | Keep | Distinct component/contract with no clearly better same-owner target; retain rather than merge by line count. |
| `src/tool_body.rs` | 14 | Merge | `src/tool_executor.rs`. Two shared completion-policy functions belong to the existing sans-I/O tool owner; runtime imports new path, never duplicates policy. |
| `src/transcript_tests.rs` | 125 | Keep | Cargo-declared integration target; preserve external-crate contract and explicit process/test isolation. |
| `src/transcription/effects.rs` | 48 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/transcription/machine/mod.rs` | 6 | Keep | Required crate/domain namespace or machine producer map; small namespace files are deliberate. |
| `src/workflows/effects.rs` | 15 | Keep | Explicit effect declaration or crate effect-index boundary; protected by agreed architecture. |
| `src/workflows/end_call.rs` | 68 | Keep | Distinct built-in workflow or external Twilio protocol feature; keep its lifecycle/request rules localized. |
| `src/workflows/instructions.rs` | 17 | Merge | `src/workflows/mod.rs`. Two shared template helpers belong at common workflow entry; rendering algorithm remains generation::safe_render. |
| `src/workflows/prompts.rs` | 55 | Merge | `src/workflows/dtmf.rs` + `src/workflows/warm_transfer.rs`. Each nested constant group belongs beside its sole workflow feature; update warm-transfer runtime import too, no second prompt copy. |
| `src/workflows/twilio.rs` | 149 | Keep | Distinct built-in workflow or external Twilio protocol feature; keep its lifecycle/request rules localized. |

### lk-agents-worker inventory

10 small files: 0 merge participants, 1 conditional, 9 keep.

| File | Lines | Decision | Destination / reason |
| --- | ---: | --- | --- |
| `src/config.rs` | 142 | Keep | Worker configuration conversion and validation is a public feature boundary. |
| `src/hw/cgroup.rs` | 126 | Keep | Container CPU quota/cgroup detection is distinct from CPU sampling and hardware path lookup. |
| `src/hw/mod.rs` | 8 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/ipc/executor.rs` | 95 | Keep | Shared Runner/RunnerExecutor contract and thread implementation used by process and thread supervision. |
| `src/ipc/mod.rs` | 49 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/ipc/wire.rs` | 73 | Keep | Versioned parent/child IPC message protocol; keep wire contracts visible. |
| `src/lib.rs` | 39 | Keep | Required crate entry point and public composition surface; keep compact. |
| `src/server/health.rs` | 134 | Keep | Explicit worker HTTP health endpoint and deadline handling; separate from server orchestration. |
| `src/transport/mod.rs` | 137 | Keep | Existing category/namespace entry point with children or public composition contracts; do not delete while children remain. |
| `src/worker/load_driver.rs` | 114 | Conditional | `src/worker/mod.rs`. This is only impl Worker methods using parent-private load fields and completion variants; merging removes one owner-navigation hop (~1078 lines). Optional because only ~172 lines headroom remain under1250. Keep load.rs decision model separate. |

## Verification

For each batch, update imports, module documentation and ownership maps; preserve visibility,
feature gates, invocation identity, lock/drop ordering and Cargo target isolation. Regenerate
bindings through codegen for declaration moves, never by editing generated output. Run affected
Rust/Python contracts, architecture checks (including file length and boundary enforcement),
codegen drift checks and formatting. File-only moves do not justify changing scheduling,
approval, cancellation, playout or delegation behavior.

Verified the original 438-entry inventory, all 65 retired source files, their final destinations,
and the six deliberately retained conditional source files (plus the unchanged cache destination).

Execution checks:

- ABI: 69 unit tests; core: 434 unit tests and five A2A contracts.
- Voice: 221 unit tests plus AMD, IVR and transcript-pacing integration targets.
- Inference: 487 unit tests and its local transport/context suites; media: 166 unit tests and
  nine isolated native-cache tests. CLI shutdown tests retain their process isolation.
- Runtime: 182 unit tests plus local A2A, EOT, image, MCP and speech-text integration targets.
  MCP fixtures explicitly use the repository Python through `MCP_TEST_PYTHON`.
- DST: all 210 production regressions, rule-registration checks and ordinary support tests.
- Cargo-linked Python: 143 contracts, plus native binding, adapter, conversion and reference
  tests. Generated export names are preserved. Existing live/soak tests remain ignored.
- Complete architecture gate: workspace compilation and Clippy, default/all-feature purity
  and I/O policies, dependency and source boundaries, file lengths, export placement and
  generated drift all pass. Zero warnings in the 15 warning-clean crates; no exemptions added.
- Documentation: Rustdoc builds for ABI, core, voice, inference, media and runtime with broken
  intra-doc links treated as errors; local Markdown links and `git diff --check` pass.
