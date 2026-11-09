# Python framework ownership inventory

This file complements the [Rust core architecture](rust-core.md) with file-level ownership.
A listed file does not imply that every method or cached Python object belongs in Rust.

Its target groups handwritten implementations while preserving root application exports.

This inventory covers every handwritten `.py` file under `livekit-agents-next/livekit/agents`. Provider plugin packages are intentionally Python and are outside this inventory. Generated implementations, extension stubs and tests are excluded. Line counts are a snapshot, not a measure of remaining framework logic.

Snapshot (2026-11-08): 59 files, 15,834 lines. Native delegation and Python object views are intentional. Completion requires one policy/lifetime owner, not removal of every Python caller.

## Ownership contract

Rust owns framework policy and resource lifetimes. Synchronous decisions belong in core/voice/media; runtime executes effects through typed capabilities. Existing async I/O crates own their transports, providers, storage and supervision. Python retains language reflection, Pydantic hydration, invocation-local contextvars, user code and actual host SDK calls. ABI declarations generate routine signatures, records, proxies, views and forwarding; arbitrary Python programs embedded in metadata are not a generation strategy.

Toolset setup failure aborts initialization in both text and voice. Acquired resources settle before failure escapes. RTC media uses one shared capability boundary and the existing room connection. No alternate SDK connection or avatar-specific host manager.

## File coverage

### Public exports and Python packaging

Keep Python imports, lazy module loading, entrypoint and package metadata. Generate re-exports where ABI declarations define them.

| File | Lines |
| --- | ---: |
| [__init__.py](../../livekit-agents-next/livekit/agents/__init__.py) | 254 |
| [__main__.py](../../livekit-agents-next/livekit/agents/__main__.py) | 43 |
| [http/__init__.py](../../livekit-agents-next/livekit/agents/http/__init__.py) | 26 |
| [inference/__init__.py](../../livekit-agents-next/livekit/agents/inference/__init__.py) | 269 |
| [llm/__init__.py](../../livekit-agents-next/livekit/agents/llm/__init__.py) | 138 |
| [plugin.py](../../livekit-agents-next/livekit/agents/plugin.py) | 44 |
| [storage/__init__.py](../../livekit-agents-next/livekit/agents/storage/__init__.py) | 198 |
| [utils/__init__.py](../../livekit-agents-next/livekit/agents/utils/__init__.py) | 210 |
| [version.py](../../livekit-agents-next/livekit/agents/version.py) | 15 |

### Boundary generation

Generate record conversion, callback dispatch, views and forwarding from ABI declarations. Keep only Python SDK/object conversion and context capture handwritten. A proxy calls the host object's own methods and reads its attributes (`name = "a.b"`, `getter = "a.b"`, `#[host(kw)]`, `sdk` enum numbers; rust-core.md "Declarations and placement"): no handwritten class renames `_core_x` to `obj.x`. The io legs (`io.AudioInput`, `VideoInput`, `AudioOutput`, `TextOutput`), the SDK stream writers and video source are called directly; `VideoInput` yields `rtc.VideoFrameEvent`, pulled as an `await, iter` proxy; `_bindings.py` keeps only `next_audio_frame` and `audio_output_capabilities`, declared as `function = ".."`.

| File | Lines |
| --- | ---: |
| [llm/chat_context.py](../../livekit-agents-next/livekit/agents/llm/chat_context.py) | 348 |
| [types.py](../../livekit-agents-next/livekit/agents/types.py) | 15 |

### Provider execution

Native inference owns attempts, retry/fallback selection and stream admission. `ProviderAttempts` (lk-agents-inference `provider.rs`) executes a host provider's attempts too, standalone or in a session: it runs the host's one-attempt `_core_attempt`, its deadlines, retries and error events; Python implements `_run` and the SDK feed/drain tasks, with no retry loop of its own. The TTS sentence adapter (`_core.TtsStreamAdapter`) and `SentenceStreamPacer` (`_core_PacedSentences`) split, pace and synthesize natively, their timers and reads included; a custom tokenizer's `SentenceStream` is read through `_core_next`. Shared aio helpers retain unfinished cleanup; native close proceeds independently of stalled host input cleanup. Generated provider APIs remain supported.

| File | Lines |
| --- | ---: |
| [_bindings.py](../../livekit-agents-next/livekit/agents/_bindings.py) | 654 |
| [inference/llm.py](../../livekit-agents-next/livekit/agents/inference/llm.py) | 186 |
| [inference/realtime.py](../../livekit-agents-next/livekit/agents/inference/realtime.py) | 102 |
| [inference/stt.py](../../livekit-agents-next/livekit/agents/inference/stt.py) | 46 |
| [inference/tts.py](../../livekit-agents-next/livekit/agents/inference/tts.py) | 44 |
| [llm/_realtime.py](../../livekit-agents-next/livekit/agents/llm/_realtime.py) | 93 |
| [llm/duplex.py](../../livekit-agents-next/livekit/agents/llm/duplex.py) | 215 |
| [llm/realtime.py](../../livekit-agents-next/livekit/agents/llm/realtime.py) | 32 |
| [stt.py](../../livekit-agents-next/livekit/agents/stt.py) | 73 |
| [tts.py](../../livekit-agents-next/livekit/agents/tts.py) | 120 |

### Native inference services

Keep Python SDK object hydration and public authoring options; generate options and forward lifecycle/configuration to inference. Check for duplicate provider selection, credentials and lifecycle decisions.

| File | Lines |
| --- | ---: |
| [inference/avatar.py](../../livekit-agents-next/livekit/agents/inference/avatar.py) | 153 |

### Agent configuration and authoring

Core owns configuration validation, freeze checks, hook replacement and retry resolution. Python keeps callable inspection, decorators, module-name discovery, Pydantic validation, InstructionsView and contextvars. Generated records/facades expose the native configuration.

| File | Lines |
| --- | ---: |
| [llm/utils.py](../../livekit-agents-next/livekit/agents/llm/utils.py) | 483 |
| [voice/__init__.py](../../livekit-agents-next/livekit/agents/voice/__init__.py) | 693 |

### Shared toolset lifecycle

Core/runtime own acquisition identity, reference counts, concurrent setup, rollback, cancellation and last-user close. Search/proxy toolsets declare hidden resource dependencies to that group. Python retains authored setup/aclose callbacks, tool identity and schema inspection.

| File | Lines |
| --- | ---: |
| [llm/mcp.py](../../livekit-agents-next/livekit/agents/llm/mcp.py) | 292 |
| [llm/tool_context.py](../../livekit-agents-next/livekit/agents/llm/tool_context.py) | 592 |

### Run and text lifecycle

Runtime RunResources owns thread/context/toolset acquisition, rollback and release. Native RunStream owns stream admission, cancellation and settlement. Python retains typed result/exception hydration and invocation-local context installation.

| File | Lines |
| --- | ---: |
| [text_session.py](../../livekit-agents-next/livekit/agents/text_session.py) | 172 |
| [voice/agent_run.py](../../livekit-agents-next/livekit/agents/voice/agent_run.py) | 742 |

### Voice session construction and lifecycle

Voice owns decisions; runtime SessionOwner owns startup sequencing, acquisition, rollback and close. SessionHost constructs Python SDK/models/context views through explicit callbacks. Event/hook dispatch and Python object identity remain adapters, not a second session owner.

| File | Lines |
| --- | ---: |
| [voice/_session_bindings.py](../../livekit-agents-next/livekit/agents/voice/_session_bindings.py) | 547 |
| [voice/_session_callbacks.py](../../livekit-agents-next/livekit/agents/voice/_session_callbacks.py) | 514 |
| [voice/agent_session.py](../../livekit-agents-next/livekit/agents/voice/agent_session.py) | 513 |

### Voice results and events

Use native authoritative state and generated projections for run/event/playback lifecycle. Keep Python assertion/evaluation ergonomics, type hydration and user callback invocation.

| File | Lines |
| --- | ---: |
| [voice/events.py](../../livekit-agents-next/livekit/agents/voice/events.py) | 236 |
| [voice/io.py](../../livekit-agents-next/livekit/agents/voice/io.py) | 471 |
| [voice/run_result.py](../../livekit-agents-next/livekit/agents/voice/run_result.py) | 500 |

### RTC SDK boundary

Runtime owns room orchestration and playback accounting through the shared RTC capability boundary. Python retains direct SDK calls, frame/object translation, publication options and callback registration. Generated boundary extraction reads resolved RoomOptions directly; no intermediate field dictionary is maintained.

| File | Lines |
| --- | ---: |
| [voice/room_io.py](../../livekit-agents-next/livekit/agents/voice/room_io.py) | 356 |

### Avatar media and transport

Runtime reuses RTC source/publication capabilities and owns pacing, bounded frame delivery, interruption accounting and cleanup. Python retains SDK frame conversion and user generator/receiver callbacks; the retired avatar-specific media host protocol is not needed.

| File | Lines |
| --- | ---: |
| [voice/avatar.py](../../livekit-agents-next/livekit/agents/voice/avatar.py) | 351 |

### Voice auxiliary lifecycle

Native voice/runtime own AMD admission, playback selection and remote-session decisions. Python retains completion/result views, user callbacks and SDK sources/sinks. Rejected AMD attachment cannot replace or close the active detector.

| File | Lines |
| --- | ---: |
| [voice/amd.py](../../livekit-agents-next/livekit/agents/voice/amd.py) | 248 |
| [voice/background_audio.py](../../livekit-agents-next/livekit/agents/voice/background_audio.py) | 118 |

### Workflows and tool discovery

Core/voice own workflow ordering, prompt plans and tool-search ranking. Python retains authored tool bodies, domain validation, reflection and SDK calls. Toolset dependencies use the shared native group even when the visible search/proxy tool list hides a child.

| File | Lines |
| --- | ---: |
| [beta/__init__.py](../../livekit-agents-next/livekit/agents/beta/__init__.py) | 346 |
| [beta/workflows.py](../../livekit-agents-next/livekit/agents/beta/workflows.py) | 411 |

### Storage hydration and execution

Native storage/runtime own transactions, revisions, thread resolution and run-resource lifetimes. Python retains Pydantic/dataclass hydration and application services. `ctx.thread` (`Thread`) extends the generated `Thread` facade over the native `_core_ThreadHandle`, which opens the thread's database for each read and releases it; Python keeps only the `ChatContext` hydration of `chat_history()` and `Lane.history()`. An `a2a_session` handler is one awaited host call (`_TurnHost._core_handle`); its checkpoints, progress and tokens cross through the native `_core_HandlerTurn`, whose bounded queue, verdicts, cancellation and close settlement the runtime owns.

| File | Lines |
| --- | ---: |
| [storage/context.py](../../livekit-agents-next/livekit/agents/storage/context.py) | 253 |
| [storage/session.py](../../livekit-agents-next/livekit/agents/storage/session.py) | 161 |

### A2A and HTTP adapters

Keep ASGI request/response adaptation for the application's HTTP routes and authored route/handler registration. A2A endpoints are served natively: the worker registers each endpoint's `/<endpoint>/{path:path}` route as an ordinary HTTP route and hands its exchanges to the runtime's A2A door, which owns route matching, authentication, the payload budget, extension negotiation, the card and JSON/SSE framing; Python supplies only the handler. Core/net/runtime own authentication decisions, routing state, exchange lifecycle and cancellation; generate configuration and native projections.

| File | Lines |
| --- | ---: |
| [a2a.py](../../livekit-agents-next/livekit/agents/a2a.py) | 330 |
| [http/server.py](../../livekit-agents-next/livekit/agents/http/server.py) | 126 |

### Process and worker lifecycle

Worker/runtime own registration, leases, job state, process supervision, shutdown/report ordering and timeouts. Keep Python multiprocessing bootstrap, interpreter setup, import/preload operations, SDK room creation and entrypoint callbacks behind narrow ports. The job processes' port is `ipc.JobLauncher` (start the process or thread, the start method's bookkeeping, logging); each process's state (status, running job, pid, exit code) is the native `JobSupervisor`'s ([rust-core](rust-core.md#worker-and-job-processes)).

| File | Lines |
| --- | ---: |
| [cli.py](../../livekit-agents-next/livekit/agents/cli.py) | 315 |
| [ipc/__init__.py](../../livekit-agents-next/livekit/agents/ipc/__init__.py) | 242 |
| [ipc/_preload.py](../../livekit-agents-next/livekit/agents/ipc/_preload.py) | 80 |
| [ipc/_preload_forkserver.py](../../livekit-agents-next/livekit/agents/ipc/_preload_forkserver.py) | 49 |
| [ipc/child.py](../../livekit-agents-next/livekit/agents/ipc/child.py) | 365 |
| [job.py](../../livekit-agents-next/livekit/agents/job.py) | 684 |
| [worker.py](../../livekit-agents-next/livekit/agents/worker.py) | 364 |

### Telemetry host integration

Rust owns spans, metrics, report records, export and the session report: its assembly from the session owner, the console's `session_report.json` and the upload ([telemetry internals](telemetry.md)); `JobContext` passes only the job's identity, the report's events and the caller's history. Python keeps one module: the OpenTelemetry API provider over the native span, the bridge for an application's SDK provider, `configure`, `set_capture_content` and `diagnostics`, the event loop monitor and the framework's spans around Python-only code (`room_connect`, `wait_for_participant`); `rpc_call` and `rpc_handler` are recorded natively by the job's `_core_RpcTracing` interceptor.

| File | Lines |
| --- | ---: |
| [log.py](../../livekit-agents-next/livekit/agents/log.py) | 95 |
| [metrics.py](../../livekit-agents-next/livekit/agents/metrics.py) | 4 |
| [telemetry.py](../../livekit-agents-next/livekit/agents/telemetry.py) | 204 |

### Evaluation and simulation

Inference/core own judge execution, aggregation policy and finalize state. Keep application verdicts, authored evaluator callbacks and Python type hydration; generate projections/forwarding.

| File | Lines |
| --- | ---: |
| [evals.py](../../livekit-agents-next/livekit/agents/evals.py) | 316 |

### Media and tokenization adapters

Keep public Python iterator/callback interfaces. Media/core own buffering, framing, decode, tokenization and resource state; runtime owns pumps. Keep optional Python library calls (for example blingfire) as provider capabilities.

| File | Lines |
| --- | ---: |
| [tokenize.py](../../livekit-agents-next/livekit/agents/tokenize.py) | 253 |
| [utils/media.py](../../livekit-agents-next/livekit/agents/utils/media.py) | 75 |

### Public Python utilities

Python-facing channels, asyncio task references and aiohttp hooks remain host adapters. Shared cancellation retains tasks until settlement and exposes pending cleanup. Native ConnectionPoolState owns reuse/retirement; Python holds actual SDK connections. Generic user-data containers and dynamic user keys remain valid.

| File | Lines |
| --- | ---: |
| [http/client.py](../../livekit-agents-next/livekit/agents/http/client.py) | 577 |
| [utils/aio.py](../../livekit-agents-next/livekit/agents/utils/aio.py) | 478 |

## Kept handwritten

Classes reviewed for generation and kept in Python, each for the reason given:

- `voice.io.AgentInput` / `AgentOutput`: legs set before `start()` are host objects held until the running session exists; the native setters and `set_io_enabled(IoKind)` apply them after.
- `VoiceSession.history`, `agent_state`, `user_state`: read the state delivered to the event being handled, an invocation-local contextvar, else the owner's latest.
- `voice.run_result` `RunAssert` / `EventAssert` / `EventRangeAssert` / `*Assert`: pytest ergonomics (typed overloads, slices, `__tracebackhide__`, a class per event kind) over the host items; native `RunEvents` owns matching and failure text.
- `voice.background_audio.BackgroundAudioPlayer`: binds the SDK room and its publication, subscribes to the session's `agent_state_changed` and warns in console jobs; the native player owns mixing, selection and playback.
- `beta.KeywordSearchStrategy` / `BM25SearchStrategy`: implement the public `SearchStrategy` protocol over host `SearchItem`s (index data stored on the item, results the caller's items); native code ranks.
- `tokenize.BufferedTokenStream` / `BufferedSentenceStream` / `BufferedWordStream`: an asyncio channel and segment ids around the synchronous core `TokenBuffer`; the public constructor is what custom tokenizers (nltk) build.
- `ipc.child` `_core_ping`: an empty coroutine; the measurement is the host event loop running it.
- `A2AEndpoint._door()`: assembles the native `_core_A2ADoor` over the Python handler, its thread and the server's late-bound agent name (the former `A2AService`).

## Completion criteria

- Remove the Python owner and duplicate state when migrating a responsibility; splitting files does not count.
- Shared Agent and toolset contracts serve both text and voice.
- Test failure, cancellation, concurrent reuse, rollback and close at native boundaries, plus public Python compatibility.
- Run generation checks, architectural checks and relevant Python/Rust suites. Keep this inventory honest about unfinished work.

## Migration ledger

Implemented ownership changes (validation is recorded with the change):

- Avatar media uses shared RTC source/publication capabilities. `_AvatarMedia` and its host proxy are removed. Voice owns the pacing plan; runtime owns publication, bounded frame delivery and cleanup. The generated native synchronizer preserves the operations exposed by `runner.av_sync`; its concrete type is native rather than `rtc.AVSynchronizer`.
- Queue audio admission, segment epochs, discarded-segment ordering and stale playback report handling are in `voice::avatar::queue`. Python forwards frames and executes playback event effects; it owns no segment queue.
- An `AvatarSession`'s lifecycle is the runtime's `_core_AvatarLifecycle`: the join wait and `wait_for_join`'s timeout, the join metric, the playback-latency metric (from each assistant `ItemAdded` the session owner delivers while the avatar is attached), the avatar's `aclose` registered on the job's shutdown as a host framework shutdown on the job's native run (`SessionJob.run`, the `_run` getter; in the shutdown's span), and the participant's removal on close through `lk-agents-net`'s RoomService client with the job's `_server_env` credentials (`not_found` skipped, any other failure logged). The Python join task, `conversation_item_added` listener, `_emit_metrics` and the `livekit.api` removal call are removed; Python keeps the audio output's ownership and emits `metrics_collected`.
- Shared toolset lifetime and group/scope selection are in `core::toolsets`; runtime owns setup/close effects, cancellation and rollback. Python refcounts, locks and separate text/voice acquisition/rollback loops are removed. Object identity, rather than the public toolset label, identifies a shared connection.
- Voice startup, handoff rollback, per-agent release and session shutdown address the native toolset registry directly. Python provides SDK toolset identities and schema reflection; `_core_setup_tools`, `_core_close_toolsets` and the bootstrap abort callback are removed. Failed or cancelled admissions do not occupy a later activation’s release slot.
- Owned toolset setup uses generated HostInvoker cancellation signalling and waits for the Python task’s actual settlement, including async `finally` blocks.
- Direct agent runs and whole text-session turns use one native `RunResources` lifetime for temporary thread storage, context activation and toolset leases. Python ownership flags, per-agent acquisition/release loops and `AsyncExitStack` cleanup are removed. Borrowed application resources remain borrowed; native close ends owned activations and releases Python hydration through a narrow SDK callback. Text and voice share core start/close admission.
- Request streams now bind directly to native runner sources. Core owns consumer/read admission, text snapshots, event envelopes and presentation acknowledgments; runtime owns cancellation and source/resource settlement. Public event records and cursor bindings are generated. Python retains result/exception hydration and language protocol forwarding, with no stream scheduler, consumer flags, text buffer or release map.
- Native cleanup failures are cached as the public stream’s Python exception outcome. Repeated result retrieval and context exit do not invoke cleanup again or consume a captured exception twice.
- Agent authoring configuration, callback ID order and hook replacement, freeze checks, retry resolution and configuration projection are in `core::agent_authoring`. `AgentRetries` and `ModelSettings` are generated. Python keeps callable registries, module-name discovery, Pydantic/schema reflection and callback invocation.
- `session.history` reads the runtime's delivered session state (`_core_SessionState`): the host drain applies each added or reloaded item as it delivers the event, holding the Python objects it asked the host for, and hands every event its state. Python's history copy and its item-added, output-replacement and reload mirroring are removed.
- `session.agent_state` and `user_state` read the same delivered state; the generated `MIRRORED_STATES` attribute mirroring and its codegen route are removed. A `SpeechHandle`'s done callbacks wait on the native `wait_reported()`, which the host drain resolves once it has delivered the speech's `speech_done` (and the close resolves for every speech still held); the session's Python handle registry and `_mark_done` bookkeeping are removed.
- `session.stt`, `tts` and `vad` read the native options' effective provider (`SessionOptions.provider`, topmost override else base); the Python provider attributes, their re-assignment on every update, override and release, and the `_using_default_vad` flag (now derived from the default VAD object) are removed.
- `say`, `generate_reply` and the session's other commands are methods of the generated `VoiceSession` facade (`#[host(facade)]`: `not_given`, the caller's `origin`, `returns`, per-argument `annotation` and `resolve`) calling the native session once; `say` is one native command over `str | AsyncIterable[str]` with optional `audio`, and a hook's origin carries its name, so the core decides an `on_enter` reply. `SessionAdapter.say`/`generate_reply` and the `_hook_name` context variable are removed.
- `update_agent` calls the native owner directly; the history it names before `start()` is the owner's (`_start_chat_ctx` and `SessionAdapter.update_agent` are removed).
- `update_options` is a method of the generated `VoiceSession` facade (a `Maybe` argument is `NotGivenOr[T] = NOT_GIVEN`, sent as given) calling the native owner once with flattened keywords; the owner validates (the interruption `mode`, even `None`, is refused), applies the settings and provider layers and installs only the changed providers, asking the host's `SessionHost.bind_providers` for their legs under the current agent's nodes. `override_options` is a small context manager over the owner's `override_options` and `release_override`, which install natively too. `SessionAdapter.configure`/`override`/`_install_models`, the `SessionOverride` record and the `install_providers` export are removed. The interruption detector's gate reads the native options' `turn_detection` and `interruption_mode`; the `_turn_detection_mode` and `_interruption_mode` session attributes are removed.
- A job's session startup is the runtime `SessionOwner`'s: after `SessionHost.prepare` (which keeps the console claim, the room and the legs, and names the job in `args.job`), the owner claims the job's primary session, plans and initializes the job's recording, decides hosting and the job run, sets the start time, opens the audio recorder in the job's session directory and registers the session's native close on the job's shutdown (`JobExecution::add_framework_shutdown` on the job's native run), through the `SessionJob` proxy over `JobContext` (`_core_claim_primary`, `job.enable_recording`, `_core_recording_dir`, `init_recording`, `_run`). The owner holds what the report reads (`started_at`, `recording_options`, `recording_path`, `recording_started_at`) and the report's event retention (`allows_report`, `accepts_event`, `admit_event`); the session's `_started_at`, `_recording_options`, `_recording_path`, `_recorder` and `_event_retention` attributes, `JobContext._claim_session` and the `plan_session_start` export are removed.
- `generate_reply(tools=)` sends the names; the voice core narrows the driving agent's installed toolset (`Toolset::narrowed`). Python's pool reflection, per-reply agent frame and the `reply_tool_indices` export are removed.
- Lifecycle hooks and session views carry the agent scope identity separately from its name. Distinct agents sharing a name retain their own hooks and resources; failed activation does not replace the current agent.
- `utils.media.AudioStreamDecoder` is the native class itself (`_core.AudioStreamDecoder`): a failure to open, push or decode is logged natively and ends the stream; the Python wrapper and `CodecFailure` are removed.
- `session.stt`, `tts`, `vad` and `current_agent` are generated facade getters (the native `SessionOptions.stt`/`tts`/`vad`, replacing `provider(slot)`; `current_agent` refuses with `SessionRefused::NotStarted`).

The inventory is **not an assertion that the whole Python framework has been migrated**.
Source-reviewed responsibilities above deliberately retain SDK, reflection and hydration code.
The RoomOptions dictionary projection, AuthGrant/Principal constructor projections and fixed
provider member probes have been removed. The lasting contracts are documented under
[shared binding semantics](rust-core.md#shared-binding-semantics).
Provider review is sampled; no claim is made that every plugin has been manually certified.

## Validation snapshot

The table covers all handwritten framework files exactly once. Behavior changes are checked
through Cargo's freshly linked Python contracts plus native ownership tests. Current execution,
transport, report and MCP changes have scoped strict mypy checks. Architecture/code-generation
checks remain separate from behavioral tests. Historical whole-repository test counts are not
presented as validation of this checkout.
