# Telemetry reference

What the framework emits and what each consumer reads. The design and the API
that produce it are in [telemetry internals](telemetry.md). The declarations in
`lk-agents-abi/src/telemetry/` are the source of truth; `otel/schema.json` is
generated from them and this page describes it.

## Standard conventions

Every model request, tool call and agent invocation carries the OpenTelemetry
GenAI semantic conventions, so any GenAI-aware backend (Langfuse, Datadog LLM
Observability, Logfire, Arize) understands it without LiveKit-specific setup.
The framework span names stay those the Cloud session view renders; the
conventions live in the attributes, the span kind and the metrics.

| Work | Span | `gen_ai.operation.name` | Attributes | Metrics |
| --- | --- | --- | --- | --- |
| LLM request | `llm_request` (kind `CLIENT`) | `chat` | `gen_ai.provider.name`, `gen_ai.request.model` and request parameters, `gen_ai.response.model`, `gen_ai.response.id`, `gen_ai.response.finish_reasons`, `gen_ai.usage.*`, `gen_ai.response.time_to_first_chunk`; with message capture `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.definitions` | `gen_ai.client.token.usage`, `gen_ai.client.operation.duration` |
| Realtime model turn | `realtime_inference` (kind `CLIENT`) | `generate_content` | as an LLM request, with audio and text token splits | as an LLM request |
| Tool call | `function_tool` | `execute_tool` | `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type`, `gen_ai.tool.description`; with message capture `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` | `gen_ai.client.operation.duration` |
| Agent turn | `agent_turn` | `invoke_agent` | `gen_ai.agent.name`, `gen_ai.conversation.id` | `gen_ai.client.operation.duration` |
| Session | `agent_session` | `invoke_workflow` | `gen_ai.workflow.name`, `gen_ai.conversation.id` | |
| Agent activation | `start_agent_activity` | `create_agent` | `gen_ai.agent.name`, `gen_ai.provider.name`, `gen_ai.request.model` | |

Speech work has no OpenTelemetry convention yet. It follows the same shape
under LiveKit keys registered in this registry, so a backend can treat every
model call alike:

| Work | Span | Shape |
| --- | --- | --- |
| STT stream | `user_turn` carries the model; requests and reconnects as events | `gen_ai.provider.name`, `gen_ai.request.model`, `lk.stt.*`, audio duration usage |
| TTS request | `tts_request` (kind `CLIENT`), `tts_request_run` per attempt | `gen_ai.provider.name`, `gen_ai.request.model`, `lk.tts.*`, `lk.response.ttfb`, character and audio duration usage |
| End-of-turn prediction | `eou_detection` | provider and model of the detector, probability, threshold, delay |
| VAD | no span per inference; `lk.agents.vad.*` metrics | inference duration and count |

The LiveKit metric instruments (`lk.agents.turn.*`, `lk.agents.usage.*`) are
recorded alongside the GenAI ones; Cloud reads the LiveKit ones.

## Span trees

### Voice

```text
job_entrypoint                          job dispatch to entrypoint exit; events job_received,
│                                       job_accepted, job_assigned, process_assigned,
│                                       job_started_on_server, entrypoint_started,
│                                       entrypoint_returned, entrypoint_cancelled
├── agent_session                       session.start() to close; events agent_state_changed,
│   │                                   user_state_changed, participant_linked, event_loop_blocked
│   ├── session_start                   start() to the first agent's on_enter
│   │   ├── room_connect, wait_for_participant, wait_for_audio_track, publish_audio_output
│   │   └── start_agent_activity > setup_toolsets, on_enter
│   ├── user_turn                       see telemetry.md, "The user turn"
│   ├── agent_turn                      one speech; one `generation` event per step, at the time
│   │   │                               the step started, with lk.generation_id and
│   │   │                               lk.parent_generation_id
│   │   ├── agent_speaking
│   │   ├── llm_node > llm_request > llm_request_run (one per attempt)
│   │   ├── tts_node > tts_request > tts_request_run (one per attempt)
│   │   ├── realtime_inference          usage and timing on the span itself
│   │   │   └── realtime_metrics        only for metrics that arrive after the inference ended
│   │   ├── function_tool
│   │   └── keyterm_detection > llm_request
│   ├── keyterm_detection > llm_request outside a reply
│   ├── pause_agent_activity, resume_agent_activity
│   ├── update_agent > drain_agent_activity > on_exit, start_agent_activity > on_enter
│   ├── amd > llm_request
│   ├── event_loop_blocked
│   ├── rpc_handler
│   └── session_close > drain_agent_activity > on_exit
└── job_shutdown > on_session_end, session_end_upload, room_disconnect, shutdown_callback
```

`room_connect` and `wait_for_participant` sit under `session_start` while the
session starts and under `job_entrypoint` when the entrypoint calls them before
`start()`.

Spans with `parent = Current` appear wherever their caller is: `llm_request`
and `tts_request` (with `llm_fallback_adapter`, `tts_fallback_adapter` and
`tts_stream_adapter` around them) for every model request in or out of a
session, `judge_evaluation` once per judge, and `rpc_call`.

Identifiers written on spans (`lk.speech_id`, `lk.generation_id`) are unique
across sessions and processes.

Every direct child of `agent_session` and `job_entrypoint` is a row in the
Cloud session view. Declaring a new span with one of them as parent adds a row.
While a session runs, its `agent_session` span is the current span of the job,
so application spans started from the entrypoint after `start()` nest under it.

The Cloud session view is keyed by room, so it shows voice sessions; text runs
reach Agent Insights and the application endpoint.

### Text

```text
a2a_request                             one A2A request; continues the caller's traceparent
└── agent_run                           one AgentRun
    ├── llm_request > llm_request_run
    └── function_tool
        └── (RemoteAgent call: the next agent's a2a_request, same trace)
```

## Consumers

| Consumer | Reads |
| --- | --- |
| Cloud session view (`livekit/web`) | span names, nesting, attributes and status; `chat_history` and `evaluations` logs; `job_entrypoint` events |
| Agent Insights (`cloud-api-server`) | span names and attributes, queried per project and agent |
| Session export | traces, metrics and logs of one room |
| Cloud ingest (`cloud-observability`) | every signal and the recording upload; redaction (next section) |
| Agent evaluations (`agent-evals.sessions`) | the upload's session summary: `agent_name`, `sdk_version`, `tags`, `duration_ms`, `user_turns`, `errors`, `close_reason`, `usage` |

How the Cloud session view renders each span, which every declaration must
keep true:

| Span | Rendered as |
| --- | --- |
| `job_entrypoint`, `agent_session` | containers, never rows; their direct children become rows; `job_entrypoint` events build the Dispatch row |
| `user_turn`, `agent_turn` | turn rows, numbered in start order; `agent_turn` counts only if a step has a direct-child `agent_speaking` |
| `user_speaking`, `agent_speaking` | speech bars, overlap and interruption markers (`lk.is_interruption` as a string) |
| `llm_node`, `llm_request`, `llm_request_run` | LLM stage; their presence marks a pipeline (not realtime) turn; TTFT and tokens |
| `tts_node`, `tts_request`, `tts_request_run` | TTS stage; TTFB, characters, audio duration; their presence marks a session as audio |
| `realtime_inference`, `realtime_metrics` | realtime TTFT and tokens |
| `function_tool` | tool stage; tool time per turn; Agent Insights tool duration |
| `eou_detection` | end-of-turn stage |
| `amd` | answering-machine rows in the transcript |
| lifecycle spans (`session_start`, `session_close`, `update_agent`, `job_shutdown`, `rpc_*`, `event_loop_blocked`) | activity rows with titles from their attributes |
| any other direct child of a container | an activity row titled by its name, colored by substring (`llm`, `tts`, `stt`, `eou`, `tool`, `speaking`) |

### Cloud ingest and redaction

`cloud-observability` receives every Cloud signal and owns redaction; the
framework sends content as recorded and keys it so the redaction finds it.

| Ingest rule | What the framework guarantees |
| --- | --- |
| Redaction runs for a PII project, or when a record carries `lk.redaction.enabled = true`, unless a record carries `lk.simulation.enabled = true` and redaction is not forced | both flags are native booleans on every span and log record of the session; a request carries one job |
| Span and log attributes are removed when a key has a `pii` dot segment or is exactly `content` or `tool_calls`, at any depth | every `Content<T>` value is under `lk.pii.*`, or is a GenAI key the Cloud destination removes itself for a redacted session |
| Span names, status messages, exception attributes and log bodies are never redacted | no content in span names or log bodies; exception text and status messages are replaced by the Cloud destination for a redacted session |
| `chat.item` log records are dropped on PII projects and rebuilt from the recording upload's `chat_history.json` after LLM-based redaction, which also mutes the redacted audio | a redacted session always uploads the `chat_history` part, even when audio is not recorded |
| The upload's first part is `header`; every part has its own `Content-Length`; part names are `chat_history`, `framework_data`, `audio`, `sqlite`; text parts are sent uncompressed | the upload is built by one declared encoder in `lk-agents-net` |
| `chat_history.json` is `{"items": [...]}` in the `ChatContext` dictionary shape; `metrics.started_speaking_at` and `stopped_speaking_at` are unix seconds on the header's `start_time` clock | the transcript encoder is golden-tested against that shape |
| `session_summary` is a JSON object of at most 16 KiB forwarded to agent evaluations (`agent-evals.sessions`); billing reads the audio's own duration | the summary is a declared record within the bound |
| Metrics sums and histograms must be delta; a record's room is read from the first top-level `room_id` or `roomID` attribute | Cloud metrics are delta; the identity stamp writes `room_id` beside the shared `lk.room_id` |
| A 401 "project data recording is disabled by owner" applies to every signal | the destination stops for that job and counts the refusal |

Compatibility rules:

- Names, types, units, parents, scopes and record shapes that a consumer reads
  change only together with the consumer: the consumer ships reading the new
  form first.
- New spans, attributes and records are additive. A new direct child of
  `agent_session` or `job_entrypoint` is a new row in the Cloud view, which the
  schema diff shows.
- Removing a legacy encoding (`assistant_turn`, the `lk.*_metrics` JSON blobs)
  happens after the consumers stop reading it.

## Inventory

Everything the framework emits. Types are OTLP value types; "JSON" is a string
holding JSON; durations are seconds. "Start" and "end" say when an attribute is
set. Keys marked *new* are additions the declarations introduce; every other key
keeps the name, type and meaning Cloud's consumers read today.

### Every span

| Key | Type | When |
| --- | --- | --- |
| `gen_ai.conversation.id` | string | start: the room session id in a job, the thread id (`AT_...`) in a text run |
| job identity (Destinations, internals) | string, bool | start, stamped by the destination |
| `error.type`, `exception.type`, `exception.message`, `exception.stacktrace`; event `exception` | string | on a recorded failure ([Errors](telemetry.md#errors)) |
| `lk.cancelled` *new* | bool | end: a `ScopedSpan` dropped without `end` |

Span kind is `INTERNAL` unless stated.

### Job

**`job_entrypoint`**: root, or continues the dispatch's `traceparent`; from the
job's receipt (backdated) to the end of shutdown.

| Key | Type | When |
| --- | --- | --- |
| `lk.job_id` | string | start |
| `lk.agent_name` | string | start, may be empty |
| `lk.pii.room_name` | string | start |
| `lk.room_id` (shared key; was `lk.room_sid`) | string | start |
| `lk.dispatch_id`, `lk.job.worker_id`, `lk.job.agent_id` | string | start |
| `lk.job.accept_latency`, `lk.job.assignment_latency`, `lk.job.launch_latency`, `lk.job.entrypoint_latency`, `lk.job.dispatch_latency` | double, s | start, each when both of its timestamps are known |

Events, each with its own timestamp and no attributes: `job_received`,
`job_accepted`, `job_assigned`, `process_assigned`, `entrypoint_started`,
`job_started_on_server`, `entrypoint_returned`, `entrypoint_cancelled`.

| Span | Parent | Keys |
| --- | --- | --- |
| `job_shutdown` | `job_entrypoint` | `lk.shutdown.reason` (string), `lk.shutdown.user_initiated` (bool) |
| `on_session_end` | `job_shutdown` | `lk.callback.name` (string); only when a callback is registered |
| `session_end_upload` | `job_shutdown` | none |
| `room_disconnect` | `job_shutdown` | none |
| `shutdown_callback` | `job_shutdown` | `lk.callback.name`; one per application callback |
| `room_connect` | `session_start`, else `job_entrypoint` | `lk.pii.room_name`, `lk.room_id`, `lk.room.auto_subscribe` (`subscribe_all`, `subscribe_none`, `audio_only`, `video_only`), `lk.room.e2ee` (bool); after connecting `lk.participant_id`, `lk.pii.participant_identity`, `lk.room.remote_participant_count` (int) |
| `wait_for_participant` | `session_start`, else `job_entrypoint` | `lk.room_io.participant_filter` (bool); when found, the participant attributes |
| `wait_for_audio_track` | `session_start`, else `agent_session` | `lk.pii.participant_identity`; `lk.first_frame_delay` (s) at the first frame; events `track_subscribed` (`lk.track_id`, `lk.track_source`), `pre_connect_audio` (`lk.pre_connect_audio.duration`), `first_frame` |
| `publish_audio_output` | `session_start` | `lk.track_id` (shared key; was `lk.track_sid`) after publishing |

The participant attributes are `lk.participant_id`, `lk.pii.participant_identity`
and `lk.participant_kind` (`PARTICIPANT_KIND_*`).

### Session

**`agent_session`**: `session.start()` to the end of close.

| Key | Type | When |
| --- | --- | --- |
| `lk.agent_label` | string | start: the first agent |
| `gen_ai.operation.name` = `invoke_workflow`, `gen_ai.workflow.name` = `agent_session` | string | start |
| `lk.pii.room_name`, `lk.job_id`, `lk.agent_name`, `lk.room_id`, `lk.dispatch_id`, `lk.job.worker_id`, `lk.job.agent_id` | string | start, in a job |
| `lk.sip.<key>` for each `sip.*` participant attribute; `lk.pii.sip.phoneNumber` | string | when a SIP participant is linked |
| `lk.blocking.count` (int), `lk.blocking.total_duration`, `lk.blocking.max_duration` (s) | | updated on each event-loop stall |

Events: `participant_linked` (participant attributes), `agent_state_changed`
and `user_state_changed` (`lk.old_state`, `lk.new_state`),
`connection_state_changed` (`lk.connection_state`), `participant_disconnected`
(participant attributes, `lk.disconnect_reason`), `event_loop_blocked`
(`lk.blocking.duration`).

| Span | Parent | Keys |
| --- | --- | --- |
| `session_start` | `agent_session` | none |
| `session_close` | `agent_session` | `lk.close_reason` (`error`, `job_shutdown`, `participant_disconnected`, `user_initiated`, `task_completed`), `lk.close.drain` (bool), `exception.type` (the close error's type) when closed by an error |
| `update_agent` | `agent_session` | `lk.previous_agent_label`, `lk.agent_label` |
| `start_agent_activity` | `session_start` or `update_agent` | `lk.agent_label`; `gen_ai.operation.name` = `create_agent`, `gen_ai.agent.name`, `gen_ai.provider.name`, `gen_ai.request.model` (the agent's LLM) |
| `setup_toolsets` | `start_agent_activity` | none |
| `on_enter` | `start_agent_activity` | `lk.agent_label` |
| `drain_agent_activity` | `update_agent` or `session_close` | `lk.agent_label` |
| `on_exit` | `drain_agent_activity` | `lk.agent_label` |
| `pause_agent_activity`, `resume_agent_activity` | `update_agent` | `lk.agent_label` |
| `event_loop_blocked` | the blocked task's span, else `agent_session` | `lk.blocking.duration`, `lk.blocking.threshold`, `lk.blocking.gc_time`, `lk.blocking.cpu_time` (s), `lk.blocking.severity` (`warning`, `error`), `lk.blocking.task`, `lk.blocking.stack`, `lk.blocking.import`, `lk.blocking.suppressed` (int); error status at `error` severity; at most 30 per minute |
| `rpc_handler` (kind `SERVER`) | `agent_session` | `rpc.method`, `lk.rpc.request_id`, `lk.rpc.caller_identity`, `lk.rpc.response_timeout` (s), `lk.rpc.handler_registered` (bool), `lk.rpc.payload_size` (int, bytes), `lk.pii.rpc.payload` (first 1024 characters); `lk.rpc.error_code` (int) on error; `lk.rpc.response_size`, `lk.pii.rpc.response` on success |
| `rpc_call` (kind `CLIENT`) | current | `rpc.method`, `lk.rpc.destination_identity`, `lk.rpc.payload_size`, `lk.pii.rpc.payload`, `lk.rpc.response_timeout`; `lk.rpc.error_code` on error; `lk.rpc.response_size`, `lk.pii.rpc.response` on success |
| `amd` | `agent_session` | at the verdict: `lk.amd.category` (string), `lk.amd.reason`, `lk.amd.speech_duration`, `lk.amd.delay` (double, s), `lk.pii.amd.transcript` |
| `keyterm_detection` | the replying `agent_turn`, else `agent_session` | `gen_ai.request.model`, `gen_ai.provider.name` at start; `lk.keyterms.count`, `lk.keyterms.added`, `lk.keyterms.removed` (int) at end |

### User turn

| Span | Keys |
| --- | --- |
| `user_turn` | start: `lk.user_turn.origin` *new*, participant attributes, `gen_ai.request.model` and `gen_ai.provider.name` (the STT); end: `lk.user_turn.outcome` *new*, `lk.pii.user_transcript`, `lk.transcript_confidence` (double), `lk.transcription_delay` and `lk.end_of_turn_delay` (s, `0` when unmeasured), `lk.provider_request_ids` (string array), `lk.stt.events` (JSON: `received_at`, `type`, `transcript_length` per event), `lk.eou.resume_count` (int, when > 0) |
| `user_speaking` | participant attributes; `lk.is_interruption` (string `"true"`/`"false"`, every verdict, `"false"` at overlap end when unset), `lk.interruption.probability` (double), `lk.interruption.total_duration`, `lk.interruption.prediction_duration`, `lk.interruption.detection_delay` (s) |
| `eou_wait` | `lk.eou.source`, `lk.eou.endpointing_delay` (s) at start and on re-arm; at end `lk.eou.outcome` (`committed`, `user_resumed`, `dropped`), `lk.eou.wait_duration` (s), `lk.eou.rearm_count`, `lk.eou.not_committed_count` (int); events `rearmed` (`lk.eou.source`), `user_resumed` |
| `eou_detection` | after the prediction: `lk.pii.chat_ctx` (JSON), `lk.eou.endpointing_delay`, `lk.eou.language`, `lk.eou.source`, `lk.eou.from_cache` (bool), `lk.eou.probability`, `lk.eou.unlikely_threshold` (when known), `lk.eou.detection_delay` (s, streaming detectors); event `superseded` (`lk.eou.outcome`) |
| `on_user_turn_completed` | `lk.agent_label`; event `stop_response` |

### Agent turn

**`agent_turn`**: one per speech, from its first generation to done.

| Key | Type | When |
| --- | --- | --- |
| `lk.speech_id` | string, unique across sessions | start |
| `gen_ai.operation.name` = `invoke_agent`, `gen_ai.agent.name` | string | start |
| `lk.generation_id`, `lk.generation_count` | string, int | each generation |
| `lk.pii.instructions`, `lk.pii.user_input` | string | pipeline reply, when present |
| participant attributes | string | pipeline and realtime replies |
| `lk.speech.queue_wait` | s | when scheduled |
| `lk.interrupted` | bool | end |
| `lk.interruption.source` (`audio_activity`, `user_turn`, `programmatic`), `lk.playout.position` (s) | | when interrupted |
| `lk.pii.response.text` | string | end |
| `lk.pii.response.function_calls` | JSON | end, realtime |
| `lk.e2e_latency` | s | end, when speech started and the user's stop is known |
| `lk.end_of_turn_delay`, `lk.transcription_delay`, `lk.on_user_turn_completed_delay` | s | end, with `lk.e2e_latency` |

Events: `generation` (`lk.generation_id`, `lk.parent_generation_id`) at each
step's start, `preemptive_generation_discarded` (`lk.speech_id`).

| Span | Parent | Keys |
| --- | --- | --- |
| `agent_speaking` | `agent_turn` (direct child) | participant attributes |
| `llm_node` | `agent_turn` | start: `lk.pii.chat_ctx` (JSON), `lk.function_tools`, `lk.provider_tools`, `lk.tool_sets` (string arrays), `lk.input.delta`, `lk.input.base_span_id`, `lk.input.dropped_from_base` with input deltas; end: `lk.pii.response.text`, `lk.pii.response.function_calls` (JSON), `lk.response.ttft` (s, measured at the node); `gen_ai.request.model`, `gen_ai.provider.name`, and `gen_ai.response.model` after a fallback; a node override that makes no `llm_request` carries the `llm_request` GenAI keys itself |
| `llm_request` (kind `CLIENT`) | current | below |
| `llm_request_run` | `llm_request` | `gen_ai.provider.name`, `gen_ai.request.model`, `lk.retry_count` (int); `lk.provider_request_ids` at the attempt's end |
| `llm_fallback_adapter` | current | `gen_ai.provider.name`, `gen_ai.request.model` (next instance), `gen_ai.request.stream`, `gen_ai.output.type`, `lk.retry_count`; on success `lk.fallback.label`, `lk.fallback.index` (int), `gen_ai.response.model`, `gen_ai.provider.name` (served); one `llm_request` child per instance tried |
| `tts_node` | `agent_turn` | `gen_ai.request.model`, `gen_ai.provider.name`; `lk.response.ttfb` (s, from the text anchor) at the first audio frame; `gen_ai.response.model` after a fallback |
| `tts_request` (kind `CLIENT`) | current | `lk.tts.streaming` (bool), `lk.tts.label`, `gen_ai.provider.name`, `gen_ai.request.model`; after success `lk.pii.input_text`; `lk.tts_metrics` (JSON: `ttfb`, `characters_count`, `audio_duration`, ...); `lk.fallback.recovery` (bool) for a recovery probe |
| `tts_request_run` | `tts_request` | `gen_ai.provider.name`, `gen_ai.request.model`, `lk.retry_count`; `lk.provider_request_ids` |
| `tts_fallback_adapter`, `tts_stream_adapter` | current | the `tts_request` start keys and `lk.retry_count`; fallback success adds `lk.fallback.label`, `lk.fallback.index`, `gen_ai.response.model` |
| `realtime_inference` (kind `CLIENT`) | `agent_turn` | below |
| `realtime_metrics` | the ended `realtime_inference` | the realtime metric keys below, when they arrive after it ended |
| `function_tool` | `agent_turn` or `agent_run` | below |

**`llm_request`**

| Key | Type | When |
| --- | --- | --- |
| `gen_ai.operation.name` = `chat`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.request.stream` (true), `gen_ai.output.type` = `text` | | start |
| `gen_ai.system_instructions`, `gen_ai.input.messages`, `gen_ai.tool.definitions` | JSON | start, with message capture |
| `lk.input.delta`, `lk.input.base_span_id`, `lk.input.dropped_from_base`, and a span link to the base | | with message capture and input deltas |
| `lk.llm_metrics` | JSON (`ttft`, `prompt_tokens`, `prompt_cached_tokens`, `completion_tokens`, `duration`, ...) | end, after a successful response |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | int | end |
| `gen_ai.usage.cache_read.input_tokens` and `gen_ai.usage.input_cached_tokens`; `gen_ai.usage.cache_write.input_tokens`; `gen_ai.usage.reasoning.output_tokens` and `gen_ai.usage.reasoning_tokens` | int | end, when > 0 |
| `gen_ai.response.id`, `gen_ai.response.model`, `gen_ai.response.finish_reasons` (`stop`, `tool_call`, `error`) | | end |
| `gen_ai.response.time_to_first_chunk` | s | end, when measured |
| `gen_ai.output.messages` | JSON | end, with message capture |
| `langfuse.observation.completion_start_time` | string, JSON-quoted ISO 8601 | end, after a first chunk |
| `lk.fallback.recovery` | bool | a fallback recovery probe |

**`realtime_inference`** and **`realtime_metrics`**

| Key | Type | When |
| --- | --- | --- |
| `gen_ai.operation.name` = `generate_content`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.request.stream`, `gen_ai.output.type` (`speech`, `text`) | | start |
| `lk.realtime_model_metrics` | JSON (`ttft` = `-1` when unmeasured, ...) | when the response's metrics arrive |
| `gen_ai.response.model`, `gen_ai.response.id` | string | same |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.{text,audio,image}.input_tokens`, `gen_ai.usage.{text,audio}.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.{text,audio,image}.cache_read.input_tokens`, and the flat forms `gen_ai.usage.input_text_tokens`, `gen_ai.usage.input_audio_tokens`, `gen_ai.usage.input_cached_tokens`, `gen_ai.usage.output_text_tokens`, `gen_ai.usage.output_audio_tokens` | int | same |
| `gen_ai.response.time_to_first_chunk` | s | same, when measured |
| `langfuse.observation.completion_start_time` | string, ISO 8601 | same, when measured |

**`function_tool`**

| Key | Type | When |
| --- | --- | --- |
| `lk.function_tool.id`, `lk.function_tool.name`, `lk.pii.function_tool.arguments` | string | start |
| `gen_ai.operation.name` = `execute_tool`, `gen_ai.tool.name`, `gen_ai.tool.type` = `function`, `gen_ai.tool.call.id`, `gen_ai.agent.name` | string | start |
| `gen_ai.tool.description`, `gen_ai.tool.call.arguments` (JSON) | | start, with message capture |
| `lk.pii.function_tool.output`, `lk.function_tool.is_error` (bool) | | end |
| `error.type` = `tool_error` | string | end, on a tool error |
| `gen_ai.tool.call.result` | JSON | end, with message capture, on success |

### Evaluation and text runs

| Span | Parent | Keys |
| --- | --- | --- |
| `judge_evaluation` | current; one per judge | `gen_ai.evaluation.name`; `gen_ai.evaluation.score.label` (the verdict); `gen_ai.evaluation.explanation` with message capture; its judge model's `llm_request` as a child |
| `a2a_request` *new* (kind `SERVER`) | the caller's `traceparent` | `gen_ai.operation.name` = `invoke_agent`, `gen_ai.agent.name`, `gen_ai.conversation.id` (thread id), `lk.a2a.task_id` |
| `agent_run` *new* | `a2a_request` or current | `gen_ai.operation.name` = `invoke_agent`, `gen_ai.agent.name`; at end `lk.run.status` (`completed`, `input_required`, `failed`) |

### Metrics

Cloud measurements carry the job identity; the `M` labels are `model_provider`
and `model_name`, the `G` labels `gen_ai.operation.name`, `gen_ai.provider.name`,
`gen_ai.request.model` and `gen_ai.response.model`.

| Instrument | Kind | Unit | Labels | Recorded |
| --- | --- | --- | --- | --- |
| `lk.agents.turn.e2e_latency` | histogram | s | the LLM's `M` | each answered turn |
| `lk.agents.turn.llm_ttft` | histogram | s | the LLM's `M` | each turn, from `llm_node` |
| `lk.agents.turn.tts_ttfb` | histogram | s | the TTS's `M` | each turn, from `tts_node` |
| `lk.agents.turn.transcription_delay`, `lk.agents.turn.end_of_turn_delay`, `lk.agents.turn.on_user_turn_completed_delay` | histogram | s | the STT's `M` | each user turn |
| `lk.agents.usage.llm_input_tokens`, `llm_input_cached_tokens`, `llm_output_tokens`, `llm_input_audio_tokens`, `llm_input_text_tokens`, `llm_output_audio_tokens`, `llm_output_text_tokens` | counter | tokens | `M` | each request's usage, when > 0 |
| `lk.agents.usage.llm_session_duration`, `tts_audio_duration`, `stt_audio_duration` | counter | s | `M` | when > 0 |
| `lk.agents.usage.tts_characters`, `interruption_num_requests` | counter | | `M` | when > 0 |
| `lk.agents.connection.acquire_time` | histogram | s | `M`, `connection_reused` (`"true"`, `"false"`) | each WebSocket connection |
| `lk.agents.event_loop.blocked_duration` | histogram | s | `severity` (`warning`, `error`) | each stall |
| `gen_ai.client.token.usage` | histogram | `{token}` | `G`, `gen_ai.token.type` (`input`, `output`) | each LLM and realtime response |
| `gen_ai.client.operation.duration` | histogram | s | `G` | each LLM and realtime response |
| `gen_ai.client.operation.time_to_first_chunk` | histogram | s | `G` | when measured |
| `gen_ai.execute_tool.duration` | histogram | s | `gen_ai.operation.name`, `gen_ai.tool.name`, `error.type` | each tool call |
| `gen_ai.invoke_agent.duration` | histogram | s | `gen_ai.operation.name`, `gen_ai.agent.name` | each agent turn |

The worker also serves Prometheus process metrics: `lk_agents_proc_initialize_duration_seconds`,
`lk_agents_active_job_count`, `lk_agents_child_process_count`, `lk_agents_worker_load`
(label `nodename`).

### Report records

| Scope | Body | Timestamp | Attributes |
| --- | --- | --- | --- |
| `chat_history` | `session report` | session start | `session.options` (the resolved options, recursively serialized; `instructions` omitted; `keyterms` as `lk.pii.keyterms`), `session.report_timestamp` (double), `session.tags` (sorted string array), `agent_name`, `sdk_version`, `usage` (the `Usage` model) |
| `chat_history` | `chat item` | the item's `created_at` | `chat.item`: one of `message`, `function_call`, `function_call_output`, `agent_handoff`, `agent_config_update` in the `livekit.agent.ChatContext.ChatItem` protobuf JSON form; severity `ERROR` for a failed tool output |
| `evaluations` | `evaluation` | report time | `evaluation`: `name`, `tag` (`lk.judge.<name>:<verdict>`), `verdict`, `reasoning`, `instructions`; severity `ERROR` for `fail` |
| `evaluations` | `tag` | the tag's time | `tag`: `name`, `metadata` |
| `evaluations` | `outcome` | report time | `outcome`: `outcome` (`success`, `fail`), `reason`; severity `ERROR` for `fail` |

### Recording upload

| Part | Content |
| --- | --- |
| `header` (`header.binpb`) | `MetricsRecordingHeader`: `room_id`, `job_id`, `start_time` (the audio recording's start), `duration` (ms), `simulated`, `redaction_enabled`, `session_summary` |
| `chat_history` (`chat_history.json`) | `{"items": [...]}`, the `ChatContext` dictionary form with timestamps; uploaded whenever the transcript is recorded, and always for a redacted session |
| `audio` (`recording.ogg`) | the session's OGG recording, when audio is recorded |

`session_summary` is a JSON object: `job_id`, `room_id`, `room`,
`audio_recording_path`, `audio_recording_started_at`, `options` (interruption,
endpointing, tool-step and recording settings), `timestamp`, `usage` (the `Usage`
model), `sdk_version`, `agent_name`, `tags`, `duration_ms`, `user_turns`,
`errors` (`llm`, `stt`, `tts`, `realtime`, `interruption` counts),
`close_reason` (`CR_*`) and, when the transcript was over its bound (2 MiB or
10,000 items), `omitted_chat_items`, the count of oldest items left out. Past
16 KiB the summary leaves out `options`, then `tags`, then `usage`.
