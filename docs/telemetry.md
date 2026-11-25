# Telemetry

Read consumption directly from `result.usage`, `stream.usage`, or
`session.usage`; no exporter or event subscription is required.
[Usage and metrics](usage.md) defines accounting records, scopes, and measurement
availability. This page defines what the framework exports and how an
application receives it.

The framework records traces and metrics for every voice session, text run and
job, and exports them over OTLP. LiveKit Cloud receives them for hosted agents
and agents connected to a Cloud project. An application can also send them to
its own observability backend. Process logs go to standard output.

## Sending telemetry to your backend

Point the framework at any OTLP/HTTP endpoint with the standard environment
variables:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.example.com
OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer ${OTLP_TOKEN}"
```

or in code, before the server starts:

```python
from livekit.agents import telemetry

telemetry.configure(
    endpoint="https://otlp.example.com",
    headers={"authorization": f"Bearer {os.environ['OTLP_TOKEN']}"},
    metadata={"deployment": "eu-1"},   # attributes on every span the process exports, Cloud's copy too
)
```

| `telemetry.configure(...)` | Environment | Default |
| --- | --- | --- |
| `endpoint=` | `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | none: no application destination |
| `headers=` | `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_EXPORTER_OTLP_TRACES_HEADERS`, `OTEL_EXPORTER_OTLP_METRICS_HEADERS` | none |
| none | `OTEL_EXPORTER_OTLP_COMPRESSION`, `OTEL_EXPORTER_OTLP_TRACES_COMPRESSION`, `OTEL_EXPORTER_OTLP_METRICS_COMPRESSION` (`gzip` or `none`) | `none` |
| none | `OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE` (`delta` or `lowmemory` for delta) | cumulative |
| `metadata=` | none | none |
| `allow_pii=` | `LIVEKIT_TELEMETRY_ALLOW_PII` | `False` |

A signal's own variable overrides the shared one. The destination speaks OTLP/HTTP protobuf; `OTEL_EXPORTER_OTLP_PROTOCOL` set to anything
else logs a warning and is ignored. Arguments override the environment. A later call replaces the earlier
configuration for records exported afterwards. `OTEL_SERVICE_NAME` and
`OTEL_RESOURCE_ATTRIBUTES` set the resource of what this destination receives.
The destination receives the framework's spans, the application's own spans and
the framework's metrics.

Backends that accept OTLP (Langfuse, Logfire, Datadog, Honeycomb, Grafana and
an OpenTelemetry Collector) need nothing else. The framework's spans and the
application's spans share trace ids, so the backend shows one trace per
conversation. Export never blocks or fails a conversation.

## Application spans

Use the standard OpenTelemetry API. Spans from application code and from
instrumentation libraries are exported with the framework's own, to LiveKit
Cloud and to the configured endpoint. When the server starts and no tracer
provider is set, the framework installs its own as the global provider.

```python
from opentelemetry import trace

tracer = trace.get_tracer("clinic")

@scheduler.tool
async def find_appointment(ctx: RunContext[PatientContext], last_name: str) -> str:
    """Find the patient's next appointment."""
    with tracer.start_as_current_span("clinic.lookup", attributes={"clinic.id": ctx.context.clinic_id}):
        appointment = await ctx.context.clinic.next_appointment(last_name)
    return f"{appointment.when:%A %H:%M} with {appointment.provider}"
```

`clinic.lookup` is a child of the tool call's `function_tool` span. Tools,
instruction functions, lifecycle hooks and node overrides all run inside the
framework span of their invocation, including after an error or a
cancellation.

An application that installs its own `opentelemetry-sdk` `TracerProvider`, before
starting the server, keeps it: its processors and exporters receive its spans as
before, and the framework also exports them with the session's telemetry. A
provider that is not the SDK's is left alone;
its spans are not exported by the framework.

Model requests, tool calls and agent turns follow the OpenTelemetry GenAI
semantic conventions, so GenAI-aware backends show them without extra setup.
`set_capture_content` below adds request and response messages.

## Distributed traces

A fast brain's call to its slow brain continues one trace. `RemoteAgent` sends
the current span as the W3C `traceparent` header of each A2A request, and the
endpoint runs the turn the request starts under it. A job continues the trace of
the worker that dispatched it the same way.

## Content and PII

| `livekit.agents.telemetry` | Purpose |
| --- | --- |
| `set_capture_content(enabled)` | GenAI message capture: whether model request and response messages (`gen_ai.input.messages`, `gen_ai.output.messages`) and judge explanations are recorded on spans |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` | The same setting from the environment, read at import |

GenAI message capture is on by default; set the environment variable to `false`
(or `0`, `no`, `off`) or call `set_capture_content(False)` to turn it off. A
change applies to spans recorded afterwards.

Conversation data (transcripts, responses, tool arguments and output, chat
history, evaluations) is governed per destination:

| Destination | Receives conversation data | Controlled by |
| --- | --- | --- |
| LiveKit Cloud | yes, as the session view needs it | the session's recording options; for a project with PII redaction, LiveKit Cloud removes personal data from traces and logs and redacts the transcript and recording before storing them |
| Application endpoint | only with `allow_pii=True` | `telemetry.configure(allow_pii=)` or `LIVEKIT_TELEMETRY_ALLOW_PII` |

Each destination filters its own copy; one destination's policy cannot change
another's data.

## LiveKit Cloud

A session in a job connected to LiveKit Cloud records its audio, traces,
metrics, logs and transcript by default. The session's recording options turn
parts off. At session end the framework uploads the session report; an
`rtc_session(on_session_end=)` callback receives the `JobContext` first, and
`ctx.make_session_report()` builds the same `SessionReport`.

## Product statistics

Separately from your telemetry, an agent connected to LiveKit Cloud sends LiveKit
anonymous product statistics about the framework and its plugins, so LiveKit can
see how they behave across versions and providers. They are never shown in your
dashboard, never billed, and never contain customer data.

- Where: `https://<your LiveKit Cloud host>/observability/productstats/agents/metrics/otlp/v0`,
  signed with your project's key, once a minute per process. Nothing is sent when
  `LIVEKIT_URL` is not a LiveKit Cloud URL.
- Opt out with `LIVEKIT_PRODUCT_STATS=0` or `DO_NOT_TRACK=1`, read once when the
  process starts. Opting out does not affect your observability data, and
  turning observability off does not opt out.

### What is collected

This list is complete: nothing else is sent. Every value is a closed set or a bounded
identifier, never content.

**Once per process, on every export**

| Key | Example | Used for |
| --- | --- | --- |
| `service.name` | `livekit-agents` | Tells the framework apart from other senders |
| `lk.product.agents.framework.version` | `2.0.0` | Which versions are in use |
| `lk.product.agents.runtime.name` | `python` | The runtime the agent runs on: `python`, `node` or `bun` |
| `lk.product.agents.runtime.version` | `3.12.4` | Which versions of each runtime to support |
| `lk.product.agents.os`, `lk.product.agents.arch` | `linux`, `aarch64` | Which platforms to support |
| `lk.product.agents.runner.cmd` | `start` | The command that started the process: `dev`, `console`, `simulation`, `start`, or `none` when no `lk agent` command ran |
| `lk.product.agents.runner.kind` | `production` | What the process is, derived from the command and its environment: `production`, `development`, `ci` or `unknown` |
| `lk.product.agents.storage` | `agent-db` | Where threads are stored: `agent-db`, `sqlite` or `none` |

`runner.kind`, first match: `ci` when `CI=true`; `production` on LiveKit Cloud;
`development` for `dev`, `console` and `simulation`, and for `start` on macOS or Windows;
`production` for `start` on Linux; `unknown` otherwise.

**Model usage**

Each metric carries the same labels:

| Label | Values |
| --- | --- |
| `lk.product.agents.model.kind` | `stt`, `tts`, `llm`, `realtime`, `duplex` |
| `lk.product.agents.model.plugin` | the code that runs: the plugin's id (`deepgram`, `openai`), or `custom` for any class outside a LiveKit plugin |
| `lk.product.agents.model.provider` | the vendor that is called (`deepgram`, `cerebras`); a self-hosted endpoint is `other` |
| `lk.product.agents.model.name` | the model's name as configured |

| Metric | Type | Counts |
| --- | --- | --- |
| `lk.product.agents.model.requests` | counter | Requests made to a model: one LLM completion or one TTS synthesis. Not counted for an STT, whose usage is `audio_duration` |
| `lk.product.agents.model.ttft` | histogram, seconds | Time to the first token of an LLM, realtime or duplex response |
| `lk.product.agents.model.ttfb` | histogram, seconds | Time to the first audio of a TTS request |
| `lk.product.agents.model.transcription_delay` | histogram, seconds | Time from the end of the user's speech to the STT's final transcript, once per user turn |
| `lk.product.agents.model.audio_duration` | counter, seconds | Audio sent to an STT. A streaming STT keeps one stream for a whole session, so its usage is the audio, not a count of streams |

The transcription delay starts at the end of speech, which the STT does not always decide.
So this metric alone is split by one extra label,
`lk.product.agents.model.transcription_delay.endpointing`: `vad` when the session's VAD ended
the speech, `stt` when the STT detects the end of speech itself.

- A turn whose final transcript arrived before the end of speech has no delay to measure:
  it records nothing, never a zero. So does a turn whose end of speech was not detected.
- Delays are compared only within one `endpointing` value, as medians and percentiles per
  provider and model. A `vad` delay includes the VAD's silence window; an `stt` delay
  includes the STT's own silence window.

Plugin and provider are usually the same name, and both are kept because they differ when
one plugin reaches several vendors:

| Setup | `plugin` | `provider` |
| --- | --- | --- |
| The Deepgram plugin | `deepgram` | `deepgram` |
| The OpenAI plugin pointed at Cerebras (`openai.LLM.with_cerebras()`) | `openai` | `cerebras` |
| The OpenAI plugin pointed at a self-hosted endpoint | `openai` | `other` |

The plugin says which code to maintain; the provider says which vendor people use.

A model served by LiveKit Inference records no model usage here: LiveKit already has it
from the Inference service. Only the `inference` models themselves are left out, by what
they are, never by the code they share: an OpenAI-compatible plugin is always recorded,
whichever endpoint it calls.

The plugin id is the last part of the package name, lowercase with hyphens, the same in
every language: `livekit-plugins-deepgram` in Python and `@livekit/agents-plugin-deepgram`
in Node are both `deepgram`.

Model usage is recorded by the framework around every model, never by a plugin: a plugin
written in Python is counted exactly like a built-in one, and a plugin cannot add, change
or skip a statistic. The plugin, provider and model names are the ones the model reports
about itself; a plugin that reports none appears as `unknown`.

**App shape**

One count per conversation, recorded when it ends: a call (`rtc_session`) or a text request
(`a2a_session`). Its labels are read together, so the combinations are visible.

| Metric | Counts |
| --- | --- |
| `lk.product.agents.sessions` | Conversations handled |

| Label | Values |
| --- | --- |
| `lk.product.agents.session.entrypoint` | `rtc_session`, `a2a_session` |
| `lk.product.agents.session.topology` | `single`: one agent for the whole conversation; `handoff`: one agent replaced another; `agent_as_tool`: a local agent was called as a tool; `handoff+agent_as_tool`: both |
| `lk.product.agents.session.delegation` | `none`: everything ran in this process; `remote`: a `RemoteAgent` was called |

**Voice configuration**

One count per voice session, recorded when it starts, with the configuration it starts with.

| Metric | Counts |
| --- | --- |
| `lk.product.agents.voice.sessions` | Voice sessions started |

| Label | Values |
| --- | --- |
| `lk.product.agents.voice.pipeline` | `stt-llm-tts`, `realtime`, `realtime+tts` (a realtime model with a separate TTS), `duplex` |
| `lk.product.agents.voice.turn_detection` | `vad`, `stt`, `realtime_llm`, `manual`, or `model` for a turn detection model |
| `lk.product.agents.voice.interruption` | how the session decides that the person is interrupting: `adaptive` (the interruption model classifies overlapping speech), `vad` (any detected speech interrupts), `off` (interruptions disabled) |
| `lk.product.agents.voice.false_interruption` | whether a false interruption pauses the agent's speech and resumes it: `on`, `off` |

**Feature usage**

`lk.product.agents.feature.sessions` counts, for each feature, the conversations that used
it. A conversation can use several features, or none:

- When a conversation ends, it adds 1 for each feature it used, whether it used that
  feature once or a hundred times.
- A call that used MCP and `override_options` adds 1 under `mcp` and 1 under
  `override_options`. A call that used no listed feature adds nothing.
- Dividing a feature's count by `lk.product.agents.sessions` gives the share of
  conversations that use it.

The feature is the label `lk.product.agents.feature`, one of:

| Value | The conversation |
| --- | --- |
| `mcp` | ran an agent that has an MCP toolset |
| `update_options` | called `session.update_options(...)` |
| `override_options` | entered `session.override_options(...)` |
| `realtime_model_switch` | changed its realtime model while running |
| `background_audio` | played background audio |

A feature that needs more than its name gets its own metric, counted by the same rule.
There is one so far:

| Metric | Label | Counts |
| --- | --- | --- |
| `lk.product.agents.voice_task.sessions` | `lk.product.agents.voice_task` | Conversations that ran the task, 1 for each different task |

`lk.product.agents.voice_task` is the name of a prebuilt task (`GetEmailTask`,
`GetDtmfTask`, `WarmTransfer`), or `custom` for any task an application defines. A call
that ran `GetEmailTask` twice and two of its own tasks adds 1 under `GetEmailTask` and 1
under `custom`.

**Never collected**

- Content: transcripts, prompts, replies, tool arguments or results, audio.
- Names and ids of rooms, participants, jobs, threads, agents, tools, endpoints or
  application-defined tasks; host names, user names or paths.
- Failures, retries, fallbacks, reconnects, crashes, stalls and job outcomes.

Product statistics are Telemetry Data under section 7.2 of the LiveKit Terms of
Service.

## Logs

Each process writes its logs to standard output; every record a job writes
carries its `room_id`, `job_id` and agent identity. On LiveKit Cloud hosting,
the hosted agent's log collector ingests them into the session view. The
framework does not export process logs over OTLP, so a self-hosted agent's
logs do not appear in the Cloud session view; route standard output to your own
log pipeline. The session report's records (transcript, events) reach Cloud as
OTLP log records.

## Prometheus

A worker given a `prometheus` listener serves its own state (load, running jobs)
at `/metrics` in the Prometheus text format;
see [listeners](server.md#listeners).

## Reference

The framework's span names, attributes, metrics and records are its telemetry
schema, published as an OpenTelemetry semantic-conventions registry under
`rust/otel/`. How each signal is produced and exported is described in
[telemetry internals](internals/telemetry.md).
