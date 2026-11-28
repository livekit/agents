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
anonymous product statistics about how the framework, its plugins and its features are
used, so LiveKit can decide what to build and maintain. They are never shown in your
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

Naming, the same everywhere below:

- Every key starts with `lk.product.agents.`; the tables leave that prefix out.
- A session is one call (`rtc_session`) or one text request (`a2a_session`).
- A label that belongs to one metric is named under it (`model.transcription_delay.endpointing`).
- Values LiveKit defines are lowercase with underscores (`agent_as_tool`). A plugin id keeps
  the hyphens of its package name (`livekit-inference`).

**Process**

Sent once per process, with every export.

| Key | Example | Meaning |
| --- | --- | --- |
| `service.name` | `livekit-agents` | The sender; the only key without the prefix |
| `framework.version` | `2.0.0` | The `livekit-agents` version |
| `runtime.name` | `python` | The runtime the agent runs on: `python`, `node` or `bun` |
| `runtime.version` | `3.12.4` | That runtime's version |
| `os`, `arch` | `linux`, `aarch64` | The platform |
| `runner.cmd` | `start` | The command that started the process: `dev`, `console`, `simulation`, `start`, or `none` when no `lk agent` command ran |
| `runner.kind` | `production` | What the process is: `production`, `development`, `ci` or `unknown` |
| `storage` | `agent_db` | Where threads are stored: `agent_db`, `sqlite` or `none` |

`runner.kind`, first match: `ci` when `CI=true`; `production` on LiveKit Cloud;
`development` for `dev`, `console` and `simulation`, and for `start` on macOS or Windows;
`production` for `start` on Linux; `unknown` otherwise.

**Models**

Every model metric carries the same four labels:

| Label | Values |
| --- | --- |
| `model.kind` | `stt`, `tts`, `llm`, `realtime`, `duplex` |
| `model.plugin` | the code that runs: a plugin id (`deepgram`, `openai`), `livekit-inference` for LiveKit Inference, `custom` for any class outside a LiveKit plugin |
| `model.provider` | the vendor whose model runs, as the model declares it (`deepgram`, `cerebras`) |
| `model.name` | the vendor's own id for the model (`nova-3`, `gpt-4o-mini`) |

Plugin and provider are usually the same name. Both are kept because one plugin can reach
several vendors: the plugin says which code to maintain, the provider says which vendor
people use.

| Setup | `model.plugin` | `model.provider` | `model.name` |
| --- | --- | --- | --- |
| The Deepgram plugin | `deepgram` | `deepgram` | `nova-3` |
| LiveKit Inference with `deepgram/nova-3:en` | `livekit-inference` | `deepgram` | `nova-3` |
| The OpenAI plugin pointed at Cerebras (`openai.LLM.with_cerebras()`) | `openai` | `cerebras` | `llama-3.3-70b` |

- One model has one provider and one name, however it is reached and in every language.
  Through LiveKit Inference the provider is the vendor behind it, never `livekit`, and the
  name loses its vendor prefix and its options: `deepgram/nova-3:en` is `deepgram` and
  `nova-3`. So one query compares a model across plugins and Inference.
- No table maps names, in the framework or anywhere else. The provider is the name the
  model declares for itself: a plugin names its vendor where the vendor is chosen
  (`openai.LLM.with_cerebras()` declares `cerebras`), and LiveKit Inference takes it from
  the model string. The model name is the vendor's id as the application gave it; nothing
  is renamed or grouped, so a dated snapshot (`gpt-4o-mini-2024-07-18`) stays itself.
- A model that declares no provider, like a plugin given only a base URL, is `unknown`:
  its host name is never sent.
- A plugin id is the last part of the package name, the same in every language:
  `livekit-plugins-deepgram` in Python and `@livekit/agents-plugin-deepgram` in Node are
  both `deepgram`.
- LiveKit Inference is recorded like any other model. The Inference service keeps its own
  usage records, and those, not these, are the ones billing uses. Only the `inference`
  models get the `livekit-inference` id, by what they are, never by the code they share:
  an OpenAI-compatible plugin is always recorded under its own id.
- The framework records these around every model; a plugin never does. A plugin written in
  Python is counted exactly like a built-in one, and cannot add, change or skip a
  statistic. A model that does not report its provider or name appears as `unknown`.

Usage: the unit is the metric, and the kind of model is only a label.

| Metric | Type | Counts |
| --- | --- | --- |
| `model.requests` | counter | Requests made to the model: one LLM completion, one TTS synthesis, one STT stream (one recognition for a non-streaming STT) |
| `model.input_tokens` | counter | Tokens the model consumed, as its provider reports them |
| `model.output_tokens` | counter | Tokens the model produced, as its provider reports them |
| `model.characters` | counter | Characters of text the model was given to speak |
| `model.audio_duration` | counter, seconds | Audio the model heard (an STT) or spoke (a TTS) |

A model adds to each unit it is measured in and to no other:

| Model | Adds to |
| --- | --- |
| An LLM | `requests`, `input_tokens`, `output_tokens` |
| A TTS billed by characters | `requests`, `characters`, `audio_duration` |
| A TTS billed by tokens | `requests`, `input_tokens`, `output_tokens`, `characters`, `audio_duration` |
| A streaming STT | `requests`, `audio_duration` |

A new way of measuring a model is one new unit, never a metric for one kind of model.

Latency:

| Metric | Type | Measures |
| --- | --- | --- |
| `model.ttft` | histogram, seconds | Time to the first token of an LLM, realtime or duplex response |
| `model.ttfb` | histogram, seconds | Time to the first audio of a TTS request |
| `model.transcription_delay` | histogram, seconds | Time from the end of the person's speech to the STT's final transcript, once per turn |

The transcription delay starts at the end of speech, which the STT does not always decide.
So it alone has a fifth label, `model.transcription_delay.endpointing`: `vad` when the
session's VAD ended the speech, `stt` when the STT detects the end of speech itself.

- A turn whose final transcript arrived before the end of speech has no delay to measure:
  it records nothing, never a zero. So does a turn whose end of speech was not detected.
- Delays are compared only within one `endpointing` value, as medians and percentiles per
  provider and model. A `vad` delay includes the VAD's silence window; an `stt` delay
  includes the STT's own.

**Sessions**

`sessions` counts sessions, once each when the session ends. It has one label.

| Label | Values |
| --- | --- |
| `session.entrypoint` | `rtc_session`, `a2a_session` |

**Voice sessions**

`voice.sessions` counts voice sessions, once each when the session starts, with the
configuration it starts with.

| Label | Values |
| --- | --- |
| `voice.pipeline` | `stt_llm_tts`, `realtime`, `realtime_tts` (a realtime model with a separate TTS), `duplex` |
| `voice.turn_detection` | `vad`, `stt`, `realtime_llm`, `manual`, or `model` for a turn detection model |
| `voice.interruption` | how the session decides that the person is interrupting: `adaptive` (the interruption model classifies overlapping speech), `vad` (any detected speech interrupts), `off` (interruptions disabled) |
| `voice.false_interruption` | whether a false interruption pauses the agent's speech and resumes it: `on`, `off` |

**Features**

One counter per feature. Each counter counts uses of the feature. A session that calls
`update_options` three times adds 3.

| Metric | Counts each time |
| --- | --- |
| `feature.handoff` | an agent replaced another in a voice session |
| `feature.agent_as_tool` | a local agent was called as a tool |
| `feature.remote_agent` | a `RemoteAgent` was called |
| `feature.mcp` | an MCP toolset was set up |
| `feature.update_options` | `session.update_options(...)` was called |
| `feature.override_options` | `session.override_options(...)` was entered |
| `feature.background_audio` | a background audio player was started, with or without a session |
| `feature.language_switch` | the STT reported a different language than the one before |
| `feature.model_switch` | a model was replaced with another while a session ran |

A feature that needs more than a count gets labels on its own counter. One does so far.
`feature.model_switch` says which kind of model was replaced by which, in two labels:
`feature.model_switch.from.kind` and `feature.model_switch.to.kind`, each `stt`, `tts`, `llm`,
`realtime` or `duplex`. No provider and no model name is recorded.

- A different TTS in place of the TTS is `tts -> tts`. A session that moves from a realtime
  model to a duplex one is `realtime -> duplex`.
- "From" is the model installed just before, however the switch is made (`update_options`,
  `override_options`, or an agent that brings its own model). A model installed where the
  session had none is not a switch, and neither is the same model installed again or a
  fallback adapter moving to another of its members.
- Each switch adds 1: a session that swapped its TTS three times adds 3 under `tts -> tts`.

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
