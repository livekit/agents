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
    metadata={"deployment": "eu-1"},   # attributes on every span the process exports
)
```

| `telemetry.configure(...)` | Environment | Default |
| --- | --- | --- |
| `endpoint=` | `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | none: no application destination |
| `headers=` | `OTEL_EXPORTER_OTLP_HEADERS` | none |
| `protocol=` (`"http/protobuf"`, `"http/json"`) | `OTEL_EXPORTER_OTLP_PROTOCOL` | `"http/protobuf"` |
| `metadata=` | none | none |
| `allow_pii=` | `LIVEKIT_TELEMETRY_ALLOW_PII` | `False` |

Arguments override the environment. A later call replaces the earlier
configuration for records exported afterwards. `OTEL_SERVICE_NAME` and
`OTEL_RESOURCE_ATTRIBUTES` set the resource of what this destination receives.
The destination receives the framework's spans, the application's own spans and
the framework's metrics.

Backends that accept OTLP (Langfuse, Logfire, Datadog, Honeycomb, Grafana and
an OpenTelemetry Collector) need nothing else. The framework's spans and the
application's spans share trace ids, so the backend shows one trace per
conversation.

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
provider that is not the SDK's is left alone and reported in `diagnostics()`;
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
`on_session_end` callback receives the same `SessionReport` first.

## Diagnostics

`telemetry.diagnostics()` returns the process's export counters as
`dict[str, int]`:

| Key | Counts |
| --- | --- |
| `dropped` | records dropped because a bounded export queue was full |
| `failed` | export requests a destination rejected or that failed |
| `conflicts` | configuration conflicts reported |
| `incomplete_cleanup` | job cleanups whose final flush did not finish in time |

Export never blocks or fails a conversation.

## Logs

Each process writes its logs to standard output; every record a job writes
carries its `room_id`, `job_id` and agent identity. On LiveKit Cloud hosting,
the hosted agent's log collector ingests them into the session view. The
framework does not export logs over OTLP, so a self-hosted agent's logs do not
appear in the Cloud session view; route standard output to your own log
pipeline.

## Reference

The framework's span names, attributes, metrics and records are its telemetry
schema, published as an OpenTelemetry semantic-conventions registry under
`otel/`. How each signal is produced and exported is described in
[telemetry internals](internals/telemetry.md).
