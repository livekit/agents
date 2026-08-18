# Floe plugin for LiveKit Agents

Route LiveKit's LLM through [Floe](https://floelabs.xyz/) so agent inference is
metered against a spend budget. Drop-in OpenAI-compatible `LLM` plus a usage
reconciler that checks LiveKit-reported token usage against Floe pricing.

Two ways to connect:

- **Keyless gateway (default)** — Floe holds the upstream provider keys and
  bills your Floe balance. You only need a Floe API key.
- **Bring your own key (BYOK)** — you supply an upstream provider key; Floe
  forwards it and meters spend against your budget.

STT/TTS are intentionally not included — Floe's voice surfaces are not yet GA.
This plugin covers the LLM only.

## Installation

```bash
pip install livekit-plugins-floe
```

## Quickstart (keyless)

Set your Floe API key:

```bash
export FLOE_API_KEY=floe_...
```

Use it like any other LiveKit LLM:

```python
from livekit.agents import AgentSession
from livekit.plugins import floe

session = AgentSession(
    llm=floe.LLM(model="openai/gpt-4o"),
    # ... stt, tts, vad
)
```

The Floe API key can be passed directly instead of via the environment:

```python
floe.LLM(model="openai/gpt-4o", api_key="floe_...")
```

## Bring your own provider key (BYOK)

Supply an upstream provider key and Floe forwards it (via the
`X-Floe-Provider-Key` header) while still metering spend against your budget.
Requests default to the metered proxy at `https://credit-api.floelabs.xyz/v1/llm`.

```bash
export FLOE_API_KEY=floe_...
export FLOE_PROVIDER_KEY=sk-...
```

```python
from livekit.plugins import floe

llm = floe.LLM(model="openai/gpt-4o")  # BYOK auto-detected from FLOE_PROVIDER_KEY
```

Or pass it explicitly:

```python
llm = floe.LLM(
    model="openai/gpt-4o",
    api_key="floe_...",
    provider_key="sk-...",
)
```

## Usage reconciliation

`FloeUsageReconciler` tracks the per-model LLM token usage LiveKit reports during
a session (via the `session_usage_updated` event) and prices each served model
against the Floe cost map. The local estimate is advisory — Floe's billed amount
is authoritative — so a divergence between the two is the thing worth watching.
It reads the model id off each usage entry, so a session that swaps or fans out
across models is priced correctly; no model has to be named up front.

```python
from livekit.agents import AgentSession
from livekit.plugins import floe

session = AgentSession(llm=floe.LLM(model="openai/gpt-4o"))

reconciler = floe.FloeUsageReconciler()
reconciler.attach(session)

# ... run the session ...

report = reconciler.summary()
print("Floe-estimated USD:", report.total_estimated_usd)
for m in report.per_model:
    print(f"  {m.provider}/{m.model}: {m.input_tokens} in + {m.output_tokens} out -> ${m.estimated_usd}")
if report.unpriced_models:
    print("unpriced (excluded from total):", report.unpriced_models)
```

## Per-turn cost receipt

For the "what did that call cost" moment, `enable_cost_receipts` logs a one-line
receipt after every Floe-routed turn — zero config:

```python
from livekit.plugins import floe

session = AgentSession(llm=floe.LLM(model="openai/gpt-4o"))
floe.enable_cost_receipts(session)
```

Each turn prints a line like:

```
floe · gpt-4o · $0.0012 est · left $99.88
```

The cost half is always shown (priced locally from the bundled cost map — free,
offline, no account). The `left $…` budget half appears when a `FLOE_API_KEY` is
set, read best-effort from hosted Floe; a failed read never breaks the session
(the cost still prints). A live-prod screenshot with a funded key is captured
separately.

## Fallback: export Floe cost over OpenTelemetry

If you'd rather ship Floe's numbers into your existing observability stack than
read them inline, the same reconciler feeds OpenTelemetry. LiveKit Agents already
emits OTel traces, so this lands Floe cost as standard OTLP metrics next to them —
`floe.cost.usd` and `floe.tokens`, tagged by agent. This is cost *observability*,
not enforcement: the budget guard stays in `floe-guard`; OTel just carries the
receipt to where ops already looks.

```python
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

from livekit.agents import AgentSession
from livekit.plugins import floe

# Point OTLP at any backend via OTEL_EXPORTER_OTLP_ENDPOINT.
reader = PeriodicExportingMetricReader(OTLPMetricExporter())
metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
_meter = metrics.get_meter("floe")
_cost = _meter.create_counter("floe.cost.usd", unit="USD")
_tokens = _meter.create_counter("floe.tokens", unit="1")


def attach_floe_otel(session: AgentSession, *, agent: str) -> floe.FloeUsageReconciler:
    reconciler = floe.FloeUsageReconciler()
    reconciler.attach(session)

    @session.on("close")
    def _drain(_ev: object) -> None:
        report = reconciler.summary()
        tokens = sum(m.input_tokens + m.output_tokens for m in report.per_model)
        _cost.add(report.total_estimated_usd, {"agent": agent})
        _tokens.add(tokens, {"agent": agent})

    return reconciler
```

## Pre-requisites

A Floe account and a Floe API key. For BYOK, also an upstream provider key.
Credentials can be passed directly or via the `FLOE_API_KEY` and
`FLOE_PROVIDER_KEY` environment variables.
