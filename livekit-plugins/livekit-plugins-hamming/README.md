# livekit-plugins-hamming

Send OpenTelemetry traces, logs, and metrics from your [LiveKit](https://livekit.io/) voice agents to [Hamming](https://hamming.ai) for observability and evaluation.

## Installation

```bash
pip install livekit-plugins-hamming
```

## Quick Start

```python
from livekit.plugins import hamming

# In your entrypoint, before AgentSession.start():
telemetry = hamming.setup_hamming(
    # Uses HAMMING_API_KEY env var by default
    metadata={"livekit.room_name": ctx.room.name},
)

# Flush on shutdown
async def flush():
    telemetry.force_flush()
ctx.add_shutdown_callback(flush)
```

## Configuration

| Parameter | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| `api_key` | `HAMMING_API_KEY` | (required) | Your workspace API key |
| `base_url` | `HAMMING_BASE_URL` | `https://app.hamming.ai` | Hamming API URL |
| `metadata` | — | `None` | Attributes on all spans |
| `service_name` | — | `"livekit-voice-agent"` | OTel service name |
| `enable_traces` | — | `True` | Export traces |
| `enable_logs` | — | `True` | Export logs |
| `enable_metrics` | — | `True` | Export metrics |
| `metrics_export_interval_ms` | — | `5000` | Metrics export interval |
| `log_level` | — | `logging.INFO` | Min log level for export |

## Getting Your API Key

1. Go to [Settings > API Keys](https://app.hamming.ai/settings) in your Hamming dashboard
2. Create a new API key or use an existing one
3. Set it as `HAMMING_API_KEY` env var or pass it directly

## Correlation Attributes

Pass these in `metadata` to correlate telemetry with Hamming entities:

| Attribute | Purpose |
|-----------|--------|
| `hamming.test_case_run_id` | Link to a specific test run |
| `hamming.monitoring_trace_id` | Link to production monitoring |
| `livekit.room_name` | Match by LiveKit room |
| `livekit.room_sid` | Match by room SID |

## Example

See [examples/voice_agents/hamming_trace.py](../../examples/voice_agents/hamming_trace.py) for a complete working example.

## License

Apache 2.0
