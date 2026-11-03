# Telemetry baseline: livekit-agents 1.8.10

`schema_1_8_10.json` is the value-free shape of everything livekit-agents 1.8.10
sends to LiveKit Cloud: span names, parents, kinds, attribute keys and types,
events, log records, metric instruments, resources and the recording upload. It
was captured from three `lk agent simulate audio` sessions (run
`SR_wRcvH65q684R`, project nativesdk, 2026-11-02) through `capture_proxy.py`,
which records every observability request and forwards it to Cloud.

Re-capture:

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv "livekit-agents[silero,turn-detector]==1.8.10" opentelemetry-proto
.venv/bin/python agent.py download-files
.venv/bin/python capture_proxy.py 4319 https://<project>.livekit.cloud captures &
echo 'livekit-agents[silero,turn-detector]==1.8.10' > requirements.txt
PATH=$PWD/.venv/bin:$PATH LIVEKIT_OBSERVABILITY_URL=http://127.0.0.1:4319 \
  lk --project <project> agent simulate audio --scenarios scenarios.yaml --yes agent.py
.venv/bin/python -I decode.py captures decoded
.venv/bin/python -I schema.py decoded schema_1_8_10.json
```

The same capture against an Agents 2.0 agent produces a snapshot to diff with
this one; see `docs/internals/telemetry-reference.md`.
