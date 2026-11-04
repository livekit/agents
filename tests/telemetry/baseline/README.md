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

## Agents 2.0

`agent_v2.py` is the same agent on the 2.0 API (`VoiceSession`, `@agent.tool`, the handoff as
`ctx.live_session.update_agent(billing)`, `inference.TurnDetector()`). No 2.0 snapshot is kept:
capture one on demand and compare it with the 1.8.10 baseline. From the repository root, after
`make install` and `make build`:

```bash
D=$(mktemp -d) && cp tests/telemetry/baseline/{agent_v2.py,scenarios.yaml} $D
echo livekit-agents-next > $D/requirements.txt   # lets `lk` detect a Python project
.venv/bin/python -I tests/telemetry/baseline/capture_proxy.py 4329 https://<project>.livekit.cloud $D/captures &
(cd $D && PATH=$OLDPWD/.venv/bin:$PATH LIVEKIT_OBSERVABILITY_URL=http://127.0.0.1:4329 \
  lk --project <project> agent simulate audio --scenarios scenarios.yaml --yes agent_v2.py)
.venv/bin/python -I tests/telemetry/baseline/decode.py $D/captures $D/decoded
.venv/bin/python -I tests/telemetry/baseline/schema.py $D/decoded $D/schema_2_0.json
python3 -I tests/telemetry/baseline/diff.py tests/telemetry/baseline/schema_1_8_10.json \
  $D/schema_2_0.json
```

`diff.py` prints what only the old snapshot has (`-`), what only the new one has (`+`) and
what changed (`~`) for spans, logs, metrics, resources and the recording upload. A key the run
did not reach (an event-loop stall, a fallback) is absent from a snapshot without being a
regression; see `docs/internals/telemetry-reference.md` for what each span carries.
`schema_1_8_10.json` is also read by the codegen test `declarations_cover_the_1_8_10_wire`.
