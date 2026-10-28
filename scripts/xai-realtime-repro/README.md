# xAI realtime local repro harness

Local probes for three xAI realtime situations that show up next to the
id-less delete-ack hang fix. Run them from the monorepo root after
`make install`.

## Setup

```bash
make install
export XAI_API_KEY=...          # required for live probes
# optional:
# export XAI_REALTIME_MODEL=grok-voice-latest
# export XAI_RECYCLE_SECONDS=45
```

Live probes talk to `wss://api.x.ai/v1/realtime`. They do not need LiveKit
room credentials for the default path.

## Commands

Entrypoint:

```bash
./scripts/xai-realtime-repro/run.sh <unit|ref-probe|recycle|context|all>
```

### 1. Mid-session hang / delete-ack (unit) + `$ref` probe

Hermetic coverage for the id-less delete-ack fix:

```bash
./scripts/xai-realtime-repro/run.sh unit
# same as:
uv run pytest tests/test_realtime/test_xai_realtime_model.py --unit -q
```

Expect every test to pass (including
`test_delete_ack_without_id_answers_the_oldest_pending_delete`).

Live mid-session nested `$ref` tools update:

```bash
./scripts/xai-realtime-repro/run.sh ref-probe
```

Sends a tools `session.update` whose schema is named `NESTED_REF_RAW_SCHEMA`
(nested `$ref` / `$defs`), then `response.create`.

- **PASS:** tools update accepted and a response completes without a schema error
- **FAIL:** connection error, timeout, or server `error` after the tools update

### 2. Websocket recycle / long-lived session

```bash
# print construction notes only (no network)
./scripts/xai-realtime-repro/run.sh recycle --dry-log

# live demo with a short recycle window (default 45s)
./scripts/xai-realtime-repro/run.sh recycle
```

The plugin default is `max_session_duration=None` (no recycle). OpenAI
realtime defaults to 20 minutes. Callers can opt in today:

```python
from livekit.plugins.xai.realtime import RealtimeModel

RealtimeModel()  # no recycle
RealtimeModel(max_session_duration=20 * 60)  # production-like
RealtimeModel(max_session_duration=45)  # local demo
```

What to watch on a live run:

- log lines about reconnecting / reconnected
- the `session_reconnected` event (this script treats that as **PASS**)

### 3. Context loss across turns

```bash
./scripts/xai-realtime-repro/run.sh context
```

Seeds a secret code, runs a few filler turns, then asks for the code again.
Only `XAI_API_KEY` is required. If `LIVEKIT_*` is set, the script notes that
and still uses the websocket path.

- **PASS:** the recall reply contains `BLUE-ORBIT-7`
- **FAIL:** the code is missing from the recall reply, or the socket errors

The probe also prints `WARNING` lines and a final count when conversation
item events omit `item_id` / `previous_item_id` (or empty `item.id`).

## All

```bash
./scripts/xai-realtime-repro/run.sh all
```

Runs `unit`, then the three live probes. Live steps fail fast if
`XAI_API_KEY` is unset.

## Notes

- No API keys belong in this directory.
- These scripts are for local diagnosis. They are not part of the pytest gate.
