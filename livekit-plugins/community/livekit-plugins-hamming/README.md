# Hamming plugin for LiveKit Agents

Post-call monitoring for LiveKit Agents with Hamming.

## Installation

Install the plugin from a checked out repository or extracted source archive:

```bash
python -m pip install ./livekit-plugins/livekit-plugins-hamming
```

## Pre-requisites

You will need Hamming credentials before configuring the plugin.

- `HAMMING_API_KEY`
- `HAMMING_EXTERNAL_AGENT_ID`

Credentials can be passed directly or through environment variables.

## Usage

```python
import os

from livekit.agents import AgentSession, JobContext
from livekit.plugins import hamming


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    hamming.configure_hamming(
        api_key=os.environ["HAMMING_API_KEY"],
        external_agent_id=os.environ["HAMMING_EXTERNAL_AGENT_ID"],
        recording={"mode": "session_audio"},
    )

    session = AgentSession()
    hamming.attach_session(session, job_ctx=ctx)
```

## Recording Modes

The plugin supports explicit recording modes through `configure_hamming(recording=...)`.
If `recording` is omitted, the plugin exports monitoring payloads only and does not start or resolve recordings.

- `none`
  - Send monitoring payload only.
- `session_audio`
  - Use LiveKit session recording and inline `recording_capture` on the final `/collect` payload.
- `participant_egress`
  - Start managed LiveKit participant egress and send dual-track recording URLs derived from deterministic output paths.
- `room_composite`
  - Start managed LiveKit room composite egress and send a single `recording_url` derived from a deterministic output path.

### Session Audio (Default Simple Path)

```python
hamming.configure_hamming(
    api_key=os.environ["HAMMING_API_KEY"],
    external_agent_id=os.environ["HAMMING_EXTERNAL_AGENT_ID"],
    recording={"mode": "session_audio"},
)
```

### Managed Participant Egress

```python
hamming.configure_hamming(
    api_key=os.environ["HAMMING_API_KEY"],
    external_agent_id=os.environ["HAMMING_EXTERNAL_AGENT_ID"],
    recording={
        "mode": "participant_egress",
        "livekit": {
            "url": os.environ["LIVEKIT_URL"],
            "api_key": os.environ["LIVEKIT_API_KEY"],
            "api_secret": os.environ["LIVEKIT_API_SECRET"],
        },
        "s3": {
            # Required for deterministic artifact URL resolution.
            "public_url_base": os.environ.get("LIVEKIT_RECORDING_PUBLIC_URL_BASE", ""),
            # Optional. Include upload credentials only if the plugin should write
            # directly to S3 instead of relying on LiveKit project defaults.
            "access_key": os.environ.get("LIVEKIT_RECORDING_S3_ACCESS_KEY", ""),
            "secret": os.environ.get("LIVEKIT_RECORDING_S3_SECRET", ""),
            "region": os.environ.get("LIVEKIT_RECORDING_S3_REGION", ""),
            "bucket": os.environ.get("LIVEKIT_RECORDING_S3_BUCKET", ""),
        },
    },
)
```

### Managed Room Composite

```python
hamming.configure_hamming(
    api_key=os.environ["HAMMING_API_KEY"],
    external_agent_id=os.environ["HAMMING_EXTERNAL_AGENT_ID"],
    recording={
        "mode": "room_composite",
        "livekit": {
            "url": os.environ["LIVEKIT_URL"],
            "api_key": os.environ["LIVEKIT_API_KEY"],
            "api_secret": os.environ["LIVEKIT_API_SECRET"],
        },
        "audio_only": True,
        "file_type": "ogg",
    },
)
```

Notes:

- Managed remote recording modes require LiveKit server API credentials.
- Managed remote recording modes also require `job_ctx` when calling `hamming.attach_session(...)`.
- Managed remote recording modes require deterministic artifact URL resolution via `recording.s3.public_url_base` or `recording.s3.bucket` + `recording.s3.region`.
- If you provide full S3 credentials, the plugin uses those upload settings for egress output.
- If you omit S3 upload credentials, your LiveKit project must already have default file output storage configured.
- The plugin does not poll LiveKit for completed artifact locations; it derives the final public URLs from the configured output path.

## Verify configuration

```python
import os

from livekit.plugins import hamming

report = hamming.doctor(api_key=os.environ["HAMMING_API_KEY"])
print(report.to_dict())
```

## Notes

- Sessions are exported when attached through `hamming.attach_session(...)`.
- Recording is opt-in. Omitting `recording=...` sends monitoring payloads without recording artifacts.
- `auto_record_audio=True` is still supported as a backward-compatible alias for `recording={"mode": "session_audio"}`.
- Managed remote recording modes stop egress on close and send deterministic artifact URLs through the same `/api/rest/v2/collect` ingestion path.

## Troubleshooting

- `RuntimeError: hamming is not configured`
  - Call `hamming.configure_hamming(...)` before `hamming.attach_session(...)`.
- `ValueError: Hamming API key required`
  - Set `HAMMING_API_KEY` or pass `api_key=...`.
- `external_agent_id is required`
  - Set `HAMMING_EXTERNAL_AGENT_ID` or pass `external_agent_id=...`.
- `Unsupported recording mode`
  - Use `none`, `session_audio`, `participant_egress`, or `room_composite`.
- `recording mode 'participant_egress' requires LiveKit server credentials`
  - Provide `recording.livekit` or set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`.
- `recording mode 'participant_egress' requires deterministic artifact URL resolution`
  - Provide `recording.s3.public_url_base` or `recording.s3.bucket` + `recording.s3.region`.
- `recording mode 'participant_egress' requires JobContext`
  - Pass `job_ctx=ctx` to `hamming.attach_session(...)` or call it from inside the active LiveKit job.
