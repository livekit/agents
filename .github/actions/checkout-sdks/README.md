# SDK source checkouts

Jobs that resolve the workspace's local SDK dependencies use this action after checking out agents.
Set these repository variables to published source revisions compatible with Agents 2.0:

| Variable | Source repository | Required APIs |
| --- | --- | --- |
| `AGENTS_PYTHON_SDKS_REF` | `livekit/python-sdks` | Thread protocol messages, thread/database clients and agent token grants. |
| `AGENTS_RUST_SDKS_REF` | `livekit/rust-sdks` | Matching thread protocol, `ThreadClient`, `DatabaseClient`, and agent token grants. |

Use commit SHAs so all jobs build the same SDK sources.
The Rust revision must include its matching `livekit-protocol/protocol` submodule.
CI fails explicitly if a required variable is absent. Compatible source revisions must be published
before configuring these variables; the upstream default branches currently lack the required schema.

The action checks out SDKs inside the Actions workspace, then moves them beside agents so uv and
Cargo resolve the same relative paths used in development. Rust-only jobs set `python-sdk: 'false'`.

Wheel builds install libopus and the codec build tools on Linux and macOS. Windows builds additionally
require a static MSVC FFmpeg installation supplied through `LK_CODECS_FFMPEG_PREFIX`; provisioning
that installation on the Windows runner remains required.
