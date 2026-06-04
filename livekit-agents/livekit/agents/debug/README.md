# Memory profiling for agent workers

The framework can run with `multiprocessing_context="fork"`, which is what lets
`memray run --follow-fork` capture the worker plus every job/inference child
in one shot — at memray's full native-symbol fidelity.

> **Linux only.** `fork` is unsafe on macOS (libdispatch + objc runtime
> initialization issues), which is why Python's `multiprocessing` defaults to
> `spawn` there. memray + `fork` on macOS will crash. Profile from a Linux
> machine — Docker on the host works fine.
>
> Also: `fork` is a profiling-only setting. Don't ship to production with
> this default; it's unsafe with already-initialized native/threaded libs
> (onnxruntime threadpools, CUDA, etc.).

## Install

```bash
uv pip install memray
```

## Capture

Run the agent with memray and `--follow-fork`:

```bash
AGENT_ENTRYPOINT=src/agent.py uv run --no-sync \
  memray run --follow-fork "$AGENT_ENTRYPOINT" dev --no-reload
```

memray writes one `.bin` per process into **the parent directory of the
entrypoint script** (`src/` in the example above, not `cwd`):

```
src/
├─ memray-agent.py.70321.bin             # the worker (parent)
├─ memray-agent.py.70321.bin.70324       # forked child (job/inference)
├─ memray-agent.py.70321.bin.70327       # …
├─ livekit-proc-70321-70321.json         # manifests written by the framework
├─ livekit-proc-70321-70324.json         #   one per process; pid → kind/job_id
└─ livekit-proc-70321-70327.json
```

Use `--no-reload` so the watcher process doesn't add noise.

The framework writes the `livekit-proc-*.json` manifest files next to the
captures so the viewer can label each `.bin` as worker / job (with `job_id`) /
inference. Override the manifest directory with `LK_DEBUG_DIR` if you need to.

## View

```bash
uv run --no-sync python -m livekit.agents.debug.memory list src
#   worker pid=70321
#   job pid=70324  job=AJ_abc
#   inference pid=70327

uv run --no-sync python -m livekit.agents.debug.memory report src
uv run --no-sync python -m livekit.agents.debug.memory serve src
# → serving src at http://localhost:8042/
```

Open the index in Chrome, click into any capture, switch to the **peak** view
(top of the memray UI). For a flat, non-growing process the peak is essentially
the whole resident footprint.

When the flamegraph is too noisy (lots of `<unknown>` frames from stripped C
extensions), use memray's other reporters directly:

```bash
uv run --no-sync python -m memray table src/memray-agent.py.70321.bin.70324 -o table.html
uv run --no-sync python -m memray summary src/memray-agent.py.70321.bin.70324
uv run --no-sync python -m memray tree src/memray-agent.py.70321.bin.70324
```

## Share

Captures contain function names and source paths from your code — review
before sending externally. The rendered HTML is self-contained:

```bash
tar czf flamegraphs.tar.gz src/*.flamegraph.html src/index.html
```

## Caveats

- Linux only (see top of file).
- Run long enough for a clean exit — a `SIGKILL` (container OOM-kill) may
  leave the last memray buffer unflushed.
- Capture size scales with duration × allocation rate; expect tens of MB per
  child for a normal session.
