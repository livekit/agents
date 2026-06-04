# Memory profiling for agent workers

The framework can run with `multiprocessing_context="fork"`, which is what lets
`memray run --follow-fork` capture the worker plus every job/inference child
in one shot — at memray's full native-symbol fidelity.

> `fork` is a profiling-only setting. It is unsafe once native/threaded libs
> are initialized in the parent (onnxruntime threadpools, CUDA, etc.). Don't
> ship to production with this; use it for a local profiling session.

## Install

```bash
uv pip install memray
```

(Linux + macOS; memray does not support Windows.)

## Capture

Run the agent with memray, telling it to follow forks:

```bash
AGENT_ENTRYPOINT=src/agent.py uv run --no-sync memray run --follow-fork "$AGENT_ENTRYPOINT" dev --no-reload
```

memray writes one `.bin` per process into the cwd:

```
src/
├─ memray-agent.py.70321.bin             # the worker (parent)
├─ memray-agent.py.70321.bin.70324       # one per forked child (jobs, inference)
└─ memray-agent.py.70321.bin.70327
```

`--no-reload` is recommended so the reloader process doesn't add noise.

## View

Render flamegraphs and an index that groups children under their parent
worker:

```bash
uv run --no-sync python -m livekit.agents.debug.memory report src
```

Browse them locally:

```bash
uv run --no-sync python -m livekit.agents.debug.memory serve src
# → serving src at http://localhost:8042/
```

Open the index in Chrome. Click into any capture, then switch to the **peak**
view (top of the memray UI) — for a flat, non-growing process the peak is
essentially the whole resident footprint.

You can also use memray's other reporters directly when the flamegraph is too
noisy (e.g. lots of `<unknown>` frames from stripped C extensions):

```bash
uv run --no-sync python -m memray table src/memray-agent.py.70321.bin.70324 -o table.html
uv run --no-sync python -m memray summary src/memray-agent.py.70321.bin.70324    # terminal
uv run --no-sync python -m memray tree src/memray-agent.py.70321.bin.70324       # interactive TUI
```

## Share

Captures contain function names and source paths from your code — review
before sending externally. The rendered HTML is self-contained and opens in
any browser without memray installed:

```bash
tar czf flamegraphs.tar.gz src/*.flamegraph.html src/index.html
```

## Caveats

- `fork` is unsafe in production. Pass `multiprocessing_context="fork"` only
  for a profiling session.
- Run long enough for a clean exit — a `SIGKILL` (container OOM-kill) may
  leave the last memray buffer unflushed.
- Capture size scales with duration × allocation rate; expect tens of MB per
  child for a normal session.
