# Never-block measurements

Run qualification captures from the repository root. Each workload uses a 1 ms absolute-deadline
heartbeat and the frame-derived limits in [the shared budgets](../../rust/bench-budgets.json). Use `--check` to fail a breached gate.
Failed workloads preserve the previous output file.

```bash
# Real image encoding, cold decoder construction, WAV resampling and Python-thread progress
PYTHONPATH=livekit-agents-next:. uv run --no-sync python scripts/bench/media.py --output media.json
# Real native images, 32 Ogg streams and ten-second recording bursts (run from rust/)
cargo test -p lk-agents-runtime --test media_measurements --features media-images,media-ffmpeg \
  -- --ignored --nocapture
```

Keep captures outside the repository; compare runs on equivalent hardware and CPU quotas.

## Rust-to-asyncio bridge

```bash
uv run --no-sync python scripts/bench/async_bridge.py --output /tmp/async-bridge.json
# Optimize dependencies as well for a full release-profile comparison:
uv run --no-sync python scripts/bench/async_bridge.py --release --output /tmp/async-bridge-release.json
```

This pairs the original generic bridge and the shared bridge in one Rust test binary,
alternating their order over 12 warmed samples. It measures ready futures, one yield,
32 yields, and creation-only Python-thread CPU. Every result is checked. By default,
the binding crate uses optimization level 3 while dependencies retain their test profile;
`--release` optimizes the whole graph. `--target-dir` selects a private build directory.
The report retains raw samples; runtime noise is not a machine-independent timing gate.

## IPC process pool

The IPC benchmark always checks correctness and exits nonzero on a violation; it needs the
current Python package and built native extension, but no credentials or external services.

```bash
# Short smoke run
PYTHONPATH=livekit-agents-next:. uv run --no-sync python -m scripts.bench.ipc \
  --rounds 2 --jobs 8 --warm 2 --concurrency 4 --output /tmp/ipc-smoke.json

# Larger process churn run
PYTHONPATH=livekit-agents-next:. uv run --no-sync python -m scripts.bench.ipc \
  --rounds 10 --jobs 64 --warm 4 --concurrency 16 --payload-bytes 65536 \
  --output /tmp/ipc-churn.json

# Also exercise the production Linux forkserver start method on a Linux host
PYTHONPATH=livekit-agents-next:. uv run --no-sync python -m scripts.bench.ipc \
  --start-method forkserver --output /tmp/ipc-forkserver.json
```

Every run uses real child interpreters and the production pool, IPC channel, supervisor and
reaping path. Only the jobs' room is simulated. Local inference model preloading is disabled
so the capture measures IPC/process overhead, not model loading or inference.

The workload consumes warmed processes, verifies their replacement, and bursts concurrent jobs
while cancelling launch callers and racing kill, close and join on the same executors. It also
checks duplicate admission, rejection during drain, and a cancelled close caller followed by
repeated close. One entrypoint deliberately stays blocked during final close, exercising the
shutdown-deadline kill path; the report separates forced kills and graceful exits. Intentional
kills can produce supervisor EOF/deadline error logs; the exit status and JSON report are the verdict.
Child-written receipts independently verify payload hashes, exactly-once job entry, process
identity and setup-before-job ordering. Lifecycle observations check terminal
ownership and completion counts; OS inspection checks that children (including zombies) are gone.
Descriptor/handle counts are checked after an untimed calibration round, with zero growth allowed
by default. The final check also rejects surviving descendants and asyncio tasks.

Correctness is verified before emergency cleanup, so fallback cleanup cannot hide a pool failure.
The JSON report contains pool startup, process creation-to-ready, warm dispatch acknowledgement,
entrypoint start, replenishment and shutdown latency, request throughput and parent event-loop lag. Performance
numbers are observations rather than machine-independent pass/fail limits. Cancellation may race
with acceptance, so a cancelled request may have started once; the report counts these cases.
Successful captures replace `--output` atomically. Failures leave its previous contents intact
and retain child receipts in the temporary directory printed with the failure.
