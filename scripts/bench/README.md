# Never-block measurements

Run qualification captures from the repository root. Each workload uses a 1 ms absolute-deadline
heartbeat and the frame-derived limits in [the shared budgets](../../rust/bench-budgets.json). Use `--check` to fail a breached gate.
Failed workloads preserve the previous output file.

```bash
# Python-history conversion, native report preparation and host OTel SDK emission
uv run --no-sync python -m scripts.bench.reports \
  --sessions 1 8 32 64 --items 10000 --output bench-reports.json --check

# Real image encoding, cold decoder construction, WAV resampling and Python-thread progress
PYTHONPATH=livekit-agents-next:. uv run --no-sync python scripts/bench/media.py --output media.json
# Real native images, 32 Ogg streams and ten-second recording bursts (run from rust/)
cargo test -p lk-agents-runtime --test media_measurements --features media-images,media-ffmpeg \
  -- --ignored --nocapture
```

- Report captures replace the final network destination, retaining native policy, bounded preparation
  and SDK emission. They record at least 1,000 idle/workload heartbeats, capacity refusals, SDK stages,
  event-loop callback wall/thread CPU, and overlapping GC. `--check` requires complete reports and
  sufficient samples, with passing idle/workload scheduling and SDK/event-loop CPU gates.

Keep captures outside the repository; compare runs on equivalent hardware and CPU quotas.
