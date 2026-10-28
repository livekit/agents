# Never-block measurements

Run qualification captures from the repository root. Each workload uses a 1 ms absolute-deadline
heartbeat and the frame-derived limits in `budgets.json`. Use `--check` to fail a breached gate.
Failed workloads preserve the previous output file.

```bash
# Concurrent voice sessions with paced provider fakes
uv run --no-sync python -m scripts.bench.sessions \
  --sessions 1 8 32 --cadence-ms 10 20 --seconds 6 --output bench-clean.json

# Synchronous native wall/thread CPU and visible payload copies; run profiling separately
uv run --no-sync python -m scripts.bench.sessions \
  --profile --output bench-profiled.json

# Native stack sampling on macOS
uv run --no-sync python -m scripts.bench.sessions \
  --sessions 32 --cadence-ms 10 --sample --output bench-native.json

# Python-history conversion, native report preparation and host OTel SDK emission
uv run --no-sync python -m scripts.bench.reports \
  --sessions 1 8 32 64 --items 10000 --output bench-reports.json --check

# Real image encoding, cold decoder construction, WAV resampling and Python-thread progress
PYTHONPATH=livekit-agents-next:. uv run --no-sync python scripts/bench/media.py --output media.json
# Real native images, 32 Ogg streams and ten-second recording bursts (run from rust/)
cargo test -p lk-agents-runtime --test media_measurements --features media-images,media-ffmpeg \
  -- --ignored --nocapture

# Direct audio data-plane steps at 1/8/32 instances
cargo build --manifest-path rust/Cargo.toml -p lk-agents-bench-steps --release --locked
rust/target/release/lk-agents-bench-steps --check > bench-steps.json
```

- Voice fakes stream at declared STT/LLM/TTS/VAD timings through generated crossings. Frame dispatch,
  startup, steady-state and shutdown measurements are separate.
- Report captures replace the final network destination, retaining native policy, bounded preparation
  and SDK emission. They record at least 1,000 idle/workload heartbeats, capacity refusals, SDK stages,
  event-loop callback wall/thread CPU, and overlapping GC. `--check` requires complete reports and
  sufficient samples, with passing idle/workload scheduling and SDK/event-loop CPU gates.
- `--profile` captures synchronous Python-to-native calls and `memoryview.tobytes()` copies.
- `--sample` records inclusive native stack counts per thread.
- FakeRoom provides 20 ms media frames; `--cadence-ms 10` exercises 10 ms control service.
- `lk-agents-bench-steps` measures 10/20 ms media frames and 48-to-16 kHz resampling with wall and thread CPU clocks.

Keep captures outside the repository; compare runs on equivalent hardware and CPU quotas.
