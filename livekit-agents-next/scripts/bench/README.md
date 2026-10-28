# Never-block measurements

Run qualification captures from the repository root. Each workload uses a 1 ms absolute-deadline
heartbeat and the frame-derived limits in `budgets.json`. Use `--check` to fail a breached gate.
Failed workloads preserve the previous output file.

```bash
# Concurrent voice sessions with paced provider fakes
PYTHONPATH=livekit-agents-next uv run --no-sync python livekit-agents-next/scripts/bench/sessions.py \
  --sessions 1 8 32 --cadence-ms 10 20 --seconds 6 --output bench-clean.json

# Synchronous native wall/thread CPU and visible payload copies; run profiling separately
PYTHONPATH=livekit-agents-next uv run --no-sync python livekit-agents-next/scripts/bench/sessions.py \
  --profile --output bench-profiled.json

# Native stack sampling on macOS
PYTHONPATH=livekit-agents-next uv run --no-sync python livekit-agents-next/scripts/bench/sessions.py \
  --sessions 32 --cadence-ms 10 --sample --output bench-native.json

# Python-history conversion, native report preparation and host OTel SDK emission
PYTHONPATH=livekit-agents-next uv run --no-sync python livekit-agents-next/scripts/bench/reports.py \
  --sessions 1 8 32 64 --items 10000 --output bench-reports.json --check

# Direct audio data-plane steps at 1/8/32 instances
cargo build --manifest-path livekit-agents-next/scripts/bench/native/Cargo.toml \
  --target-dir rust/target --release --offline
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
- `native/` measures 10/20 ms media frames and 48-to-16 kHz resampling with wall and thread CPU clocks.

`baselines/` contains prior captures for comparison on equivalent hardware and CPU quotas.
