# Never-block measurements

Performance capture for the never-block contract, separate from correctness tests. Sessions run
on `_core.FakeRoom -> VoiceSession` with paced provider fakes; no product hooks are added.

From the repository root (`PYTHONPATH` selects the next package):

```bash
# Loop lag across concurrent sessions and cadences
PYTHONPATH=livekit-agents-next uv run --no-sync python livekit-agents-next/scripts/bench/sessions.py \
  --sessions 1 8 32 --cadence-ms 10 20 --seconds 6 --output /tmp/bench-clean.json

# Synchronous native-call wall/thread CPU and visible payload copies (perturbs timing; run apart)
PYTHONPATH=livekit-agents-next uv run --no-sync python livekit-agents-next/scripts/bench/sessions.py \
  --profile --output /tmp/bench-profiled.json

# macOS native stack sampling (needs permission to sample the process; run apart)
PYTHONPATH=livekit-agents-next uv run --no-sync python livekit-agents-next/scripts/bench/sessions.py \
  --sessions 32 --cadence-ms 10 --sample --output /tmp/bench-native.json

# Core step CPU of the audio data plane (AudioForwarder::on_frame), 1/8/32 instances
cargo build --manifest-path livekit-agents-next/scripts/bench/native/Cargo.toml \
  --target-dir rust/target --release --offline
rust/target/release/lk-agents-bench-steps --check > /tmp/bench-steps.json
```

`--check` exits 1 when a budget fails; without it the capture records `measured_budget_pass`.
A failed workload never overwrites an existing output file.

## What is measured

- Loop lag: a 1 ms absolute-deadline asyncio heartbeat (p99, p99.9, max, histogram); a stall is
  recorded at its full lateness. Frame-dispatch lateness, startup and shutdown are separate.
- Provider fakes stream at declared timings (400 ms utterance, 100 ms STT finalization, 30 ms LLM
  chunks, 80 ms TTS startup, 20 ms PCM, 32 ms VAD reports) through the real generated crossings.
- `--profile` times synchronous `_core` calls with wall and thread CPU clocks (not async core
  steps) and counts `memoryview.tobytes()` copies.
- `--sample` reports inclusive native stack sample counts per thread; they are not durations.
- `native/` times one data-plane step with thread CPU and wall clocks across 10/20 ms frames and
  48->16 kHz resampling. It is a separate Cargo workspace so its real clocks stay outside the
  subject crates.

FakeRoom's microphone has a fixed 20 ms frame, so `--cadence-ms 10` means 10 ms control service
with 20 ms media frames; only `native/` exercises 10 ms media frames.

`budgets.json` records the frame-derived limits: loop p99.9 <= 20% of cadence and max < 50%;
synchronous native CPU max <= 10%; data-plane step p99.9 <= 5% and max <= 10%.

`baselines/` holds the 2026-10-28 captures. Later fixes rerun the same workload on comparable
hardware and must improve on them.
