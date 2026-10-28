# Never-block measurement harness

This directory owns performance **capture**, separate from correctness tests. It reuses the
`_core.FakeRoom -> VoiceSession -> generated provider` path in `scripts/smoke.py` without importing
that soon-to-be-retired custom runner. The useful detector/invalid-input regressions live in the
normal categorized pytest suite. No product code, ABI exports or generated files change. The native tool uses only dependencies
already used by the subject crates, with its own lockfile; the product workspace manifest/lock stay
untouched. Its real clocks belong outside the subject cores and their purity checks.

From the repository root, use the next package explicitly: the current environment also contains
the reference package under the same import name.

```bash
PYTHONPATH=livekit-agents-next uv run --no-sync python \
  livekit-agents-next/scripts/bench/sessions.py \
  --sessions 1 8 32 --cadence-ms 10 20 --seconds 6 --output /tmp/bench-clean.json

# Separate instrumented pass: synchronous native-call wall/thread CPU and visible tobytes copies.
PYTHONPATH=livekit-agents-next uv run --no-sync python \
  livekit-agents-next/scripts/bench/sessions.py --profile --output /tmp/bench-profiled.json

# Optional macOS native stack sampling. Needs permission to sample this process.
# Run separately: sample's attach/symbolication noticeably perturbs the process.
PYTHONPATH=livekit-agents-next uv run --no-sync python \
  livekit-agents-next/scripts/bench/sessions.py \
  --sessions 32 --cadence-ms 10 --sample --output /tmp/bench-native.json

# Warm direct audio data-plane core transitions; same shared Cargo target, no new dependency.
cargo build --manifest-path livekit-agents-next/scripts/bench/native/Cargo.toml \
  --target-dir rust/target --release --offline
rust/target/release/lk-agents-bench-steps --check > /tmp/bench-steps.json

PYTHONPATH=livekit-agents-next uv run --no-sync pytest \
  tests/test_never_block_measurements.py --unit -q
```

Put the test path **before** `--unit`: the current category option consumes an optional argument.
`--no-sync` prevents measurement from modifying the concurrently maintained workspace lock/environment.
Apply the shared build environment in `plan-align/AGENT_RULES.md` before building.

`sessions.py --check` saves a complete capture and exits 1 when its measured cadence budgets fail.
Without it, capture succeeds even when budgets fail, with `measured_budget_pass=false` in the result.
Workload/provider/cleanup/sampling errors always fail and never replace an existing JSON capture.
Raw samples get unique paths, so a failed rerun also preserves earlier raw evidence.
`lk-agents-bench-steps --check` independently enforces the direct-core CPU budget. Neither checker claims
unobserved interfaces passed. Validate nonempty JSON before copying it into a named baseline.

## Measurements

- An absolute-deadline 1 ms asyncio heartbeat records p99, p99.9, maximum and exclusive histogram
  buckets. After a stall it records the full lateness and skips missed ticks, avoiding a catch-up
  flood or an understated stall. Frame-dispatch lateness is measured separately. Startup and steady
  state are separate; shutdown is bounded and timed.
- Provider fakes stream at declared timings: 400 ms utterance, 100 ms STT finalization, 30 ms LLM
  chunks, 80 ms TTS startup, 20 ms PCM at 150 words/minute, 32 ms VAD reports. They exercise actual
  generated crossings, conversion, framing and playback; fake inference does not run a real model.
  Every session must consume STT/VAD frames, finalize recognition, synthesize and play audio.
- `--profile` uses the interpreter's profiling API, scoped/restored on the asyncio thread, to time
  synchronous `_core` calls with wall and thread CPU clocks. This is **not** an async core-step timer.
  Visible `memoryview.tobytes()` calls are counted. The independent counted STT/VAD deliveries give
  an audited binding-copy lower bound: one `PyBytes::new` per received PCM frame in
  `lk-agents-abi/src/utils/audio.rs`. In this workload, two consumers mean at least two payload
  copies per room input frame. Internal copies and TTS crossings are outside that bound.
- The standalone Rust example times the existing `AudioForwarder::on_frame` synchronous data-plane
  transition across 1/8/32 independent instances, 10/20 ms frames and 48->16 kHz native resampling.
  It installs STT/VAD/EOT sinks and performs 1,000 frames per instance. POSIX thread CPU plus wall
  clocks produce typed serde histograms, including clock overhead. Input construction and effect
  execution/destruction are outside the measured step. It reports warm core throughput separately;
  it is neither a complete decision-Driver timer nor concurrent full-session execution.
- `--sample` captures native stacks externally and emits typed thread/sample/symbol/kind records.
  Original stack files retain ancestry needed to distinguish idle parks from harmful lock waits.
  Counts are inclusive and overlap. They are **not CPU durations** and contain no consecutive-sample
  order; do not sum them or label a parked worker as a stalled worker. Exact worker heartbeat and
  async Driver CPU stay null. This is the non-invasive sampling alternative, pending runtime-owned
  observation hooks after the active runtime extraction.

`budgets.json` is a contract record (not runtime configuration). It records workload assumptions and the frame-derived limits: scheduling p99.9 <=20%
of cadence and max <50%; synchronous native CPU max <=10%; direct data-plane p99.9 CPU <=5% and
max <=10%. Thus the 10 ms case gives 2/<5 ms loop and 0.5/1 ms step budgets. These are bootstrap
engineering gates, not measured production capacity or a proof of the never-block contract.

## Baseline and scope

`baselines/` preserves the clean and profiled captures, direct-core histograms and a compact native
sampling summary from 2026-10-28. The detailed report is
`/Users/theo/livekit/plan-align/reviews/bench-baseline.md`; the raw native recording is retained there
alongside it. Record the native library hash as well as HEAD: Python uses the installed extension,
while the direct Rust example is rebuilt from the checkout. Dirty-tree hash and machine identity
are captured. Future fixes should repeat the same workload on comparable hardware/load and improve
the reported baseline; never replace an over-budget measurement with a claimed pass.

FakeRoom's public microphone has a fixed **20 ms** frame shape. Full-session `--cadence-ms 10`
means 10 ms clock/control service with 20 ms microphone/TTS frames. The Rust example supplies real
10 ms media frames. The script does not claim otherwise.

Shared-machine OS scheduling, instrumentation overhead and Python/native work all affect wall lag;
wall minus thread CPU is **not** a precise OS-descheduling attribution. These short fake-provider
runs do not qualify real VAD/EOT CPU, codecs, GIL/native waits, saturation, retained resources,
allocations, multi-process jobs, remote rendering or 30-minute stability. K-23/K-26/K-30 remain open
for those measurements and for exact per-Driver-step/per-worker observations.
