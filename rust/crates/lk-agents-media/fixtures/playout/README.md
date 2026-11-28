# Ordered PCM playout comparisons

Provider-independent, synchronous mono 24 kHz PCM16 playout. Input chunks have arbitrary
lengths; callers render 1–480 samples with supplied monotonic time. The library performs
no I/O, clock reads, spawning or hot-path allocation. A bounded queue rejects an entire
chunk atomically; all accepted source samples remain ordered and accounted for.

## Startup and answer lifecycle

Default settings are a 20 ms minimum reserve target, 100 ms maximum target, 80 ms cold
startup prior, 2 s capacity and 3% cumulative time-scaling budget. Targets are not hard
end-to-end latency caps: accepted speech is preserved during delivery outages.

Startup uses learned source-coverage deficits from the first 500 ms of each answer.
The first chunk's duration is not mistaken for jitter reserve. Bursts with enough extra
queued coverage can release startup early; known EOF releases a complete short reply
without waiting. Acceleration cannot consume the selected startup reserve before early
deliveries have been observed.

`begin_utterance()` clears queued PCM, old phonemes, recovery, counters, budget and clock
alignment, while retaining normalized arrival/startup distributions. Inter-answer idle
time is never a network delay sample. `reset()` also clears timing knowledge for a new
provider/connection. These are explicit cancellation boundaries; the caller owns routing
frames to the correct answer. `finish()` drains a completed answer and is idempotent.

Startup grows after bad coverage and relaxes by at most one 20 ms histogram bin per
observed answer, subject to configured bounds. A single finished chunk provides no
inter-arrival evidence and does not train the connection as good. A configurable cold
prior is honored; clean feedback cannot unexpectedly raise a smaller prior to 80 ms.

## Reserve building and recovery

Arrival estimates use actual first-sample chunk arrivals and cumulative source samples.
Splitting a 100 ms payload into imaginary 20 ms transport packets used to invent an 80 ms
arrival spread and hide a shortage of real reserve. The two-second source-time envelope
now measures delivery variation without this batching artifact. A bounded Q30 histogram
samples 500 ms peaks and selects the 95th percentile. Old observations decay.

Time scaling validates adjacent periods and the actual crossfade, instead of requiring a
whole 20 ms window of stationary speech. Genuine quiet intervals are preferred. Voiced
matches require correlation and amplitude agreement; new consonant onsets are left alone.
The onset before the splice is preserved. History-assisted expansion supports 20 ms
chunks. The timing filter compensates successful scaling and ordinary operations are at
least 50 ms apart. Severe backlog can use verified pitch multiples and a lower correlation
threshold, while still respecting the same configured cumulative rate budget.

Concealment combines multiple pitch lags with stabilized LPC residual excitation, then
fades into conservative source-trained room tone. A bounded continuation prefix aligns a
strongly matched voiced return; an energy-normalized merge restores gain after fading.
Generated audio never trains room tone. Faded recovery can rebuild half the target reserve;
EOF never waits for future source. Cancellation clears speech/DSP state immediately.

Counters expose reserve requests, budget limits, rejected splices and passive insertion,
as well as source/output accounting, queue depth, target/startup budgets and underruns.
No scenario-specific behavior exists in the library.

## Listening source and scenarios

`source.json` describes the 26.25-second GPTLive-generated passage (`gpt-live-1`, voice
`marin`), observed chunks, transcript and SHA-256. Only source edge silence was trimmed,
retaining 20 ms padding; internal pauses remain. Credentials and customer audio are absent.
Captures, WAVs, MP4s and PNGs remain outside Git. Hermetic tests synthesize PCM in memory.

The PR uses short, common wall-clock previews: 3 s for Startup/Severe, 6 s for the other
rows. Full renders retain every word and all playback-added delay. Stalls are clustered
at 2.20, 2.78, 3.36 and 4.04 s. The source is strong for 20 ms before and 50 ms after each
boundary (1000 PCM RMS threshold). Actual browser MP4 decoding is checked as well as raw
PCM; tiny raw gaps can be a poor listening demonstration after lossy encoding.

Startup delays the second 100 ms chunk to 160 ms. Severe retains the original first-answer
schedule: 0, 160, 300, 460, 700, 700, 700, 700 ms. Its 300 ms coverage deficit cannot be
eliminated within a 100 ms startup budget. A regression verifies that a learned 320 ms
startup budget prevents it when the caller explicitly allows up to 400 ms.

Other cases include clean batching, 20/30 ms lateness, repeated stalls, bounded jitter,
20 s of 1% slower delivery plus 0–30 ms jitter, twenty-second 80 ms pacing pauses, a
400 ms burst, a permanent delay step, a one-second outage, 20–200 ms chunks, 1000 ppm
clock drift and faster-than-realtime catch-up. A bounded-jitter FIFO eventually accumulates
enough reserve; recurring-pause tests explicitly retain delay rather than secretly resetting
or discarding untreated audio. These are synthetic stress schedules, not customer traces.

## Reproduce

With Cargo, FFmpeg and a C++20 compiler:

```sh
python3 rust/crates/lk-agents-media/tools/compare_playout.py \
  --input /path/to/gptlive-speech.wav \
  --webrtc /path/to/mac-arm64-release --output /tmp/playout-comparison
```

The extracted native distribution is `webrtc-a545de4` from
https://github.com/livekit/rust-sdks/releases/tag/webrtc-a545de4. The helper records source
and archive hashes, never downloads an oracle or substitutes a mock. NetEq alone needs
offline 24→48→24 kHz conversion for L16; simultaneous <=20 ms L16 packets retain each
transport chunk's actual arrival. The native algorithm remains the quality reference.

`--delay-ms`, `--startup-ms`, `--max-delay-ms` and `--stretch` control Rust budgets.
`--webrtc-max-delay-ms 0` retains NetEq's original unlimited maximum target; explicitly
select `100` for a separate target-cap comparison. Native rate control is not a Rust-style
3% cumulative budget. Report actual startup and buffered audio rather than assuming that
equal target settings imply equal achieved latency. The no-reserve diagnostic explicitly
disables time scaling on both paths.

`timeline.csv` records Rust decisions; `fifo_timeline.csv` records actual untreated gaps;
`neteq_timeline.csv` records native queue/target and lifetime operation counters. The report
uses a common pre-EOF window for comparisons, avoiding NetEq's trailing concealment after
source EOF. Added startup is measured from signal onset at 1 ms resolution. All accepted
source is checked to drain with exact source/output accounting on every Rust render.

```sh
CARGO_TARGET_DIR=/tmp/playout-target cargo run --manifest-path rust/Cargo.toml \
  -p lk-agents-media --release --example playout_turns -- \
  /tmp/playout-comparison/source.pcm /tmp/playout-comparison/turns.csv
uv run --no-project --with matplotlib python \
  rust/crates/lk-agents-media/tools/plot_catchup.py \
  --input /tmp/playout-comparison --output /tmp/playout-timing.png
```

The answer replay deliberately uses a 20 ms cold prior and a 60 ms late second chunk to
contrast retained timing with full resets. Long idle gaps and every answer's audio state
are reset independently. It is a replay using captured speech, not ten live API calls.
The one-image plot shows recorded native/Rust queues, normal/fast scaling and answer reuse.

Run media tests, purity-policy Clippy and the offline placement regressions. The allocation
regression includes startup, warm answer transitions, scaling, concealment, merge, finish
and reset. The standalone release benchmark covers normal, stalled, backlog and 1% drift
processing without hardware-specific pass thresholds.

See [NetEq coverage](neteq-coverage.md) for the operation inventory and exclusions. Histogram
arithmetic and analysis are adapted from WebRTC NetEq under `LICENSE.webrtc` and
`PATENTS.webrtc`. This remains independent floating-point Rust DSP, not a sample-exact port
or demonstrated perceptual equivalent. Provider/runtime wiring remains outside this
media-only change; broader listening and real arrival traces remain important.
