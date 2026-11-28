# Ordered PCM playout

Synchronous, bounded mono 24 kHz PCM16 processing. Callers supply monotonic arrival/render
clock values, push complete chunks, and render 1–480 samples per call. Chunk lengths may
change. Queue pressure rejects a whole chunk atomically; accepted source stays ordered.
The caller owns downstream pacing and format conversion.

Defaults: 80 ms startup prior, 20–100 ms adaptive target, 2 s capacity, 200 ms maximum
speech concealment and a 3% cumulative insertion/removal budget. Targets are not hard
latency caps: accepted speech is preserved. Startup can release early for bursts or EOF.
`begin_utterance()` clears PCM/DSP and retains normalized arrival history; `reset()` also
clears learned timing. Clean answers gradually reduce startup; worse observed deficits
increase it within the configured bounds. Recent bad history takes time to decay.

Reserve insertion and acceleration prefer low-energy splices or matching pitch periods.
Concealment uses mixed voiced/unvoiced continuation and fades into source-trained room
tone. Recovery aligns a strong voiced match with a bounded generated prefix and blends
back to source. Generated sound never trains the background model. These algorithms
cannot reconstruct missing words or guarantee transparent sound during a long outage.

In-memory regressions cover sample preservation/accounting, changing chunks, atomic
errors, bounded startup/learning, cancellation/EOF, pitch/energy, transient rejection,
recovery edges, drift, backlog and rate limits. Allocation recording shares the crate's
existing test allocator. The CPU benchmark remains `cargo run --release --example
playout_bench -p lk-agents-media` from the Rust workspace.

## Optional native comparison

One standalone script generates its tiny native/Rust bridges in a temporary directory.
No source audio, generated fixtures, plot tools or native implementation are vendored.
Requires Python 3.11+, Cargo, FFmpeg, a C++20 compiler and an extracted distribution from
[webrtc-a545de4](https://github.com/livekit/rust-sdks/releases/tag/webrtc-a545de4):

```sh
python3 rust/crates/lk-agents-media/tools/neteq_compare.py \
  --input /path/to/mono-24k-pcm16.wav \
  --webrtc /path/to/mac-arm64-release --output /tmp/playout-comparison
```

A copied script accepts `--crate /path/to/lk-agents-media`. It renders clean, late startup,
20 s of 1%-slow jitter, 400 ms stall, 1 s outage and faster-than-realtime catch-up schedules.
It rejects silent fault onsets. WAVs, arrival/per-frame CSVs and a settings/hash/onset report
stay in the chosen output directory. All source accounting and final Rust drain are checked.

Both paths receive identical chunk arrivals. Only NetEq resamples 24→48→24 kHz for L16;
its <=20 ms encoded packets keep the enclosing chunk's arrival time. Native rate control
and default maximum differ from Rust; use `--native-maximum-ms` for an explicit alternative.
Counters are sampled before EOF because native NetEq lacks this PCM finish signal.
Onset delay excludes model, live network and device latency; no automatic quality score
or claim of perceptual parity is made. PR listening attachments remain review evidence.

RTP ordering, retransmission, encoded FEC/PLC, codec switching, DTMF and SID comfort-noise
packets belong outside this ordered PCM layer. Relevant NetEq processing informed this
independent Rust implementation; it is not a sample-exact port. Adapted components retain
[BSD attribution](../../LICENSE.webrtc) and the [patent grant](../../PATENTS.webrtc).
Provider/runtime integration and validation against real arrival traces remain separate.
