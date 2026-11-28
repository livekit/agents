# Ordered PCM playout

Bounded mono 24 kHz PCM16 playout. Push complete chunks with monotonic arrival times and
render 1–480 samples on the consumer's clock. Chunk sizes may change. The caller owns
pacing and format conversion; capacity errors reject the entire chunk atomically.

## Choose the source lifecycle

| Source | Constructor | Lifecycle |
| --- | --- | --- |
| Continuous audio | `Playout::continuous(config)` | `push` / `render`; `finish` only at actual EOF |
| Explicit segments | `Playout::segmented(config)` | First segment is ready; `end_segment`, drain, then `begin_segment` |

The continuous type has no segment methods. It starts once and keeps learning delivery
timing without boundary events. PCM silence remains part of the stream; missing audio is
an underrun, never an inferred segment boundary. Keep rendering through delivery gaps.

The segmented type exposes symmetric `begin_segment()` / `end_segment()` methods instead
of `finish()`. Use it when a provider identifies answers or other source boundaries.
`end_segment()` rejects new input while accepted audio drains. Wait for `Render::drained`
before `begin_segment()` unless deliberately cancelling queued speech. A new segment
clears audio/DSP and resets its clock, counters and scaling budget while retaining learned
delivery timing. In either mode, `reset()` discards audio **and** learned timing.

Both types use the same DSP and buffers. Mode selection has no runtime or allocation cost.

## Buffering

Defaults: 80 ms startup, 20–100 ms adaptive target, 2 s capacity, up to 200 ms extrapolated
speech and a 3% cumulative insertion/removal budget. Targets are not hard latency caps.
Queued bursts and EOF may release startup early. Across explicit segments, clean startup
observations gradually reduce the initial wait; worse history increases it within bounds.

Time scaling prefers quiet splices or matching pitch periods. Concealment blends voiced
and unvoiced continuation into source-trained room tone, then recovery merges returning
PCM. Generated sound never trains the noise model. Long outages still lose intelligibility.
Tests cover accounting, allocation, lifecycle, silence/resumption, jitter/drift, scaling,
transients, recovery, capacity errors and EOF. Provider/runtime wiring remains separate.

## Optional NetEq comparison

Requires Python 3.11+, Cargo, clang++, FFmpeg and the extracted platform archive from
[livekit/rust-sdks: webrtc-a545de4](https://github.com/livekit/rust-sdks/releases/tag/webrtc-a545de4).

```sh
python3 rust/crates/lk-agents-media/tools/neteq_compare.py \
  --input /path/to/mono-24k-pcm16.wav \
  --webrtc /path/to/mac-arm64-release --output /tmp/playout-comparison
```

The single script has typed, described presets for steady delivery, delayed startup,
bounded jitter, slow jitter, positive/negative clock drift, short/long/repeated stalls,
catch-up and variable chunks. `--help` lists exact timings. All preserve ordered source;
RTP loss, reordering and encoded FEC/PLC are outside this reliable PCM layer.

A fixed seed and source-time clock make retries reproducible with the same inputs and
toolchain. Outputs include aligned WAVs, arrival/operation CSVs and a report containing
settings, preset descriptions, seed, hashes, startup delay and pre-EOF counters. The script
checks source accounting/drain and rejects silent fault onsets. A copied script accepts
`--crate /path/to/lk-agents-media`; embedded bridges build in a temporary directory.

Only NetEq resamples 24→48→24 kHz for L16. It retains each enclosing chunk's arrival time
and its own native rate control; `--native-maximum-ms` sets an optional target cap. Onset
latency excludes model, live network and device delay. These are diagnostic comparisons,
not perceptual scores or a claim of NetEq parity. Existing PR recordings remain attached.

<details>
<summary>Upstream notices for adapted components</summary>

```text
Copyright (c) 2011, The WebRTC project authors. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

  * Neither the name of Google nor the names of its contributors may
    be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Additional IP Rights Grant (Patents)

"This implementation" means the copyrightable works distributed by
Google as part of the WebRTC code package.

Google hereby grants to you a perpetual, worldwide, non-exclusive,
no-charge, irrevocable (except as stated in this section) patent
license to make, have made, use, offer to sell, sell, import,
transfer, and otherwise run, modify and propagate the contents of this
implementation of the WebRTC code package, where such license applies
only to those patent claims, both currently owned by Google and
acquired in the future, licensable by Google that are necessarily
infringed by this implementation of the WebRTC code package. This
grant does not include claims that would be infringed only as a
consequence of further modification of this implementation. If you or
your agent or exclusive licensee institute or order or agree to the
institution of patent litigation against any entity (including a
cross-claim or counterclaim in a lawsuit) alleging that this
implementation of the WebRTC code package or any code incorporated
within this implementation of the WebRTC code package constitutes
direct or contributory patent infringement, or inducement of patent
infringement, then any patent rights granted to you under this License
for this implementation of the WebRTC code package shall terminate as
of the date such litigation is filed.
```

</details>
