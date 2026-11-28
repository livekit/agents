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
existing test allocator.

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
upstream notices below.
Provider/runtime integration and validation against real arrival traces remain separate.

## Upstream notices

For the adapted playout components:

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
