# Staging: why publisher RTT savings did not become the same audio saving

**The JOIN optimization is active and does shorten publisher signaling. The earlier aggregate startup table did not expose that saving.** These additional native and browser traces localize the remaining delay to transport/track availability and receiver subscription, while the prior actual-before/after run separately measured approximately 100 ms of synchronous preparation overhead.

All runs used **wss://xianstaging-hixkk74p.staging.livekit.cloud**, including the [original 72-run comparison](../before-after-reports/manifest.json). No tokens, API secrets, SDP bodies or ICE credentials are recorded here.

## What is now directly verified

Eight measured publisher diagnostic trials (four ordinary, four JOIN; two excluded warmups) used the same SDK and instrumented native build, with the same track-preparation work in both arms:

- Ordinary: initial JOIN has a publisher offer and zero tracks. Native ordinary publication explicitly selects parallel mode, then writes one AddTrack and one additional offer.
- Prepared: the actual serialized JOIN contains one audio AddTrack request and its publisher offer. The native session confirms single-PC mode, receives the matching track acknowledgement, commits the JOIN publication, and writes **no post-JOIN AddTrack or offer**. No fallback occurs.
- Native timestamps are recorded at the serialization, socket write/read, SDP-application and PeerConnection-state boundaries. The patch emits only names/counts/timestamps, not raw signaling contents.

Median milliseconds **after receiving JoinResponse**, across four trials per path:

| Native milestone | Ordinary parallel publication | Track bundled in JOIN |
|---|---:|---:|
| Track registration acknowledgement | 425 ms | ~0 ms |
| Final publisher SDP answer applied | 464 ms | 127 ms |
| Publisher PeerConnection connected | 273 ms | 282 ms |
| TrackSubscribed received / output allowed to accept PCM | 646 ms | 606 ms |

The publisher's final SDP milestone advanced by approximately **337 ms**, while subscription readiness advanced only about **40 ms** in these independent medians. This small diagnostic sample is not evidence of a stable 40 ms UX gain. The actual before/after benchmark's fast case remained effectively unchanged.

These are a **new diagnostic run**, not the 1,338 → 1,336 ms startup measurement. Network conditions changed: browser ICE RTT medians were approximately 144–148 ms, with a retained 860 ms outlier, versus 61 ms in the earlier comparison. Both paths in the diagnostic use the same new native binary; this isolates signaling rather than re-estimating the SDK-before/after effect. Independent medians do not form one additive timeline.

A representative matched pair, relative to native connect entry, illustrates the same distinction:

| Milestone | Ordinary | JOIN |
|---|---:|---:|
| JoinResponse received | 629 ms | 628 ms |
| Track registration acknowledgement | 924 ms | 628 ms |
| Final publisher SDP applied | 1,067 ms | 757 ms |
| Subscription-ready | 1,241 ms | 1,234 ms |

That pair saves about 296 ms of registration time and 310 ms to final SDP, but only 7 ms to subscription readiness. All trials, including less favorable JOIN trials, remain in the raw data.

## The receiver-side work is still present

A second diagnostic adds browser signal timestamps and observes outbound packet counters: four measured trials (two per path, plus two excluded warmups).

In **every** measured trial, after the agent track is announced the browser still receives `MediaSectionsRequirement`, sends an offer, and receives an answer mapping the remote track. This subscription offer/answer exchange occurs on both publication paths. The agent's output gate then waits for the server's `TrackSubscribed` event before accepting greeting PCM.

For example, in an ordinary trial the browser saw the track at native-relative 941 ms, sent its subscription offer at 976 ms, and received the answer at 1,124 ms; the agent received `TrackSubscribed` at essentially that last timestamp. In a prepared trial, the track was announced at 1,041 ms, the browser offered at 1,074 ms, and the answer/subscription acknowledgement arrived at approximately 1,227–1,234 ms. Thus earlier publisher registration is not the same event as earlier subscriber readiness.

Native outbound RTP was observed before the application supplied its first greeting PCM in all four trials. Preparation/publication therefore does not depend on an LLM or microphone producing audio first. Packet sampling only establishes that RTP was already flowing; it is not a precise timestamp for the first packet or the SFU's internal track-binding event.

The exact SFU-side reason that earlier publisher signaling does not yield proportionally earlier track announcement is **not established** by these client-side traces. Server-side transport/track-binding traces would be needed to distinguish its processing, packet arrival and scheduling. It would be incorrect to attribute all the remaining delay to ICE, DTLS or a specific server timer without that evidence.

## How to interpret “saves two RTTs”

That estimate describes removed publisher signaling dependencies relative to an older sequential publication flow. It is not a promise to subtract two RTTs from the time until a remote listener is subscribed and hearing audio. Our original baseline already includes Rust #1704's parallel AddTrack/SDP publication and early single-PC offer.

The observed critical path still includes publisher transport establishment, remote-track availability, browser subscription negotiation, the server subscription acknowledgement, and audio delivery. The proposed Agents patch also moved roughly 100 ms of synchronous source/track preparation ahead of activity startup. The signaling gain is real, but the total UX gain cannot be obtained by subtracting signaling RTTs from the aggregate startup number.

Next work should remove the measured blocking preparation regression and trace/optimize the receiver/track-availability path. These diagnostics do not justify claiming that the publisher optimization itself is ineffective, nor do they demonstrate an end-to-end two-RTT audio saving.

## Evidence and reproduction

- [Summary and all per-run phase measurements](summary.json).
- [Publisher client results](publisher/clients.json), [publisher native traces](publisher/traces/), [publisher manifest](publisher/manifest.json).
- [Receiver signal results](subscriber/clients.json), [receiver/native traces](subscriber/traces/), [receiver manifest](subscriber/manifest.json).
- [Native instrumentation patch](native-instrumentation.patch), [build/hash metadata](native-build.json). The instrumentation was applied in an isolated detached worktree based on Rust `34d114b0`; production SDK implementation in the PRs was not changed.

Build the native patch using the existing dev configuration. Set the staging credentials externally, point `LIVEKIT_LIB_PATH` at that diagnostic library, and set `LIVEKIT_PUBLICATION_TRACE=1` plus `STARTUP_NATIVE_TRACE=1`. Use the linked new Agents/RTC source trees, then run:

```sh
python examples/startup_tracing/run.py --profile native-pair --case controlled-fast \
  --repetitions 4 --output /tmp/publication-native-trace
```

The diagnostic handler accepts only the sanitized native trace target and drops other native log contents. The browser records signal types and counts, not message bodies. As in the earlier benchmark, the UI is preloaded, single-PC is requested, workers are warm, synthetic tone PCM is used, and no real LLM/TTS or capture-device startup is measured.
