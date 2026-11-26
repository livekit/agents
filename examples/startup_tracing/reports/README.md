# Agent startup bottleneck experiment

**These are controlled staging measurements, not production model latencies.** A real browser button joined a room, the token dispatched a real named worker, and a real Python `AgentSession` sent audio through the remote SFU to the browser. Controlled audio is a tone; the activity/audio delays below were deliberately injected. No SDK implementation was changed. Existing `publish_track()` behavior was used; no initial publication was supplied to connection.

The real OpenAI Realtime attempt failed with `invalid_api_key` using the locally configured key. It produced no successful latency sample and is excluded. A valid provider credential is still needed to determine which path dominates that actual agent.

## Measured user impact

45 measured starts: 3 cases × 3 existing API usage patterns × 5 repetitions. Nine warmups are excluded. All 45 process assignments were warm. Median browser ICE RTT was 62 ms. Single-PC was requested on both sides; all 45 agent diagnostics contained publisher stats and no separate subscriber-PC stats. Browser errors: 0. All 45 received audio files were independently decoded and contained non-silent audio.

Median click → first non-silent rendered audio, seconds:

| Controlled case | Connect first, start custom output afterward | Connect first, start custom output concurrently | Automatic connect and default RoomIO |
|---|---:|---:|---:|
| Fast activity / immediate PCM | 1.461 | 1.435 | 1.461 |
| Activity initialization includes 1.2 s wait | 2.635 | 2.388 | 1.831 |
| First PCM generation includes 1.2 s wait | 2.424 | 2.393 | 1.847 |

The second column follows the local example's connection/output ordering, with video, VAD, and tools omitted so ordering is isolated. The third overlaps `custom_output.start()` with `session.start()` after connecting. The fourth uses the framework's automatic connection and output startup. They are application configuration variants; the SDK code is identical.

In the slow-activity case, starting custom output concurrently improved the median by about **246 ms**. Automatic connection/output setup improved it by about **804 ms** relative to the example ordering. These effects demonstrate the cost of serialization under the controlled delay; they do not predict the magnitude for a real provider.

The fast-case differences are small compared with variation, and the paired intervals include zero. With five pairs, p95 is simply the maximum observed sample. [Median/p95 and descriptive paired bootstrap intervals](results.json) are included alongside [raw rows](samples.csv).

## Does faster Rust publication help?

The automatic profile exposes the distinction:

| Controlled case | First PCM at output | Output subscription ready | PCM wait for media | Median media slack before PCM |
|---|---:|---:|---:|---:|
| Fast | 515 ms | 1,337 ms | 822 ms | 0 ms |
| 1.2 s activity wait | 1,686 ms | 1,312 ms | 0 ms | 373 ms |
| 1.2 s PCM-generation wait | 1,712 ms | 1,333 ms | 0 ms | 385 ms |

Each column is an independent median; use an individual trace for an exact timeline.

- With fast PCM production, media is on the critical path. The existing publication call took about 149 ms. Removing that duration entirely, with other phases fixed, gives a roughly 149 ms **counterfactual bound**, not a demonstrated Rust optimization. The 822 ms total media wait also includes room connection and subscription; it cannot all be attributed to publication.
- With either delayed step, the automatic flow had subscription ready before PCM in **every measured sample**. Making publication earlier alone would not improve first audio in those samples.
- Participant discovery was effectively immediate in this client-triggered dispatch flow: the user was already in the room when the agent joined. Moving publication ahead of that participant wait alone does not address the observed delay.
- `AgentSession.start()` completion and first received audio are different milestones. The report measures both rather than treating the start return as user-visible audio readiness.

This supports measuring a valid real provider before adding a cross-SDK initial-publication API. Existing application ordering can be addressed independently. An initial-publication change should be justified by how often real PCM arrives before media is ready, then validated against first rendered audio rather than signaling acknowledgments.

## Inspect the experience and traces

- [Aligned phase timelines](timeline.html), with a common time scale and hover timestamps; [static image](timeline.png).
- [Client milestones](clients.json), [environment/source manifest](manifest.json), and [validation record](validation.json).
- Per-run `.trace.json` files can be loaded into a trace viewer. The corresponding raw agent spans and browser marks are in `traces/`.
- Slow-activity **example order**: [UI recording](ux/controlled-activity-1200-example-order/screen.webm), [screenshot](ux/controlled-activity-1200-example-order/client.png), [received audio](ux/controlled-activity-1200-example-order/received.webm).
- Slow-activity **automatic**: [UI recording](ux/controlled-activity-1200-automatic/screen.webm), [screenshot](ux/controlled-activity-1200-automatic/client.png), [received audio](ux/controlled-activity-1200-automatic/received.webm).

The already-loaded client showed connecting feedback in approximately 40 ms. UI paint markers use two animation frames after DOM updates. Audio timing is non-silent PCM in the browser's rendering graph, not physical speaker output. Page loading, production authentication, microphone capture, STT, real inference, and physical display/speaker latency are excluded. Both processes run on this host for clock alignment; media crosses the real staging network with no injected network delay.

The worker's configured idle-process count is one; this does not characterize a cold-worker or burst-load distribution. The synthetic wait is asynchronous, unlike CPU-blocking initialization. Real event-loop stalls, when observed, remain in the trace. Recordings and remote session hosting are disabled consistently across profiles.

## Checks

- Focused existing startup tests: **22 passed** (`pytest tests/test_startup_spans.py --unit -q`).
- All 45 measured browser/agent sessions produced and rendered non-silent audio; all 45 recordings independently decoded successfully.
- Ruff formatting/lint and JavaScript syntax checks passed for the harness.
- SDK implementation files and the original dirty checkouts were left unchanged.

[Reproduction commands and methodology](../README.md).
