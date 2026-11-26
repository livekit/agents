# Actual pre-PR versus post-PR startup benchmark

**Historical result — the synchronous-preparation regression below is addressed by the [native preparation follow-up](../native-preparation-reports/README.md), which reruns the comparison on staging.**

**Application parallelization improves delayed startup. The SDK preparation patch currently regresses the already-automatic flow and should stay draft.** This comparison uses actual pre-PR source trees and a separately built pre-PR native library, rather than disabling a feature on the new build.

72 measured browser-triggered staging sessions, 12 excluded warmups, six repetitions per arm/case. All 72 received non-silent audio and a client-to-agent message acknowledgement. All assigned agent processes were warm. Median browser ICE RTT: **61 ms**. No browser errors or excluded measured trials.

**Follow-up native signaling verification:** [staging trace report](../native-trace-reports/README.md) confirms JOIN publication is active and publisher signaling is faster; receiver subscription remains a separate wait.

## Application before/after

Both sides use custom audio output. Before: connect and wait for the user, start Session/activity, then start the output. After: start connection, activity and prepared output concurrently, as in the local application patch. Both generate the same greeting from `on_enter()`.

Median button click → first non-silent rendered audio:

| Controlled case | Before PRs + original sequencing | After PRs + parallel sequencing | Difference of medians |
|---|---:|---:|---:|
| Immediate activity and PCM | 1,338 ms | 1,336 ms | Effectively unchanged |
| Activity initialization +1,200 ms | 2,589 ms | 1,911 ms | **678 ms faster (26%)** |
| First PCM generation +1,200 ms | 2,453 ms | 1,909 ms | **544 ms faster (22%)** |

In the activity-delay case, the client painted the agent-ready indicator at **2,398 → 1,873 ms** (525 ms faster), and received its first message acknowledgement at **2,508 → 1,969 ms** (539 ms faster). First-message acknowledgement is a reliable data-channel echo after the client observes an interactive agent state; it is not an LLM response. The echo handler is identical in both builds.

The fast case does not establish a playback improvement: the paired median saving was 9.5 ms and its exploratory bootstrap interval crossed zero (−37 to +50 ms). All six paired audio savings were positive in both delayed cases; their paired median bootstrap intervals were 623–760 ms and 465–833 ms. These small-sample intervals describe this run, not production latency guarantees.

## SDK changes alone

This control uses automatic `AgentSession.start()` on both sides. Before: the unmodified pre-PR Agents/RTC/native builds. After: the proposed prepared-publication builds. This separates benefits already available through application ordering from benefits introduced by the SDK patch.

| Controlled case | Before: existing automatic Session | After: prepared automatic Session | Difference of medians |
|---|---:|---:|---:|
| Immediate activity and PCM | 1,332 ms | 1,358 ms | 26 ms slower; no demonstrated gain |
| Activity initialization +1,200 ms | 1,818 ms | 1,902 ms | **84 ms slower** |
| First PCM generation +1,200 ms | 1,854 ms | 1,954 ms | **100 ms slower** |

The median delay from `AgentSession.start()` entry to activity initialization beginning was **3.7 ms before**, **106.6 ms after** with default output, and **103.3 ms after** with custom prepared output. The extra interval occurs where the patch synchronously calls `_prepare_track()` before scheduling activity initialization and connection. This localizes the regression to early track creation; the trace does not separately time the native factory's internals.

In the activity-delay control, ready UI and first-message acknowledgement also regressed by 92 ms and 98 ms. The PCM-delay ready/message differences were smaller and noisier. Prepared runs supplied an initial track and made zero ordinary `publish_track()` calls, so the absence of a UX win is not caused by accidentally using the normal publication path.

**Recommendation:** keep the application sequencing and early greeting changes. Before merging the SDK preparation path, remove its blocking preparation from the event loop/startup critical path, preserve cancellation ownership, and rerun this exact comparison. These results do not demonstrate a two-RTT saving in audible startup.

## Exact versions and controls

| Layer | Before | After |
|---|---|---|
| Agents | `4dc50d424` | `29411e468` (SDK changes in #8746) |
| Python RTC | `407502d` | `0a9a8df` (#963) |
| Rust/native FFI | `ef9a7042` | `34d114b0` (#1732) |

The Rust baseline already includes ordinary fast publication from #1704. This comparison isolates the latest three PRs; it does not estimate #1704's separate contribution. Both native libraries used `cargo build -p livekit-ffi` with the dev profile. Their distinct hashes are in [manifest.json](manifest.json). Worker setup records verify that baseline processes imported the old source trees without `publish_tracks` and after processes imported the new source trees with it. SDK distribution version strings alone are insufficient because source overlays are used.

Four arms rotate order within each repetition: before/original sequencing, before/automatic Session, after/prepared default output, after/prepared custom output. Each arm has its own first warmup excluded. Before and after use separately registered workers, the same Python dependency environment, loaded JS client, browser, staging URL, greeting PCM, source format and output queue configuration. All request single-PC; native stats confirm publisher stats and no separate subscriber-PC stats in all measured runs.

Activity delay is an asynchronous 1.2 s wait in toolset setup; PCM delay is an asynchronous wait before generating the first tone frame. Neither models CPU-blocking application initialization. No network delay was injected. Source preparation/publication is real, and all audio crosses the staging SFU. There are no microphone/camera capture, production authentication, STT or real LLM/TTS calls. Real-provider greeting timing remains unverified; the previously invalid provider key was not retried. All arms use `on_enter()` for a comparable greeting trigger, so this test does not independently estimate moving a greeting there from another application callback.

The UI is preloaded. “Ready” means the first listening/thinking/speaking state observed and painted by the client, not guaranteed audio subscription. Playback means non-silent PCM in the browser rendering graph, not physical speaker output. Client login/page-load latency is outside the measurement. Browser/server processes share a host clock; media and signaling cross the real staging network.

## Review the evidence

- [Before/after summaries and paired samples](before-after.json), [all samples](samples.csv), [client milestones](clients.json), [timeline](timeline.html), [audio/path/cleanup validation](validation.json), [raw traces](traces/), and [baseline worker import verification](worker-before.jsonl).
- Activity-delay, original application: [UI recording](ux/controlled-activity-1200-before-example-order/screen.webm), [received audio](ux/controlled-activity-1200-before-example-order/received.webm).
- Activity-delay, new application: [UI recording](ux/controlled-activity-1200-after-prepared-custom/screen.webm), [received audio](ux/controlled-activity-1200-after-prepared-custom/received.webm).
- SDK control before: [UI recording](ux/controlled-activity-1200-before-automatic/screen.webm); SDK after: [UI recording](ux/controlled-activity-1200-after-prepared/screen.webm).

Recordings illustrate repetition zero, not selected best cases. All 72 measured recordings were independently decoded with ffmpeg and contained non-silent PCM. Per-run validation also verifies one native connect, zero/one initial tracks, exactly one/zero ordinary publish calls, an agent-side probe receive/send, completed cleanup with zero outstanding probe tasks, and warm process assignment. The 2,023 ms/2,286 ms after-default outliers and 3,002 ms original-sequence outlier remain included. With only six repetitions, p95 values in the generic analysis are effectively maxima and should not be treated as stable tail estimates.

## Reproduce

Prepare detached source trees at the revisions above and build each native version into a separate target directory. Use a shared Python environment with the Agents/provider/browser dependencies installed. Set the three `LIVEKIT_*` staging credentials outside shell history, then:

```sh
export PYTHONPATH=/path/to/agents-after/livekit-agents:/path/to/python-after/livekit-rtc
export LIVEKIT_LIB_PATH=/path/to/after/liblivekit_ffi.dylib
export STARTUP_BEFORE_NATIVE_REV=ef9a7042
export STARTUP_AFTER_NATIVE_REV=34d114b0
python examples/startup_tracing/run.py \
  --before-agents /path/to/agents-before \
  --before-rtc /path/to/python-before \
  --before-native /path/to/before/liblivekit_ffi.dylib \
  --repetitions 6 --output /tmp/agent-before-after
python examples/startup_tracing/compare.py /tmp/agent-before-after --output /tmp/agent-before-after-report
```

The comparison changes benchmark instrumentation only; it does not change production SDK implementation between or during trials.
