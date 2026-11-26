# Native audio preparation during asynchronous connection

Moving track creation into Rust removes the Python startup regression while preserving publish-on-join. Across 24 measured staging sessions, activity initialization began **102.1 → 3.8 ms** after Session start. First audible audio improved when activity or audio generation was the limiting work; immediate startup remained unchanged.

## Measured user experience

Both arms use automatic Session startup and publish one initial audio track. Before uses the previous PR implementation, which calls the synchronous native track constructor from Python before scheduling activity and connection. After supplies an audio-source description; Rust creates the track on its blocking pool during asynchronous connect while Python starts activity.

Median button click → first non-silent audio rendered by the browser:

| Controlled case | Before: synchronous preparation | After: native async preparation | Difference of medians |
|---|---:|---:|---:|
| Immediate activity and PCM | 1,302.4 ms | 1,301.6 ms | Effectively unchanged |
| Activity initialization +1,200 ms | 1,929.2 ms | 1,808.4 ms | **120.8 ms faster** |
| First PCM generation +1,200 ms | 1,909.2 ms | 1,808.0 ms | **101.2 ms faster** |

In the delayed-activity case, ready UI painted at **1,889.0 → 1,765.2 ms** and the first client message acknowledgement arrived at **1,999.3 → 1,874.9 ms**. The immediate and PCM-delay cases did not demonstrate a meaningful ready/message improvement.

Four interleaved pairs per case, six excluded warmups, no excluded measured trials. Median browser ICE RTT was **62 ms**. All 24 recordings independently decoded to non-silent PCM with ffmpeg; all received message acknowledgements, ran in warm assigned processes and completed cleanup with zero outstanding probe tasks. An old-build PCM-delay outlier at 2,874 ms remains included. With four pairs, these are exploratory measurements, not production tail-latency estimates. Paired differences and bootstrap intervals are in [before-after.json](before-after.json).

## Why immediate startup remains unchanged

Moving preparation frees the Python event loop; it does not eliminate native initialization work. In the immediate case, generated PCM already waits for publication/subscription, and that media path still dominates. Median subscription readiness was **1,177 → 1,173 ms**, essentially unchanged. Starting PCM earlier increases its wait for media instead of making playback earlier.

This comparison isolates the preparation fix: **both arms already use publish-on-join**. It cannot measure the two-RTT signaling saving against ordinary publication. The separate [native signaling trace](../native-trace-reports/README.md) verifies that JOIN carries publication and SDP, while receiver subscription still requires additional work. These results do not establish a two-RTT improvement in audible startup.

## Implementation and public APIs

AgentSession's public API is unchanged. Its output describes the source, track name and complete publication options without constructing a native track. `JobContext.connect(publish_audio_tracks=...)` forwards that description. Python RTC adds `AudioTrackToPublish` and `RoomOptions.publish_audio_tracks`; an existing local track can still use `publish_tracks`.

```mermaid
sequenceDiagram
    participant A as AgentSession
    participant P as Python RTC
    participant R as Rust FFI connect task
    participant S as LiveKit server
    A->>P: connect(source, name, publication options)
    par Native connection
        P->>R: async connect request
        R->>R: create native track on blocking pool
        R->>S: connect with prepared track and SDP
        S-->>R: joined / publication result
        R-->>P: owned track and publication
        P-->>A: connection ready; adopt track
    and Activity startup
        A->>A: initialize activity / generate greeting PCM
    end
    A->>S: send PCM when output is subscribed
```

Rust reuses its existing initial-track publication, encryption exclusion and fallback behavior. Preparation allocates no foreign track handles before success. Cancellation waits for the native callback, then releases any unclaimed room, track and publication; caller-owned sources remain caller-owned. Failed callback delivery or missing ready acknowledgement also reclaims callback-owned handles.

## Validation and scope

- Rust: native FFI build, formatting, **39 unit tests passed**, one existing live-server test ignored.
- Python: **23 focused tests passed**, plus **13 staging integration tests** covering ordinary/existing-track/source-based publication, single and dual PC, repeated audio/unpublish/destruction, rejection, successful-connect cancellation and partially invalid input.
- Agents: **160 RoomIO/startup/translation tests passed**, including native-track adoption without synchronous construction and forwarding the source description.

The benchmark uses real browser-triggered agent dispatch, real SFU media and synthetic greeting PCM. The activity/PCM delays are asynchronous waits, not CPU work. It excludes microphone capture, real LLM/TTS/STT, authentication/page download and physical speaker latency. The browser UI is already loaded. “Ready” means an interactive agent state painted by the client; the message measurement is a reliable data echo, not an LLM reply.

## Reproduce and evidence

Before revisions: Agents `29411e468`, Python RTC `0a9a8df`, Rust FFI `34d114b0`. After uses the native-preparation changes in Agents #8746, Python #963 and Rust #1732. [manifest.json](manifest.json) records source snapshots and native hashes; both binaries use the dev build profile. The main Rust fast-publication optimization (#1704) is present in both binaries.

With separate before source trees/native library, matching after source overlays/native library and staging credentials set in the environment:

```sh
python examples/startup_tracing/run.py \
  --before-agents /path/to/agents-before \
  --before-rtc /path/to/python-before \
  --before-native /path/to/before/liblivekit_ffi.dylib \
  --compare-preparation --repetitions 4 --output /tmp/native-preparation
python examples/startup_tracing/compare.py /tmp/native-preparation \
  --output /tmp/native-preparation-report
```

[Samples](samples.csv), [paired comparison](before-after.json), [phase timeline](timeline.html), [client milestones](clients.json), [raw traces](traces/), [validation](validation.json), [received audio](audio/), and [UI recordings](ux/) preserve the evidence. UI recordings show repetition zero in each arm, rather than selected best trials.
