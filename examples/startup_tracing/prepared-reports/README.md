# Prepared agent audio publication

**Newer actual pre-PR comparison:** [before/after report](../before-after-reports/README.md). That run finds a preparation regression against the existing automatic Session flow. The measurements below used the same new SDK/native build for both paths and must not be read as a pre-PR SDK comparison.

Implemented across Rust FFI, Python RTC, Agents, and the local application. `AgentSession.start()` and the greeting API keep their existing signatures. The additive options are `rtc.RoomOptions.publish_tracks` and `JobContext.connect(publish_tracks=...)`; each entry contains a local track and its complete `TrackPublishOptions`, including codec/encoding settings. AgentSession prepares its RoomIO output automatically, without waiting for model PCM or opening capture devices.

The local application patch removes the initial awaited room/participant connection, schedules connection, activity startup and custom-output startup concurrently, and starts greeting generation in `on_enter()`. It preserves its existing audio-source settings. See [application patch](../application-startup.patch). The installed older SDK still uses ordinary publication; prepared JOIN requires the matching Python/native build.

## Staging measurements

36 measured browser-triggered agent sessions plus 12 excluded warmups, four profiles, three repetitions per controlled case. Same Rust native binary in all profiles, including the previous normal fast-publication optimization. Browser ICE RTT median: **63.5 ms**. All 36 recordings independently decoded to non-silent PCM; no browser errors. This is a small controlled comparison with synthetic greeting audio and asynchronous injected delays, not real-provider or cold-worker timing. Audio input, recording upload and session hosting are disabled.

Median button click to first non-silent rendered audio:

| Case | Original application sequencing | Parallel custom output | Automatic Session, ordinary publication | Automatic Session, prepared JOIN |
|---|---:|---:|---:|---:|
| Fast activity / PCM | 1,376 ms | 1,345 ms | 1,358 ms | 1,353 ms |
| Activity initialization +1,200 ms | 2,538 ms | 2,450 ms | 1,900 ms | 1,902 ms |
| First PCM +1,200 ms | 2,378 ms | 2,450 ms | 1,933 ms | 1,934 ms |

Automatic startup improved the delayed cases by approximately **444–639 ms** compared with the original application ordering. Prepared publication added **no consistent first-audio improvement** over automatic startup with ordinary publication in this run. The fast-case median difference of 5 ms is too small to establish a meaningful gain. Earlier publication must not be reported as a demonstrated two-RTT improvement in audible startup.

The prepared runs supplied one initial track and made **zero normal `publish_track()` calls**. Ordinary automatic runs supplied none and made a normal publication call. Both paths delivered audio. In the two delayed cases, media was already ready before PCM. In the fast case, registration was folded into connect, but overall media readiness remained similar; this experiment does not isolate the remaining transport/subscription delay. Connect completion includes committing initial publications, so publishing during connect can also move registration work onto the connect-completion path, particularly with fallback configurations.

All profiles already generate greetings in `on_enter()` and overlap generation with output setup; this benchmark does not independently estimate the gain from moving an application's greeting there. A valid provider credential is still needed to benchmark actual LLM/TTS greeting latency; the earlier invalid credential was not retried.

## Implementation and compatibility

- Native connect receives prepared tracks, uses Rust's existing initial-publication/fallback implementation, and returns their owned publication handles before forwarding room events.
- Python installs the publications and their original track objects before releasing the event barrier. Duplicate tracks fail before starting connect.
- Agents adopts the exact prepared publication rather than publishing twice. Already-connected rooms and older RTC wheels use normal publication.
- RoomIO starts output setup without waiting for input-participant discovery. Its init task owns, cancels and joins output startup.
- PCM delivery still waits for subscription; greeting generation does not. `wait_for_ready()` retains its publication/subscription semantics.
- Cancelling a prepared connect waits for the outstanding native connection attempt, then disconnects an unclaimed successful room and releases callback-owned handles. Failed SID futures receive an independent error so they do not retain the connection traceback and prepared sources.
- Single-PC enables the JOIN optimization on supporting servers. Dual-PC and other unsupported configurations follow Rust's normal-publication fallback. Defaults are preserved.

## Evidence and reproduction

[Timeline](timeline.html), [sample data](samples.csv), [summary](results.json), [client marks](clients.json), [environment](manifest.json), [audio/path validation](validation.json), and [raw agent traces](traces/).

Use the linked Agents and Python source trees with a matching native build:

```sh
export LIVEKIT_LIB_PATH=/path/to/rust-sdks/target/debug/liblivekit_ffi.dylib
export PYTHONPATH=/path/to/agents/livekit-agents:/path/to/python-sdks/livekit-rtc
# Set LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET outside the command history.
python examples/startup_tracing/run.py --profile all --repetitions 3 --output /tmp/prepared-startup
python examples/startup_tracing/analyze.py /tmp/prepared-startup --output /tmp/prepared-startup-analysis
```

The harness clears initial tracks only for the `automatic` control profile; the `prepared` profile uses automatic Session preparation. Original/parallel profiles connect before starting the session. Source files, native build mode/hash and browser version are recorded; all results above used a dev native build, warm agent workers, and a preloaded browser UI. Synthetic waits do not model CPU-blocking application initialization. The normal path is already fast publication, so the prepared comparison is incremental to that optimization.

## Validation

- Rust FFI build and unit tests: **39 passed**, one existing live-server test ignored; `cargo fmt --all -- --check` passed.
- Python RTC control/publication/SID tests: **16 passed**.
- Native staging integration tests: **7 passed**, covering prepared and ordinary publication with single and dual PCs, repeated non-silent audio/unpublish/teardown, rejected connection, cancelled rejected connection, and cancelled successful connection. Weak references verified source/track/publication destruction after cleanup.
- Agents RoomIO/startup/translation tests: **159 passed**. Tests cover adopting a prepared publication without duplicate publication, normal fallback, output startup before participant arrival, and cancellation/join of that startup task.
- Changed Agents files and new Python tests pass Ruff; pre-existing style findings elsewhere in Python `room.py` were not rewritten. The local application compiles; actual provider execution remains unverified without a valid provider credential.
