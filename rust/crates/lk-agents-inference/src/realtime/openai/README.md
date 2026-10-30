# Realtime verification

From the repository root:

```sh
cargo test --manifest-path rust/Cargo.toml -p lk-agents-inference --lib realtime_
cargo test --manifest-path rust/Cargo.toml -p lk-agents-runtime --lib realtime::
cargo test --manifest-path rust/Cargo.toml -p lk-agents-runtime --test realtime
cargo test --manifest-path rust/Cargo.toml -p lk-agents-runtime --test realtime_ctx_diff_scaling -- --nocapture
```

Core tests cover connection generations, acquisition cleanup, pending response deadlines,
acknowledged versus desired history, pushed items, and reconnect replay. Runtime tests use
controlled sockets and image preparation to exercise malformed input, phase errors,
stalled writes, cancellation, and stale preparation results.

The scaling test measures allocation and elapsed time for 1,000 and 10,000 items, including
identical, append, reorder, repeated pending updates, and reconnect replay. It counts
allocations including freed scratch space, so repeated quadratic allocation cannot hide
behind a bounded peak.

Admission defaults:

| Resource | Bound | Overload |
| --- | ---: | --- |
| Pending responses | 32 | Correlated request failure |
| Cancelled response identities | 64 | Recycle the connection |
| Fallback replay requests | 32 and 1 MiB | Correlated request failure |
| Command queue | 128 | Terminal session refusal |
| Event queue | 256 and 32 MiB plus one reserved terminal observation | Terminal session refusal |
| Socket write batches | 32 | Terminal session refusal |
| Successful response write receipts | 32 | Writer waits independently of controls |
| Context continuation | 64 item operations | Yield between admitted pages |
| Inline image URL | 16 MiB before decode | Typed image preparation refusal |
| Incoming wire frame | 16 MiB | Typed malformed/oversize failure |
| Socket close | 5 seconds, one retained cleanup per session | Session stops; typed framework cleanup diagnostic |

Codec and image work uses the process worker service. The state machine performs no
serialization, image resolution, base64 conversion, resampling, or transport I/O.

`OpenAIRealtimeSession.cleanup_observation()` reads the reserved typed cleanup diagnostic. An incomplete
close keeps ownership until transport settlement and does not raise a provider error.
