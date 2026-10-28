# Models and connection policy

LLM, STT, TTS and realtime providers accept the same connection policy.
Pass `conn_options=APIConnectOptions(...)` to a model operation or a provider
that accepts a default policy. An explicitly supplied policy replaces that
default in full.

## Connection options

`APIConnectOptions` is immutable and exported from `livekit.agents`.
Durations are seconds. Each timeout accepts a finite nonnegative number or
`None` to disable that limit; invalid values raise `ValueError` before I/O.

| Field | Default | Bounds |
| --- | --- | --- |
| `pool_timeout` | `10.0` | Waiting for transport capacity or a pooled connection |
| `connect_timeout` | `10.0` | Establishing a usable connection, including DNS and TLS |
| `write_timeout` | `10.0` | Writing pending bytes without progress |
| `read_timeout` | `10.0` | Waiting without progress when a remote response is expected |
| `total_timeout` | `None` | The whole operation, including admission, retries and fallback |
| `max_retry` | `3` | Additional attempts after the first; a nonnegative integer |
| `retry_interval` | `2.0` | Delay between retries; the first retry waits 0.1 seconds |

There is no `timeout` field. Read and write limits restart on transport
progress. Waiting for application input, passive listening and downstream
backpressure do not count as remote inactivity. The total deadline starts
when the operation is accepted and does not restart on progress or retry.
Unsupported applicable timeout settings fail before I/O.

A fallback adapter's `attempt_timeout` caps each phase of a backend attempt;
it does not limit the duration of a healthy stream. The operation's
`total_timeout` still bounds retries and fallback together.

## Errors and streams

`APIError`, `APIStatusError`, `APIConnectionError`, `APITimeoutError` and
`StreamStateError` are exported from `livekit.agents`. Status errors retain
`status_code`, `request_id` and the diagnostic `body`. A timeout is a
connection error; its `phase` is `"pool"`, `"connect"`, `"write"`, `"read"`
or `"total"`, or `None` when the failing layer cannot identify it. Provider
errors keep their class and diagnostic cause through an agent run.

`retryable` permits retry consideration; it does not establish safe replay.
Delivered output is not replayed on failure without a deduplication protocol.
A TTS stream speaks every flushed segment once. Provider cancellation ends
the request without retry or a provider-health penalty.

A concurrent pull of a provider stream raises `StreamStateError` without
disturbing its accepted pull. `aclose()` requests cancellation and waits for
bounded cleanup. A cleanup deadline leaves the framework responsible for
late resources and emits a diagnostic observation.

## Connection pools for plugins

`livekit.agents.utils.ConnectionPool` takes `conn_options=` for prewarming.
Its `connect_cb` receives the resolved `APIConnectOptions` and applies the
appropriate transport phases. Call `get(conn_options=...)` or use
`async with pool.connection(conn_options=...)` for each operation. The
context manager returns a healthy connection to the pool and retires it on
failure. `await pool.aclose()` releases the pool's connections.

## Native realtime context updates

Native realtime sessions retain superseded conversation snapshots until admitted cleanup
workers release them. Preparation pauses when cleanup needs capacity. Interruption, close,
incoming protocol events and already written response deadlines continue to run.

Additional `update_chat_ctx()` calls can exhaust the retained-context limit while cleanup
is waiting, including replacements of a pending target during reconnect. The session then
emits a nonrecoverable `RealtimeModelError` with message
`realtime context retirement capacity exceeded`, fails pending reply requests and closes.
Cleanup ownership remains with the framework. Coalesce successive history changes before
submitting a new full snapshot. Submitting an update does not await cleanup admission.

The limit bounds retained ownership, so the number of updates accepted depends on the
conversation and outstanding work. The runtime does not promise an unlimited burst of
context replacements. Ordinary replay of an unchanged conversation uses bounded
continuations; external item or context growth can still reach the overload limit.
