# STT protocol contracts

Run all local WebSocket contracts:

```sh
uv run pytest tests/stt_harness --unit
```

Run one provider, or one behavior:

```sh
uv run pytest tests/stt_harness/test_contract.py --unit -k deepgram
uv run pytest tests/stt_harness/test_contract.py --unit -k cancel
```

No credentials are required. Clients connect to a local HTTP/WebSocket server.
The loopback guard rejects other socket destinations. TLS clients trust a fresh
local test certificate. The SDK's request encoding and response parsing stay real.

## Ownership

Each plugin owns `tests/stt_protocol.py`. Inference owns
`livekit-agents/tests/inference/stt_protocol.py`. Discovery reads their `PROVIDERS`
entries. Add separate entries when models use different protocols. The same
catalogue supplies wire, SDK, optional fault, and live-test drivers. Live entries
own `required_env`, `live_options`, and `live_options_from_env`; there is no second provider list.
A single AST check checks streaming-provider registration and coordinator use.

A protocol implements configuration, audio decoding, transcript encoding, and
completion ordering. Cite the provider specification in `sources`. Do not import
production serializers, parsers, or stop-message constants. A test must detect
when production code sends the wrong message.

`finish_turn()` emits an ordinary turn final while input stays open. Override it
when a turn final must follow a client flush. Keep terminal replies separate.
Set `require_completion=False` if an idle stop has no acknowledgement; the final
response budget then bounds a successful quiet drain. Transport failures still
raise errors. `reconnect_on_idle_close` identifies providers that can safely retire
a connection after a final and open a new one for the next utterance.

Set `drain_quiet_timeout` for protocols that can finish a silent drain before the
final-response deadline. A known turn must still receive its final or time out.
The shared tests hold that final beyond the quiet period to check this rule.

The client factory constructs the real STT client and redirects its endpoints to
`STTServer`. It can supply test keys and local input fixtures. Do not replace
`send`, `receive`, or SDK event iterators with mocks in this factory.

The shared driver sends PCM through `push_frame`, calls `flush`, `end_input`, and
`aclose`, and consumes public events. The server checks every decoded audio byte,
including the short final frame. Its reader can answer pings while final results
are held. Receipt acknowledgements can precede those results. A server assertion
fails the test even when the client retries or falls back.

Shutdown checks run before the harness closes any client session or server socket.
Non-pooled streams must release their connections at `stream.aclose()`. Providers
marked `pools_connections` may retain them until `STT.aclose()`, which must release
every connection, including idle prewarmed sockets. The SDK fixtures also track
stream cancellation and transport closure before their own cleanup runs.
Fallback recovery streams belong to the adapter and are checked after it closes,
before closing the underlying providers.

## Coverage and limits

The common scenarios cover active transcription, delayed final results, flush,
active and draining cancellation, remote close and abort, unanswered probes,
answered client and server pings, bounded retries, fallback, empty input, and
long-lived input, and retry recovery after useful provider traffic. OpenAI also
has an out-of-order commit-completion scenario. Mistral contracts cover both Voxtral
generations, server turn events, and diarized segments. Flush completions leave the
stream open; a separate end-of-input completion closes it. SLNG keeps a separate replay
budget: its contract requires a final before buffered-audio replay debt clears.

`completion_unknown_reason` records a missing provider guarantee. These providers use a
bounded best-effort drain: normal `end_input()` does not emit an error when the
grace period expires. Write failures, abnormal closes, and failed liveness probes
still fail. The tests verify final audio bytes and events without inventing a
completion acknowledgement. This applies to Fireworks, Gnani, Nabrah, legacy
Sarvam, Simplismart, SLNG, older Baseten Whisper runtimes, and the Telnyx Google
and Azure engines. Baseten completion becomes required once the server acknowledges
`end_audio`; missing `finished` then fails within the final-response budget.

`completion_on_close=False` identifies protocols that require a terminal event or
correlated final results. Their tests also close a healthy socket without that
response and require an error. A transport close cannot stand in for that event.

`uses_websocket_monitor=False` identifies clients whose SDK owns heartbeat timing. Their real
ping/pong and close behavior is tested, but the common short liveness deadline is
not applied to them.

AWS, Azure, Google Cloud STT, and NVIDIA use native SDK transports rather than
WebSockets. Their colocated `SDKProvider` entries in `PROVIDERS` supply boundary fixtures to:

```sh
uv run pytest tests/test_stt_sdk_contract.py --unit
```

These are SDK-boundary tests, not wire-protocol tests. Both suites observe public
STT events and terminal errors. Failed stop writes, blocked transport writes, and
long idle regressions under virtual time remain in `tests/test_stt_stream_contract.py`;
its fixtures are optional metadata on discovered provider entries. Ordinary lifecycle
scenarios belong in this harness.

The coordinator and monitor timing rules use virtual time in their own unit
tests. Long-input and quiet-server scenarios run on a representative shared-monitor
provider. Every provider still runs protocol, drain, cancellation, close, retry,
and fallback checks. Wire tests use short final-response budgets and event barriers
where possible; they do not advance virtual time over real socket or SDK-thread I/O.
`APIConnectOptions.final_response_timeout` is independent of the connect `timeout`
and is preserved by the fallback adapter's `attempt_timeout`.

Real service validation remains separate and opt-in:

```sh
uv run pytest tests/test_stt_stream_live.py --stt -k deepgram --show-capture=no
```

That runner requires the provider's credentials and can incur API charges. The
local protocols check client behavior; they do not prove that a service still
implements its published protocol. Use the live runner when changing a protocol.
