# External boundary resilience audit

This pass reviewed the worker's LiveKit control protocol, RTC SDK enum conversion, HTTP and
WebSocket framing, MCP lifecycle, native provider parsing and realtime fallback. It follows
the plugin input audit in `realtime-provider-contracts.md`. The focus is unexpected input,
peer disappearance, cleanup and request settlement; internal invariant assertions are not
removed merely because their spelling includes `expect` or `unreachable`.

## Reproduced and fixed

| Boundary | Assumption and failure | Result after the fix |
| --- | --- | --- |
| Worker registration | The first decoded protobuf oneof must be registration. A future oneof member decodes as `None`, rejects the connection and can exhaust reconnects. | Unknown members are skipped before and after registration. They neither register the worker nor extend its handshake deadline. Known invalid handshake messages still fail. |
| MCP transport close | The SDK never calls `send` after `close`. A late send panicked. | A late send returns `McpError::Unavailable`; repeated close and receive remain settled. |
| Duplex input transcription | Any finite timestamp fits a Rust duration and wall clock. A large finite provider timestamp panicked. | Checked wall-time conversion falls back to the session clock, preserves the transcript and allows the next event. |
| Duplex output alignment | Provider span/audio offsets can be added and subtracted without overflow. Extreme offsets panicked in debug builds and wrapped in release builds. | Saturating timeline arithmetic avoids wrapping; retained words are still emitted and their backlog is released. |
| Realtime fallback | An unexpectedly ended child always emits a fatal error first. Bare EOF bypassed a healthy fallback. | Bare EOF enters the existing failover path as a connection error. A healthy backup is used; with none left, one terminal error is returned, then EOF. Explicit close does not mark a provider down. |

Each finding has a regression that failed against the previous implementation. Tests use
actual unknown protobuf wire fields, real transport/session adapters and the production
segmenter and fallback policy. The realtime DST exercises the voice executors and production
provider bookkeeping over fake ports.

## Existing protections retained

- RTC SDK numbers unknown to this build already map to ABI defaults: unknown disconnect
  reason, disconnected connection state, standard participant kind, joining participant state
  and unknown track source. Option validation remains strict where accepting a new value could
  widen an allow-list.
- Unknown worker control messages after registration are already ignored. Malformed protobuf
  and oversized control messages become transport errors; they are not treated as valid jobs.
- Shared HTTP, WebSocket and SSE readers bound message/body assembly and report framing and
  decoding failures as typed errors. Unknown socket messages do not count as a provider reply.
- Native realtime parsers already ignore unknown event/item variants. Invalid fields that are
  required to interpret a known event remain protocol failures, including unsupported response
  statuses; a response cannot be silently counted as successful when its outcome is unknown.
- Provider retry/fallback tests cover refused connections, mid-stream failure, partial output,
  exhausted providers, reconnect, cancellation and cleanup. Already-delivered ordinary streaming
  output is not replayed as a fresh attempt.

These checks use deterministic fakes, mocked SDK transports and local peers. They do not
certify every future protocol extension or constitute live outage testing against hosted
services. When every usable provider is unavailable, the required outcome is a settled,
reported failure and cleanup, rather than a promise that a reply can still be produced.

## Follow-up: jobs, workers, inference and credentials

The second pass reproduced failures at these boundaries before applying their fixes:

| Boundary | Failure | Regression contract |
| --- | --- | --- |
| Job acceptance | A decision task died; its lost oneshot became a no-op, retaining the pending job and reserved capacity. | Refuse that job, release its slot and answer the next offer. Generation checks still reject late completions from retired connections. |
| Job launch | An inline host callback panic unwound the worker loop. Catching it alone could leave a child that started before its receipt was lost. | Report the assignment failed, first attempt termination of a possibly committed launch, and continue accepting other jobs. The fake records child creation before panicking and checks termination. |
| Job termination and load sampling | A panic in either host callback unwound the worker. | Contain the callback failure, retain normal load-failure/admission policy and keep the worker responsive. No success is fabricated for a failed termination. |
| Token refresh | Rebuilding a verified token from selected fields dropped current grants and unknown signed restrictions, including nested restrictions. | Verify the original signature, issuer and validity window, retain the complete claim document, and change only `nbf` and `exp`. Preserve the signer's required room/identity validation. |
| Token lifetimes | Room, inference and A2A signing forwarded unrepresentable TTLs to an SDK duration addition that panicked. | Refuse the lifetime with a typed error before signing; normal credentials remain unchanged. |
| Interruption inference | Non-finite, negative or oversized detector durations passed construction and could panic when opening a stream or produce unusable buffer settings. | Refuse invalid constructor values through the existing invalid-input boundary. Valid durations and the three-second audio-window limit retain their existing behavior. |
| Dev reload | One invalid retained token or malformed job payload aborted the entire reload batch and closed the CLI channel. | Skip individual invalid entries, reload the healthy jobs on either side, and continue answering CLI requests. |

Worker tests use the actual executor with injected transport and host ports, paused time and
wire-level assertions. Credential tests use synthetic keys; they cover permission/restriction
preservation, foreign signatures, expired and not-yet-valid tokens, invalid room-join claims
and oversized lifetimes. The dev-channel test exchanges real local protocol frames and checks
both reload outcomes and a subsequent request. JWT signing uses the pinned library and the
same HMAC provider installed by LiveKit's verifier; it does not implement cryptography locally.

Callback containment handles unwinding panics and lost tasks. It cannot make an unavailable
external host successfully terminate a process; a failed cleanup is reported. Application
callbacks that intentionally keep waiting retain their existing lifecycle and cancellation
contracts. This pass does not add a new acceptance deadline or change the documented trust
model for endpoint tokens arriving through a LiveKit node or an insecure direct listener.

## Follow-up: state-owned job operations

Job phases now own their in-flight operations rather than relying on parallel bookkeeping:

- `AwaitingUser(Decision)` holds the scoped callback task and its completion future. The
  callback owns the signal's producer; task death ends that signal and yields rejection.
  Removing the pending phase cancels the callback through `ScopedTask`'s drop. There is no
  separate decision-task table or decision waiter in the worker's general operation queue.
- `AwaitingAssignment` holds the acceptance arguments and acceptance time; a job that has not
  been accepted carries neither as placeholder state.
- `Launching(PendingLaunch)` holds its job information, launch future and any status received
  before the launch receipt. These cannot be cloned into independent bookkeeping records.
  Launches remain owned across worker reconnects, preserving the existing migration behavior.
- The worker polls decision and launch futures directly from their owning phases. Removing a
  phase also removes its waiter, so cancelled operations cannot leave stale completions in a
  separate queue. Reserved capacity remains derived from the pending records.

Shutdown closes decision-task admission, removes pending phases and joins the cancelled
callbacks before publishing the final snapshot. It clears reserved capacity and reports the
worker as disconnected. Joining is bounded by the existing connection shutdown timeout;
cleanup that fails to settle is reported rather than silently called complete.

Two regressions failed against the earlier implementation: removing a pending decision alone
left its callback running, and closing the worker left reserved capacity in its final snapshot.
The worker suite also checks lost decision tasks, failed/committed launches, early child status,
parallel slow callbacks, reconnect migration, drain and repeated assignment handling. These are
worker subsystem tests over injected ports and paused time; the voice DST does not run the
worker control-plane executor.

The same pass reproduced an endpoint-shutdown callback panic skipping control-transport close
and final worker settlement. Shutdown now reports that callback failure and still cancels the
endpoint tasks, closes the transport and publishes the terminal snapshot. Its regression
checks the close receipt and terminal state after fault injection.

Final review also reproduced a terminal child status being overwritten by a late running
status before the launch receipt arrived. The worker then retained a completed job and drain
timed out. Pending launches now preserve their first terminal status. The regression exercises
both success and failure followed by running, and verifies drain waits for the launch receipt
before completing with no active jobs.
