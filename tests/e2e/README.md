# E2E foundation

Run from the repository root against the next package and its built extension:

```sh
PYTHONPATH="$PWD/livekit-agents-next:$PWD" uv run --no-sync pytest tests/e2e --unit
PYTHONPATH="$PWD/livekit-agents-next:$PWD" uv run --no-sync --env-file /path/to/project.env pytest tests/e2e --e2e
```

Install/build the workspace separately before running. The harness does no installation,
code generation, reference build or environment repair. Put paths **before** category flags:
the existing `--<category> [PROVIDER]` syntax otherwise consumes a following path as a target.
`--e2e` is the category proposed and registered by this foundation. Its modules are excluded
before import by the existing category machinery. `V0_STACK` is unnecessary; explicitly
selected E2E tests fail when credentials or services are unavailable.

## Ownership and evidence

- `conftest.py` owns two isolated Python worker processes, TCP listeners inherited through
  bound sockets, per-test AgentDB threads, a recording service, and optional RTC room/caller
  fixtures. A partially started second worker still closes the first one. Thread acquisition
  is shielded so cancellation waits for its result and then deletes the owned thread.
- `resources.py` terminates the owned process group, escalates to kill, and awaits the direct
  child. `Child.reaped` records the wait receipt separately from process exit. Tests additionally
  check `waitpid` reports no remaining child. These workers do not launch job children; this
  does not claim G-12's process-pool/forkserver/resource-tracker reaping coverage.
- `worker.py` is a scripted application on the production `AgentServer` HTTP app, served over
  real TCP with uvicorn. Its shared Agent uses request-local scripted LLM responses. Agent/tool
  execution, approval, lane leases, storage and HTTP/SSE use the current implementation.
  Each worker has a different lease holder. This bypasses worker registration and node routing;
  it does not claim full deployment, IPC, voice or remote-delegation coverage.
- `drivers.py` sends upstream protobuf A2A messages and captures upstream protobuf replies.
  Drivers raise protocol/transport errors; they contain no scenario assertions or verdicts.
  `events()` permits observations before a stream ends, including killing a busy worker.
- The service records worker identity, thread, operation and exact terms at the application
  boundary. Approval tests require no booking before the answer, one exact booking on B after
  yes, none after no, and no repeated booking after a message retry. The result must reach the
  A2A caller. History is read from the authoritative lane, not inferred from model text.
- Worker output remains in pytest's per-test `tmp_path`, including failed startup. Use pytest's
  normal retention options or `--basetemp` to choose artifact storage. There are no retries or
  normalizers. Separate pytest invocations are separate attempts; retain their output and do
  not treat a later pass as erasing a previous failure.

Fixtures clean up in `finally`, including setup failure and cancellation. Teardown cancellation
is deferred until owned cleanup settles. Cleanup errors propagate. RTC teardown disconnects
the receiving peer, closes audio readers, deletes the room and closes the API client. Room names
are allocated before creation, so deletion is attempted even if creation loses its response.
Remote APIs that fail to return an allocation ID cannot prove remote cleanup; the fixture
fails explicitly rather than claiming success. Avatar ownership waits for H-3/H-4's assembled
transport instead of inventing a test-only product interface.

## Journey contracts

All text journeys require live AgentDB/threads services and credentials (`LIVEKIT_URL`,
`LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`). Model responses and the application service are
scripted; workers, bindings, HTTP/SSE and persistence are real. Each action must complete
inside the request watchdog below. Assertions live beside the actions.

| Journey from the v0 checks | Expected and forbidden behavior |
| --- | --- |
| Thread across two workers + retry | B recalls A's history; the same message ID retains its task; no extra handler call |
| Resubscribe on B | Task first, complete answer replay, terminal status last with matching task/thread |
| Text interruption | B's durable waiter interrupts A; no stale artifact; B delivers the replacement answer |
| Inbox burst | Two SMS messages form one handler turn; exactly one answer and a correct mergedInto link |
| Killed worker mid-turn | Kill A after external lookup entry; reap A; B's next request settles A as worker-lost and answers; no booking |
| Input-required answered on B | B receives the pending call's answer; no second question; exact email in stored tool output |
| Approval yes/no on B | Zero effects before confirmation; exact approved operation once after yes, zero after no; retry cannot duplicate |
| Stateless Agent, two threads | Concurrent requests to one worker retain independent first messages |
| Thread lifecycle | Create database, update attributes, list scope, delete, refuse deleted reads |

`test_pending.py` contains explicit non-running xfails, not passing coverage:

| Missing contract | Enabling scope |
| --- | --- |
| Worker-lost settlement on read/subscribe without another message | I-8 |
| Audio stops at receiving caller after speech interruption | H-13, K-23 |
| Spoken call with readable recording and expected media | H-6, G-6 |
| Console chat/call on the same thread | L-6, I-20 |

Remove each non-running xfail when the named slice supplies the missing driver or behavior,
then implement/execute its documented assertions. `strict=True` is retained for any subsequent
executable xfail. These entries are a coverage ledger, not evidence of an attempted journey.
The text interruption test is not evidence of caller audio interruption.

## Bounds and coverage limits

| Bound | Meaning and basis |
| --- | --- |
| Startup 30 s | Harness watchdog for importing one warm worker and opening its inherited listener; local two-worker smoke measured about 2 s including teardown |
| Request 30 s | Harness watchdog for each A2A action/service barrier; also caps whole SSE consumption, not just idle reads |
| Close 5 s, then kill + wait 5 s | Idle test worker grace and forced reaping; uvicorn grace is 1 s; the initial nine-case local failure-injection suite completed in 0.31 s |
| Debounce 0.5 s | Explicit application configuration defining the SMS burst window; requests start concurrently, without a guessed sleep |
| Lease 2 s, renewal 0.5 s | Explicit test storage configuration; crash recovery waits on production acquisition, not a fixed sleep |

These bootstrap watchdogs bound the harness. They are not calibrated product latency limits
or K-23/K-26 acceptance evidence. In particular, no caller audio-stop bound is invented here;
that journey remains xfailed until K-23 publishes one.

`test_harness.py` injects scenario failure, timeout, cancellation, repeated teardown
cancellation, worker startup failure and malformed/missing replies. `test_selection.py`
checks selection before import, selected import errors, pytest's empty-selection exit 5 and
missing-credential failure. These hermetic tests run under `--unit`/`make unit-tests`.

This is the K-18/K-20/K-21 E2E foundation. It does not close the complete smoke/fuzz migration
(K-20), replace the root category parser with the agreed static parser (K-21), or close real
RTC/recording/console acceptance. Existing `tests/test_v0_checks.py` is left untouched as required
by the assigned new-file scope; its public-door assertion predates decision 57 and is not copied.
