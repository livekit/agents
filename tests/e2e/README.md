# E2E journeys

Run from the repository root against the next package and its built extension (paths before
category flags, or `--<category> [NAME]` takes the path as its name):

```sh
PYTHONPATH="$PWD/livekit-agents-next:$PWD" uv run --no-sync pytest tests/e2e --unit
PYTHONPATH="$PWD/livekit-agents-next:$PWD" uv run --no-sync --env-file project.env pytest tests/e2e --e2e
```

`test_endpoints.py` registers its own worker under a unique agent name and calls it through the
node's `/agents/{agent}/{deployment}/` front; against a local development node, also set
`LIVEKIT_WORKER_INSECURE=1`.

`--unit` runs the hermetic harness checks (`test_harness.py`, `test_selection.py`). `--e2e` runs
the journeys; they need AgentDB and `LIVEKIT_URL`/`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET`, and a
missing prerequisite fails the run instead of skipping it.

## Layout

- `conftest.py`: fixtures owning two worker processes, per-test threads, the recording service
  and RTC room/caller. Cleanup runs on failure and cancellation and reaps the children.
- `worker.py`: the application each worker serves, on the production `AgentServer` HTTP app.
  Only the model and the application service are scripted.
- `drivers.py`: A2A actions and wire observations; no verdicts.
- `resources.py`: process, service and caller ownership, and the harness bounds.

The application service records each call (worker, thread, operation, exact terms): those calls,
the caller's replies and the stored lane history are the oracles, never model text or logs.
There are no retries; every action is bounded by `REQUEST_SECONDS`.

## Journeys

| Journey | Expected |
| --- | --- |
| Thread across two workers + retry | B recalls A's history; a resent message keeps its task, no extra turn |
| Text interruption | B's message interrupts A's blocked tool; A delivers no answer, B does |
| Inbox burst | Two SMS messages form one turn, one answer and a `mergedInto` link |
| Worker killed mid-turn | B's next turn settles A's task as worker-lost and answers; no booking |
| Input required answered on B | The answer becomes the pending tool's output; no second question |
| Approval yes/no on B | Nothing before the answer; the exact booking once after yes, none after no |
| Stateless agent, two threads | Concurrent threads on one worker keep their own history |
| Thread lifecycle | Create, update attributes, list by scope, delete, refuse deleted reads |
| HTTP through the node (`test_endpoints.py`) | Bodies both ways, streaming, a late failure is never a clean body, 100 concurrent |
| WebSocket through the node | Echo, 1 MiB message, pings, close codes both ways, 403 refusal, 50 sessions, a vanished client is 1006, drain closes with 1001 |

`test_pending.py` lists journeys the implementation cannot run yet, as non-running xfails named
after the scope item that enables them (I-8, H-13/K-23, H-6/G-6, L-6/I-20).
