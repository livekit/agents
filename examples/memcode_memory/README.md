# MemCode memory for returning callers

An optional `Agent` example using an external memory service, as discussed in
[issue 7546](https://github.com/livekit/agents/issues/7546). This is an Agent
subclass, not a model or media provider plugin.

## Application wiring

Resolve the caller from your authenticated application database before creating
the agent. Do not use a caller-supplied phone number, room metadata, participant
name, or model argument as authorization. Provision a separate MemCode space
for each user and credentials that authorize the mapped actor and space.

```python
from memcode_sdk import AsyncMemcodeV2Client
from examples.memcode_memory.agent import ReturningCallerAgent
from examples.memcode_memory.memory_service import MemoryService

# caller comes from an application-controlled, authenticated identity resolver.
client = AsyncMemcodeV2Client(
    api_url="https://memory.memcode.in",
    api_key=caller.memcode_api_key,
    timeout=1,
)
memory = MemoryService(client, caller.space_id, caller.user_id, caller.actor_id)
agent = ReturningCallerAgent(memory)
# Pass agent to your existing configured AgentSession.start(...).
# Close client with await client.close() when the call ends.
```

`on_enter` starts a bounded prefetch and returns immediately. The greeting has
no recalled facts. `on_user_turn_completed` awaits that task before the first
LLM response to the caller, so this can add up to the remaining recall timeout
to that turn. Errors let the call continue. `on_exit` cancels unfinished work.
There is no per-turn transcript search or background save.
The bounded recall set is cached for this call and added to each turn's temporary
context without issuing another search or changing persistent conversation history.

Search sends a fixed preferences query to MemCode. Original chunks are excluded;
only results with the authorized space and exact user provenance are added to
the chat context as untrusted application reference data, at most five facts of
1000 characters each. The current call's messages and audio are never ingested.

## Explicit saves

From an authenticated application form, preview the exact fact and obtain human
approval, then call `await memory.save_approved_fact(content)`. This method is
deliberately not registered as a model tool. A model-generated `approved=True`
is not approval. The receipt is queued, not searchable yet; use
`await client.get_ingest_status(receipt.id)` until completed, and handle failed
or cancelled jobs. Writes are idempotent and not automatically retried.
Manage retention and deletion through your authorized MemCode lifecycle flow.

## Offline tests

From the repository root, with the dependencies and pytest/pytest-asyncio installed:

```bash
python -m pytest -q examples/memcode_memory/test_memory.py
```

Tests use the real Agent hooks and ChatContext with a mocked memory boundary.
They cover returning callers, isolation, explicit writes, idempotency, provider
failure, nonblocking entry and task cancellation. No LiveKit server, model,
audio device, or MemCode service is required. Live voice smoke testing is pending.
