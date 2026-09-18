# GPT-Live

Two agents for the OpenAI GPT-Live full-duplex voice model: the ordinary one, and the variant where this process does the reasoning.

For setup instructions and more details, see the [main examples README](../../README.md).

```bash
python gpt_live_agent.py console
python client_delegation.py console
```

## Voices

Both examples use `voice="marin"`. `GPTLiveVoices` also offers `aster`, `beacon`, `cinder`, `stone`, and `vesper`. Other supported names and custom voice objects still pass through to the API.

## Context acknowledgments

`append_instructions`, `append_thinking`, and `append_commentary` queue context and return without waiting for an acknowledgment. Their `session.*.appended` events arrive at the estimated context-injection end. They do not mean speech has finished. The plugin does not gate later commands on these events or apply an acknowledgment timeout. The adapter uses output audio to determine when speech ends.

`session.closed.reason` accepts `close_requested`, `expired`, `content`, `remote_hangup`, and `connection_lost`. The plugin logs the reason and collects the final usage for each close event.

## Delegation

GPT-Live listens and speaks at the same time, but it does no reasoning and runs no tools of its own. When the conversation needs either, it delegates. Where that work goes is fixed when the session opens and cannot change afterwards.

| file | `delegation=` | who does the work |
| --- | --- | --- |
| `gpt_live_agent.py` | `"responses"` (the default) | a backend Responses model |
| `client_delegation.py` | `"client"` | your own process |

## Responses delegation — `gpt_live_agent.py`

The backend model calls the agent's `@function_tool` the usual way, so this reads like any other voice agent. `responses_options` selects that model and gives it its own instructions, separate from the voice persona.

The example also seeds a prior conversation as startup history and adds `WebSearch()`, a hosted tool the backend runs with no client round trip.

## Client delegation — `client_delegation.py`

The example consumes `livekit.agents.ClientDelegation`, an SDK toolset built on
`AsyncToolset` and the existing tool executor. GPT-Live client delegation becomes
an internal `lk_agents_delegate` call, so registration, cancellation, draining and
`tool_execution_updated` events use the ordinary session lifecycle. There is no
example-owned task executor or mandatory routing LLM.

```python
from livekit.agents import AgentSession, ClientDelegation, inference
from livekit.plugins.openai.realtime import GPTLiveModel

backend = ClientDelegation(
    model=inference.LLM("openai/gpt-5.5"),
    tools=[check_order_status, lookup_weather],
    select_task=lambda request: "desk",
)
session = AgentSession(llm=GPTLiveModel(delegation="client"), tools=[backend])
```

Put backend tools on `ClientDelegation`, not directly on the voice agent. The voice
model does not see their schemas. A toolset on `AgentSession` retains its context
and running work through agent handoff; one on `Agent` has activity scope. Supplied
models and external backend connections remain caller-owned and should be closed
when their owning application ends.

### Start or update backend work

`select_task(request)` is a synchronous application decision. The request carries
its delegation ID, connection identity, transcript and conversation snapshot.
Returning the same task key advances that task's revision; distinct keys preserve
independent work. The default assigns a fresh task to each request. A conversation
backend can use one key and interpret corrections and outstanding independent
requests from its retained context. This is not automatic semantic per-topic
cancellation: deciding whether Monday became Tuesday still belongs to the backend.

Revision advancement happens when the event arrives, before execution can yield.
`DelegationContext` exposes `task_id`, `revision`, `request`, `is_current`, a retained
`chat_ctx` snapshot, and the normal `run_context` for session/userdata access.
The default LLM loop retains tool calls and actual outputs, observes already-running
tool outcomes before planning another step, and refuses new tool execution from an
obsolete revision. Updating an open transcript replaces its existing message;
it does not duplicate the accumulated words or mutate earlier request snapshots.

For an existing long-lived backend, provide `handler` instead of `model`, or
override `ClientDelegation.delegation_node(ctx)`. The handler receives each new
revision and can call its backend's start-or-steer interface. No second model is
required to classify the request. Use `ctx.execute_tool(name, arguments, call_id=...)`
to run tools through the shared executor and retain their actual outcomes.

### Progress, results, and closeout

- `await ctx.update(text, silent=True)` sends quiet context.
- `await ctx.update(text)` sends a speakable chunk.
- Returning text sends a final result; returning `None` closes out without speech.
- Exceptions produce an error terminal in the shared tool lifecycle. `ToolError`
  messages can be returned to GPT-Live; unexpected exception details stay in backend
  lifecycle evidence rather than being spoken.

Every chunk is checked for task revision and connection identity both before
queuing and immediately before the WebSocket sends it. Previously queued output
cannot migrate to a new connection, even if a delegation ID is reused. Old
operations are not automatically replayed after reconnect. Appends are context
injection, not proof of speech or playback; already-sent output cannot be retracted.

Managed chunks use a conservative **500 UTF-8 byte** bound, staying within the
service's 500-token limit without adding a tokenizer dependency. Oversized chunks
fail visibly; backends should stream concise, coherent chunks rather than rely on
truncation. Low-level manual `append_commentary` remains available with its existing
service-enforced token limit.

Cancellation does not undo external actions. Backend tool calls have their own
call IDs, distinct from transport delegation IDs and application booking/payment
operation IDs. The executor keeps an already-running tool's actual outcome in
backend history and its terminal event, even if its parent is cancelled or its
answer becomes obsolete. Applications still own confirmation, business records
and reconciliation before retrying uncertain outcomes. Session shutdown invalidates
delivery, cancels cancellable work and drains non-cancellable tools.

### Existing manual integration

With no `ClientDelegation` toolset, `GPTLiveSession` retains the original
`delegation_created` event and manual append methods. Managed mode dispatches through
the toolset instead, avoiding two handlers executing the same request. Hosted
Responses delegation remains unchanged.

### Validation and upstream scope

The original example at `34a4e8f` could return Tuesday and then an obsolete Monday
answer. The regression suite retains that reproduction and drives its replacement
through a real `AgentSession`, tool executor and GPT-Live WebSocket loops using an
in-memory transport. It covers corrections, independent requests, streaming,
completion races, cancellation-resistant work, actual tool outcomes, reused IDs on
reconnect, failures, handoff and shutdown.

```bash
uv run pytest tests/test_gpt_live_client_delegation.py --unit -q
```

These are deterministic SDK/transport tests, not live model interpretation or voice
latency measurements. There is no demonstrated latency gain or continuous prefill
implementation in this change.

Draft [PR #6602](https://github.com/livekit/agents/pull/6602), reviewed at
`197cbaa`, uses the same async-tool direction and builtin tool name. This work is
scoped to client transport integration rather than copying its broader
`AgentSession(delegation_llm=...)` interface or adding another general task executor.
Adjacent open PRs #7230, #7234, #7229, #7239, #7231 and #7238 cover dispatch,
continuation, turn finalization, input clock, usage and playback respectively; their
fixes are not included here.
