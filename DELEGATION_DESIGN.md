# Delegation — Design

Status: **design**. Scope: the current framework — pipeline agents and realtime models. The
upcoming full-duplex model (OpenAI GPT-Live) shapes the API but is not implemented here; see
[Shaped for the duplex model](#shaped-for-the-duplex-model).

## Goal

Split a voice agent into a **fast brain** and a **slow brain**. The fast brain talks to the user: it
owns the conversation, latency and turn-taking, and gets exactly one tool, `delegate`. Everything
that needs reasoning, retrieval or tool use goes to the slow brain, which never speaks — it returns
facts the fast brain phrases.

The two jobs want different models. Realtime and low-latency models are good at conversation and
weak at tool calling; frontier text models are the reverse. Today an agent picks one and accepts the
other half.

```python
session = AgentSession(
    llm=openai.realtime.RealtimeModel(),   # fast: voice, turn-taking
    delegation_llm="gpt-5.5",              # slow: reasoning, tools
)

agent = Agent(
    instructions="You are a support agent. Keep replies short.",
    tools=[lookup_order, issue_refund],    # the slow brain's tools
)
```

The fast model's tool list is `[delegate]`. `lookup_order` and `issue_refund` are never shown to it.

## The slow brain is the executive

The single most load-bearing idea, and the one that decides most of the rest: **a delegation is a
request *to* the slow brain, not a job the fast brain spawns.**

The fast brain does not know what a request means. Asked to book a flight it delegates; asked a
moment later to "change it to Tuesday" it delegates again — and at that instant nothing in the
system knows whether the second request is a rejection, a cancellation, or a cancel-and-rebook. Only
the slow brain knows, and only after it reasons.

Everything downstream follows from that:

- The fast brain cannot usefully describe the work, so nothing it says can be specific.
- The fast brain cannot manage the work, so it gets no management tools.
- The slow brain owns the work, so it needs to see and control what is in flight — and what is in
  flight are **its own tool calls**, never delegations. It never sees the envelope.

## Delegation is a tool call, and the node is the API

The mechanism is **one builtin tool**. No new event, no new session method, no new capability. The
alternative — a first-class delegation concept with its own event and result channel — has to
rebuild what the tool path already provides: tracing, duplicate handling, cancellation,
`RunContext`, history bookkeeping, and the deferred-reply coalescer. A tool call gets all of it and
works identically for a pipeline agent and a realtime model, because neither knows the tool is
special.

```python
DELEGATE_TOOL_NAME = "delegate"   # exported from livekit.agents.llm
```

The name is a constant, not user-configurable: a plugin for a model that emits delegations natively
has to synthesize a call against a fixed name. A user tool that collides with it is an error at
session start.

**What the user is exposed to is the node, not the tool.** The tool's behavior — withholding the
other tools from the fast model, building the slow brain's context, routing its calls to the
session-scoped delegation executor, translating their progress — is not expressible in user code,
and its name cannot move. The node is the framework's existing idiom for exactly this split, where
the framework owns the plumbing and the user owns the step.

```python
AgentSession(delegation_llm=llm.LLM | str | None, delegation=DelegationOptions)
Agent(delegation_llm=..., delegation=...)
```

```python
class DelegationOptions(TypedDict, total=False):
    """Configuration for the delegate tool and the slow brain."""

    instructions: str
    """The slow brain's prompt. Defaults to the agent's instructions."""
    tool_description: str
    """What the fast model reads on `delegate` — when to delegate, and to say one short line
    in the same turn. See Announcing work."""
    announce: bool
    """Whether `tool_description` asks the fast brain to say a short line in the same turn as
    the delegate call. Defaults to True."""
    timeout: float
    """Seconds before an unanswered delegation fails. Defaults to 60."""
```

The feature is inert without a `delegation_llm`: no tool is injected and every tool goes to the fast
model as it does today.

## Not blocking the fast brain

The delegate tool releases the turn immediately and **stays in flight**:

```python
async def _delegate(ctx: RunContext, task: str) -> str:
    await ctx.update(_DISPATCHED, silent=True)
    return await _run_delegation_node(ctx, task)
```

`ctx.update()` resolves the tool's first result to dispatch (`voice/tool_executor.py:318`) while the
body keeps running. `silent=True` is new: it records the output item — the fast model needs a result
for its call — but suppresses the reply, which is already what `StopResponse` means for a tool
result (`:321`).

Staying in flight rather than detaching behind a returned value is what makes the rest work.
`_inject_running_tool_calls` (`voice/agent_activity.py:2984`) injects an in-progress placeholder for
every running tool so the model "leaves the call alone" instead of re-issuing it — so the fast brain
sees the pending delegation, with its task text, on every later turn and does not re-delegate. The
same registry backs `lk_agents_get_running_tasks`, `on_duplicate`, cancellation and drain. A
detached delegation loses all of it.

`delegate` defaults to `on_duplicate="allow"`. `_check_duplicate` keys on the function *name*
(`voice/tool_executor.py:612`), so name-based dedup would make every delegation a duplicate of every
other one and serialize unrelated work. The in-flight placeholder is the real protection.

## Announcing work

There is no ack policy, no filler string and no delay knob. One rule, applied at two levels:
**text a model emits alongside a tool call is the announcement.**

| | announcement | when | cost |
|---|---|---|---|
| level 1 | the fast brain's text alongside `delegate` | immediately | free — same completion |
| level 2 | the slow brain's text alongside its own tool call | one LLM round later | one round, and only for work that turns out to be long |

Level 1 is generic and instant: the fast model writes "Let me pull that up." and the `delegate` call
in one completion, and the text streams to TTS while the tool runs. Level 2 is specific and arrives
once the intent is known — "cancelling the earlier booking first" — which is information no
dispatch-time policy could have produced. The two cover different windows and neither waits on the
other.

Level 2 goes through `ctx.update()`, not `session.say()`. The slow brain writes in its own register;
the fast brain owns the voice and persona, so it should rephrase. That also sidesteps a capability
problem — `say()` needs a TTS or `supports_say` (`voice/agent_activity.py:1325`), and only phonic
sets `supports_say`.

**Only level 1 is prompted.** Level 2 is mechanical, and falls straight out of the node's loop: text
alongside a tool call becomes `ctx.update()`, text with no tool call *is* the answer. Nothing asks
the slow brain to narrate, and if it emits no text nothing is announced. Level 1 is the opposite —
models do not reliably emit text alongside a tool call unless told to, and it is the announcement
that costs nothing, so it is worth spending prompt on. That is what `announce` toggles: whether the
default `tool_description` carries "say one short line telling the user you're looking into it, in
the same turn as the call". Off means the fast brain stays quiet and the first thing spoken is
whatever level 2 produces.

A fast answer arriving right after "let me check" needs no timer to prevent: the answer turn sees
that line in the context and phrases around it, which is what `reply_maybe_covered_template`
(`voice/tool_executor.py:64`) already exists for.

## Deliveries

Three things travel from a delegation to the fast brain, all of them on the delegate call and all of
them through the **existing async-tool delivery path**. The framework decides the content; the model
layer decides the wire form.

| kind | content | mechanism | full duplex (later) |
|---|---|---|---|
| `dispatched` | the tool's silent first result | `ctx.update(silent=True)` | nothing sent |
| `progress` | the slow brain naming its work, or a level-2 tool's update | a subsequent `ctx.update()` | `delegation.context.append` |
| `answer` | the result, or an account of why there isn't one | the tool's return, coalesced as a `_final` pair (`voice/tool_executor.py:350-354`) | `delegation.context.append` |

**Level 2 never generates a reply.** There is one delivery path and it is here. A level-2
`ctx.update()` is captured by the delegation executor, translated, and re-emitted on the delegate
call's `RunContext`, which is what puts it in this table.

Only two things are added to that path: `silent=True`, and the reply gate below. Nothing coalesces
specially — `_deliver_reply` already drains every pending update into one reply, and
`reply_maybe_covered_template` (`:64`) already instructs it to summarize only what has not been said,
so a batch holding both "still looking" and the answer produces just the answer.

### Skipping the redundant reply

`_deliver_reply` calls `generate_reply` unconditionally (`voice/tool_executor.py:544`). For a model
with `auto_tool_reply_generation=True` (`llm/realtime.py:67`) that is a redundant client-triggered
response: pushing the output already makes the model continue.

| `auto_tool_reply_generation` | behavior |
|---|---|
| `False` — pipeline LLM, OpenAI Realtime API, nvidia | today's: wait for idle, coalesce, `generate_reply` |
| `True` — google, aws, phonic, ultravox | wait for idle, coalesce, push the update, **no** `generate_reply` |

The idle wait and the coalescing stay in both rows: a half-duplex model cannot take a context update
mid-response regardless of who triggers the reply. Only the duplex model can drop them.

This changes async-tool behavior for four existing plugins, so it needs a manual pass against google
and ultravox before merge. It is a fix for them — one turn instead of two — but not one this design
gets to assume.

## Context and state

The slow brain is **stateless**: each delegation reads `agent.chat_ctx` when it is created and
accumulates nothing. `agent.chat_ctx` is the single store, as it is for every other node.

That costs nothing, because the delegate call and its answer are an ordinary tool call/output pair
in `agent.chat_ctx`. The next delegation reads
`delegate(task="where's my order") -> "123 shipped Tue"` verbatim, not a paraphrase of what the fast
brain said. And because history only appends, each delegation's `[instructions + history]` prefix
extends the previous one, so the slow model's prompt cache stays warm.

The slow brain's context is:

```
[system: delegation.instructions or agent.instructions, plus the framework's directive to
         return facts rather than a phrased reply]
[... agent.chat_ctx, including prior delegate call/output pairs ...]
[user:   task]
```

**The slow brain's own tool calls go into no conversation store.** They live in the node's local
`chat_ctx` copy for the duration of the loop, and surface as `function_tools_executed` /
`tool_execution_updated` events and a `delegation` trace span. They are work that happened, not
something either brain said. They cannot go into `agent.chat_ctx` because in pipeline mode that
context *is* the request: `lookup_order` appearing there tells the fast model it has a tool that is
not in its list, and it calls it and gets `unknown AI function` (`voice/generation.py:716`). An agent
that wants raw outputs to survive into the next delegation overrides `delegation_node`.

**Statelessness is about context, not about work.** In-flight tool calls are executor state, and
they must be shared — see [Managing work](#managing-work).

## The node

```python
@staticmethod
async def delegation_node(
    agent: Agent,
    task: str,
    ctx: RunContext,
    chat_ctx: llm.ChatContext,
    tools: list[llm.Tool],
    model_settings: ModelSettings,
) -> AsyncGenerator[str, None]:
    """Default implementation for `Agent.delegation_node`."""
    activity = agent._get_activity_or_raise()
    delegation_llm = activity.delegation_llm
    ...
    for _ in range(activity.session.options.max_tool_steps + 1):
        text, fnc_calls = await _one_step(delegation_llm, chat_ctx, tools, model_settings)
        if not fnc_calls:
            yield text
            return

        # text alongside a tool call announces the work being started; the fast brain
        # rephrases it in its own persona
        if text:
            await ctx.update(text)

        chat_ctx.items.extend(fnc_calls)
        chat_ctx.items.extend(await activity._execute_delegation_tools(fnc_calls, ...))

    raise ToolError("could not complete the request")
```

`ctx` is the delegate tool's own `RunContext`, so an override can report progress, hold the floor
with `foreground()`, or reach the session. Yielded chunks are joined into one answer — the fast brain
cannot speak a partial fact, so streaming to it buys nothing. An override is free to ignore the LLM
entirely.

## Tool routing

When a `delegation_llm` resolves, `delegate` is injected and every other tool is withheld from the
model. Everything delegates, and one flag opts a tool out:

```python
@function_tool(flags=ToolFlag.NO_DELEGATE)   # stays on the fast brain
async def send_dtmf(ctx: RunContext, digits: str) -> None:
    ...
```

There is no session-level or toolset-level switch and no opposite flag. Delegate-by-default is what
makes the same agent behave identically across all three model families: client delegation on a
duplex model leaves the fast brain no tools at all, so any other default would change meaning when
the model changes. An opt-in default would also make `delegation_llm` look inert until every tool is
annotated. Given that default, a group-level switch could only express "keep this whole toolset on
the fast brain", which for an MCP toolset — the archetypal deep-task tools — is not worth supporting.

Opting a tool out is for latency-critical work: a DTMF digit during an IVR prompt. It
is *not* needed for `end_call`, handoff or transfer; those are delegable, see
[Session control](#session-control). Using `ToolFlag` rather than a `delegate=` keyword avoids
touching all four `function_tool` overloads and puts the switch where `CANCELLABLE` and
`IGNORE_ON_ENTER` already live (`llm/tool_context.py:152`).

`AgentActivity.tools` (`voice/agent_activity.py:426`) currently serves two consumers that now need
different answers:

| accessor | contents | consumers |
|---|---|---|
| `tools` | everything the executor can run, including `delegate` | `perform_tool_executions` (`:3203`, `:3856`) |
| `model_tools` | `delegate` + tools flagged `NO_DELEGATE` | `perform_llm_inference` (`:2993`), `update_tools` (`:495`), initial realtime config (`:949`, `:983`) |

`ToolContext._exclude` already backs this pattern for `_on_enter_ignored_tools` (`:2936`), so
`model_tools` reuses it. Keeping the full set on the executor side is also what lets a model
synthesize a `delegate` call it was never offered — the reason the duplex model needs no API change.

### Management tools are scoped by tool visibility

`has_cancellable_tool` auto-exposes `lk_agents_cancel_task` / `lk_agents_get_running_tasks` on the
model's list when any tool is `CANCELLABLE` (`:435`). With delegation both brains can hold
cancellable tools — the slow brain's delegated ones, and any `NO_DELEGATE` async tool left on the
fast brain — so the rule is evaluated **per list**: each brain gets the management tools iff its own
list has a cancellable tool.

That is not enough on its own. Both management tools read `_RunningTasks.get(ctx.session, {})`
(`voice/tool_executor.py:172`, `:180`), a session-wide registry that `execute()` populates regardless
of which executor ran the task. Unscoped, the fast brain's `get_running_tasks` would list the slow
brain's running tool *names* — the same leak that keeps level-2 items out of `agent.chat_ctx` — and
its `cancel_task` could cancel delegated work.

So a running task is visible to whichever brain can call that tool. The fast brain sees its
`NO_DELEGATE` cancellables, the slow brain sees the delegated ones. With no delegation configured
every tool is in the one list, so the current session-wide behavior is unchanged. `delegate` is not
`CANCELLABLE`, so it never appears in either view — the fast brain cannot cancel a delegation, which
is the intent.

`_inject_running_tool_calls` (`voice/agent_activity.py:2984`) reads the same registry and takes the
same filter, for the same reason: it would otherwise place a delegated tool's name and arguments
directly into the fast brain's context. Its placeholder for the running `delegate` call is the one
entry that must survive, since that is what stops the fast brain re-delegating.

An `AsyncToolset` keeps its own session-scoped executor (`llm/async_toolset.py:76`), so its delegated
members run there rather than on the delegation executor. `_check_duplicate` is per-executor, so a
delegated `AsyncToolset` tool competes with its own siblings, not with the rest of the slow brain's
work — which is the existing meaning of that grouping and needs no change.

## Managing work

Inside a delegation, tools run **to completion**. The async early-return exists to avoid blocking
speech, and the slow brain does not speak; releasing its loop on "still running" would make it
answer from incomplete facts. A level-2 `ctx.update()` is instead captured and re-emitted as a
`progress` delivery — translated, dropping the function name and `call_id`, which are meaningless to
the fast brain. It reads as "the expert reports X", never "the tool `search_flights` updated".

**Delegated tools get a real `RunContext`, not a detached stub.** Its `foreground()` sequences after
the delegation's answer, and its `speech_handle` resolves to the turn that speaks that answer —
created up front, completed when that speech finishes. A tool written against the `RunContext`
contract then behaves correctly in either brain, instead of every tool needing an audit.

The delegation executor is **session-scoped and shared across delegations**. Per-delegation
executors make `_check_duplicate` blind to sibling work, and the case below becomes impossible.

### The scenario this is for

The user asks to book a flight, then a moment later asks to change the date. Two delegations, the
first still in flight. It needs no delegation-specific configuration — only the per-tool knobs the
user already writes:

```python
@function_tool(flags=ToolFlag.CANCELLABLE, on_duplicate="replace")
async def book_flight(ctx: RunContext, origin: str, destination: str, date: str) -> dict:
    ...
```

The second delegation's slow brain reasons "this replaces the pending booking" and simply calls
`book_flight` again. `_check_duplicate` (`voice/tool_executor.py:617-638`) cancels the in-flight call
and runs the new one — and refuses with a clear `ToolError` if it was not cancellable, which is a
correct outcome the slow brain can explain. The first delegation's loop sees its tool came back
cancelled and answers "the original booking was replaced"; the coalescer merges that with the second
answer into one turn.

The slow brain acts on `book_flight`, never on "delegation #1". Exposing delegations to it would
leak the envelope into its reasoning and create cross-request bookkeeping for no gain.

### Cancellation is a result

A delegation that is cancelled or aborted still produces an `answer` delivery describing what
happened, including side effects that already landed: "cancelled — the refund had already been
issued." That is what the fast brain needs to say, and it reuses the answer channel instead of
inventing a cancellation path.

That is also why the duplex model needs no cancel event it does not have: the same account goes out
as `delegation.context.append` and the model deals with it in prose. What differs between transports
is the **trigger** — the slow brain's own reasoning, the application, or the timeout — not the
effect.

## Session control

`end_call`, handoff and transfer are delegable, and they should be: whether to end a call is a
judgment that benefits from the big model.

`ctx.foreground()` (`voice/events.py:148`) drains this tool's pending deferred reply, *then* takes
the floor. So the goodbye is already spoken by the time the body runs, and the sequencing that would
otherwise be missing comes for free — `end_call` reduces to `session.shutdown()`, handoff to
`session.update_agent()`. With the real `RunContext` above, `end_call`'s existing
`ctx.speech_handle.add_done_callback` (`beta/tools/end_call.py:87`) also resolves against the turn
that speaks the answer, so it works unchanged.

`tool_choice="none"` on the answer reply stays as it is. Session control does not route through the
fast brain, so nothing needs to be relaxed.

## Errors

- **The slow model errors** — surfaces as the delegate tool's output through the normal `ToolError`
  path, and the fast brain tells the user it could not look that up.
- **`max_tool_steps` exhausted** — the node raises `ToolError` rather than returning nothing, so the
  fast brain has something to say instead of going silent.
- **No answer at all** — `delegation.timeout` (60 s) raises the same `ToolError`.

Reported at session start:

- a `delegation_llm` is set but a user tool is named `delegate` — error, the collision is
  unresolvable
- a `delegation_llm` is set and every tool is flagged `NO_DELEGATE` — warning, the slow brain has no
  tools (still valid as a reasoning-only escalation)
- `delegation` options are set with no `delegation_llm` — warning, they are never used

## Framework changes

| change | where |
|---|---|
| `ctx.update(..., silent=True)` | `voice/events.py`, `voice/tool_executor.py` |
| `ToolFlag.NO_DELEGATE` | `llm/tool_context.py` |
| management tools scoped by tool visibility, evaluated per brain | `voice/tool_executor.py`, `voice/agent_activity.py` |
| `DELEGATE_TOOL_NAME` + the builtin tool | `llm/`, new module |
| `AgentActivity.model_tools` split; management tools follow the cancellable tools | `voice/agent_activity.py` |
| `delegation_llm` / `delegation` slots and resolution | `voice/agent_session.py`, `voice/agent.py` |
| `delegation_node` + default implementation | `voice/agent.py` |
| session-scoped delegation executor; run-to-completion; `ctx.update()` translated to `progress` | `voice/agent_activity.py`, `voice/tool_executor.py` |
| real `RunContext` for delegated tools, with a lazily-resolved answer speech handle | `voice/agent_activity.py`, `voice/events.py` |
| `_deliver_reply` skips `generate_reply` under `auto_tool_reply_generation` | `voice/tool_executor.py` |

## Shaped for the duplex model

A full-duplex model like GPT-Live emits delegations natively: the server sends `delegation.created`
and expects the answer back as `delegation.context.append`, and it keeps talking to the user while
the client works. Its plugin lands with the duplex abstraction, not here — but this design must leave
it nothing to renegotiate. Four properties do that:

1. **The tool name is a constant**, so the plugin can synthesize
   `FunctionCall(name=DELEGATE_TOOL_NAME, arguments={"task": ev.text}, call_id=ev.delegation_id)`.
2. **The executor's tool set is independent of the model's**, so `delegate` resolves even though a
   client-delegating session advertises no tools at all.
3. **The tool's return value is opaque to the framework**, so the plugin picks the wire form —
   `delegation.context.append` there, an output item elsewhere. `progress` maps to the same append,
   which is what lets the model cover a wait.
4. **`auto_tool_reply_generation` already gates the reply**, so a model that continues on its own is
   not asked to.

The delegation then records identically for all three families, and only the sync to the provider
differs — which `mutable_chat_context` (`llm/realtime.py:73`) already explains:

| | `delegate` pair in `agent.chat_ctx` | pushed to the provider |
|---|---|---|
| pipeline | yes | yes — the context is the request |
| realtime | yes | yes — `mutable_chat_context=True` |
| duplex (later) | yes | no — the server already holds the delegation and the appended answer |

What the duplex work still has to do, deliberately not done here: drop the idle wait before a push,
add a capability for a model that owns its own output timeline, confirm the alpha acts on
intermediate appends rather than only a final one, and check that a long-running tool execution does
not interfere with segmenting turns on `turn.done`. All audio and turn-taking concerns. No line of
this design changes.

## Known limitations

- **A realtime fast brain may not announce at level 1.** Not every model emits text alongside a tool
  call, and realtime models are the worst at it. The gap is then covered only by level 2, one LLM
  round later. That is a property of the chosen fast model, not something the framework can fix.
- **If neither level announces, the delegation is silent until it answers.** A slow model that
  thinks for seconds before its first tool call produces dead air. It is a prompt problem — the
  `announce` directive and `tool_description` are the levers — and no knob is added for it.
- **A superseded delegation still answers.** Both the replaced and the replacing delegation produce
  an answer; they coalesce into one turn, but the first one's account is partly redundant.
- **A recorded transcript won't show the slow brain's tool calls.** They are events and trace spans
  only, so debugging "why did it say Tuesday" needs traces rather than `session.history`.

## Testing

Unit (`tests/test_delegation.py`), against `FakeLLM` and the existing `FakeRealtimeModel`:

- **Routing** — the model's list is `delegate` plus `NO_DELEGATE` tools; the executor still resolves
  the withheld ones; a synthesized `delegate` call runs even when never advertised; an unannotated
  toolset's members all delegate.
- **Management scoping** — each brain gets the management tools only when its own list has a
  cancellable tool; the fast brain's `get_running_tasks` does not list delegated tools and its
  `cancel_task` refuses their call ids; a `NO_DELEGATE` async tool on the fast brain is still
  cancellable by it.
- **Non-blocking** — the silent dispatch produces an output item and no reply; the delegation appears
  in the running-task placeholder on the next turn.
- **Announcing** — level-1 text alongside `delegate` is spoken in the same turn; the slow brain's
  text alongside a tool call arrives as `progress`; a `progress` pending alongside the `answer` is
  dropped.
- **Reply gate** — `auto_tool_reply_generation=True` pushes the update without `generate_reply`;
  `False` still calls it.
- **Context** — what the node receives; a prior delegation's pair visible to the next; level-2 items
  absent from `agent.chat_ctx` and present in the emitted events.
- **Managing work** — the flight scenario end to end: `on_duplicate="replace"` cancels the in-flight
  booking, the non-cancellable case raises, and the superseded delegation answers.
- **Session control** — `end_call` from a delegated tool shuts down after the answer is spoken;
  handoff retargets; a delegation in flight across a handoff still delivers.
- **Errors** — slow-model error, `max_tool_steps`, timeout, and the three start-time reports.
- **Node overrides** — returning, yielding, reporting progress, and answering without an LLM.

Manual: a console pipeline agent and a console realtime agent, each with a slow cancellable tool,
checking the level-1 line lands immediately and the answer follows without a double reply.
