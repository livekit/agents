# Multi-agent

| Pattern | How | History |
|---|---|---|
| delegation inside a tool | `(await billing.run(question, usage=ctx.usage)).output` | the expert runs on a fresh history; the caller's history gets the tool call and its result |
| handoff | a tool returns an `Agent` | the same history, continued by the new agent; saved on the lane's agent stack |
| a delegate | `Agent(tools=[A2ADelegate(url)])`, or `delegate.run(...)` from a tool | see "Delegates" |
| remote expert | `A2ADelegate(url)` | its own lane of the caller's thread when its card offers the Live extension; a context of its own otherwise |

## Delegation

```python
@main_line.tool
async def quick_balance(ctx: RunContext[Deps, Caller], account_number: str) -> str:
    result = await billing.run(f"Balance for account {account_number}.", usage=ctx.usage)
    return result.output
```

The main agent phrases the answer itself; the result is a tool result in
the history and never runs twice. The
expert answers once, inside the turn, and never talks to the person. An
agent that must talk to the person across turns is a handoff, below.

## Handoff

```python
@main_line.tool
async def route(ctx: RunContext[Deps, Caller], department: Department) -> Agent:
    ctx.state.department = department
    return DEPARTMENTS[department]
```

The returned agent takes the conversation on the same history, from the
next step of the turn; the history records an `agent_handoff` item. The
framework routes: the handoff is saved on the lane's agent stack
([lanes.md](lanes.md)), and the next message runs the agent on top of it, on
a call and in text alike. In text, `root.run(message)` runs whichever agent
is on top of the stack; the root runs while the stack is empty, and the
handler stays one line:

```python
@server.a2a_session(endpoint="main-line", description="Mercy General main line.", state=Caller)
async def text(ctx: A2ASessionContext[Caller]) -> str:
    result = await main_line.run(ctx.message, message_history=ctx.message_history, state=ctx.state)
    return result.output
```

## Delegates

A delegate answers one request inside the caller's turn and the caller
phrases the result. There is one, `A2ADelegate(url)`: an endpoint of this
project, of another deployment, or any A2A server, reached the same way. An
expert in the same process is reached through its own door too; nothing runs
an agent inside the caller's turn. The interface behind it,
`livekit.agents.delegation.Delegate`, is for anything else that answers a
request.

A delegate is a toolset. Attached to an agent, `Agent(tools=[A2ADelegate(url)])`,
it is one tool, the same in a text run and on a call ([voice.md](voice.md)):

```python
back_office = A2ADelegate("http://127.0.0.1:7880/agents/dentist/default/backend")
front_desk = Agent(instructions=FRONT_DESK, llm="google/gemma-4-31b-it", tools=[back_office])
voice = Agent(instructions=ON_THE_PHONE, tools=[back_office])
```

The agent knows what it can ask from the expert's own card, read once in
the toolset's `setup()`: the toolset's id is the card's name, the tool is
`ask`, or `name=` when one agent has two experts, and its description is
the card's description followed by one fixed sentence that says to state
the request in full. The tool hands the request on with the
run's thread and history, passes the expert's progress to `ctx.update()`
and its exact text to `ctx.say()`, returns the answer, or the question the
expert has for the person, and records the expert's task on the call. An
answer the expert flagged `livekit/verbatim` reaches the person as written,
part by part, with no model step; a task that ends with no text is an
empty answer. On a call its first
update releases the turn with a line the model acknowledges while the
expert works; `announce=False` turns that off.

A LiveKit expert gets its own lane in the caller's thread, named after the
expert agent, and never contends for the caller's lease.

```python
result = await delegate.run(
    prompt, progress=ctx.update, say=ctx.say, context_id=None, message_history=None, usage=None
)
```

`run` returns a `DelegationResult`, a finished `RunResult`: `output`, and
the expert's task: `state` (`completed`, `input-required`, `canceled`),
`task_id`, `context_id`, and, over A2A, `endpoint`, `transport` and
`profile` (`livekit` when the extension was in force, else `a2a`). A
`failed` task raises `DelegationFailed` with what the expert said.
`progress` takes each status message the expert sends while it works, and
`say` the ones flagged `livekit/verbatim`, its exact text; without `say`
they go to `progress`. `verbatim` is the answer or the question when the
expert flagged it.
`aclose()` releases what the delegate holds.

## A remote expert

```python
from livekit.agents.delegation import A2ADelegate, record_delegation

ELIGIBILITY = "https://northside.livekit.cloud/agents/eligibility/production/eligibility"

@front_desk.tool
async def coverage(ctx: RunContext[Deps, Patient], payer: str, member_id: str, visit_type: str) -> str:
    expert = A2ADelegate(ELIGIBILITY)
    try:
        result = await expert.run(f"{payer} member {member_id}, {visit_type} visit", progress=ctx.update)
    finally:
        await expert.aclose()
    record_delegation(ctx.function_call, result)
    return result.output
```

Another deployment, with its own worker and its own owner. When its card
offers the Live extension, the delegate sends the caller's thread id as
the `contextId`, and the expert keeps its own lane in the caller's thread:
its own history and state, beside the caller's. Inside `agent.run()` the
delegate finds the run's thread by itself; `context_id=` names one
explicitly. Any other A2A server gets the request as text and nothing else,
and keeps a context of its own. That context is recorded on the delegating
call as `lk.delegation.context_id`, next to the task id, and the next
delegation to the same endpoint reads it from the latest such call in its
history: one delegate serves every conversation in the process and keeps
none of them. The caller depends on nothing coming back
but core A2A: the working statuses are the progress, the artifact is the
answer. See [a2a.md](a2a.md), "Writing an expert in another framework".

`A2ADelegate(url, token=None, httpx_client=None, name=None, announce=True, conn_options=DEFAULT_API_CONNECT_OPTIONS)`.
An answer stream that drops is read again with `tasks/resubscribe` up to
`conn_options.max_retry` times, `retry_interval` apart. With no `token`, an
expert on the project's own host (the host of `LIVEKIT_URL`) gets a project
token minted from `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`, which passes a
`public=False` endpoint; an expert anywhere else gets none.

An expert that needs the person's answer ends its task `input-required`:
the question is the delegation's output. The next delegation in the thread, carrying the person's answer, is
the answer to that question on the expert's lane; the waiting task ends
`completed`, merged into the new one. The delegate's tool returns the
question as `InputRequired(question, tool=)` naming itself, so the front
desk asks it in its own words, or as written when the expert flagged its
text part `livekit/verbatim`, and what the person says next is sent to the expert as the
tool's next request, as they said it, with no model turn in between: on a
call the framework runs the tool on the turn's transcript, in text on the
next message, on whichever worker gets it ([tools.md](tools.md)). A
delegation canceled on the caller's side, as when the person speaks over a
call, sends `tasks/cancel` to the expert with `livekit/reason:
caller-canceled`.

`record_delegation(function_call, result)` writes the expert's task on the
call that delegated, in `FunctionCall.extra`: `lk.delegation.task_id`,
`lk.delegation.context_id`, `lk.delegation.endpoint`,
`lk.delegation.transport` and `lk.delegation.profile`, so a dashboard joins
the two lanes and the next delegation finds a stock expert's context. The
delegate's own tool records it by itself; a tool of your own that calls
`delegate.run` calls it and passes the history as `message_history`.

## Endpoints versus deployments

An endpoint is a door; a deployment is the boundary. Two endpoints in one
deployment share the worker and the history for a thread and differ in who
may call them. A second deployment buys its own worker, its own history,
and its own owner, deploy and version, at the price of a network hop and
state that does not cross. The test is the one for a web service: would
this thing get a URL of its own, and run on its own machines? A different
caller: an endpoint. A different worker or owner: a deployment. To organize
code: neither; use a `Toolset` or a handoff.

See `examples/use-cases/03_insurance_eligibility.py` and
`07_hospital_main_line.py`.
