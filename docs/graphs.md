# Graphs

For work whose steps are code and whose edges are known before it runs:
agents called as functions, branches and loops in code, a typed end.
In `livekit.agents.graph` a node is a dataclass whose fields are its inputs, `run()` returns
the next node or `End`, and the return annotation is the edge.

A graph is a class and an [agent task](agent-tasks.md). Run inline, it goes
from its start node to `End` in the turn. Awaited from a tool, it can stop
at a node, ask the person something, and continue on their reply, in a
later turn and on any worker.

| | |
|---|---|
| `Renewal.run(start, state=)` | the inline form: to `End` in the turn, never waits |
| `await Triage(start, state=)` | the waiting form: a task on the lane's agent stack until `End` |

## Nodes and edges

```python
from livekit.agents.graph import BaseNode, End, Graph, GraphRunContext

@dataclass
class Draft:                                          # the graph's state, for this run
    policy: Policy
    drafts: list[Email] = field(default_factory=list)

@dataclass
class WriteEmail(BaseNode[Draft]):
    feedback: str | None = None

    async def run(self, ctx: GraphRunContext[Draft]) -> ReviewEmail:          # the annotation is the edge
        result = await writer.run(prompt_for(ctx.state, self.feedback))
        ctx.state.drafts.append(result.output)
        return ReviewEmail(result.output)

@dataclass
class ReviewEmail(BaseNode[Draft, None, Email]):
    email: Email

    async def run(self, ctx: GraphRunContext[Draft]) -> WriteEmail | End[Email]:
        review = (await reviewer.run(self.email.model_dump_json())).output
        if review.approved or len(ctx.state.drafts) >= 3:
            return End(self.email)
        return WriteEmail(feedback=review.feedback)

class Renewal(Graph[Email, Draft]):
    nodes = [WriteEmail, ReviewEmail]
```

| | |
|---|---|
| `BaseNode[State, Deps, Output]` | a dataclass whose fields are the node's inputs; `run()` returns the next node, `End`, or `InputRequired` |
| `End(value)` | the typed result |
| `InputRequired(question)` | the node waits on the person; on the reply it runs again |
| `Graph[Output, State]` | a class with `nodes = [...]`; edges come from the return annotations; `Graph[Output, State, Deps]` for the inline form's deps |
| `GraphRunContext` | `state` and `deps` for the run, and `reply`, the person's answer to the node's question |
| `Annotated[Next, Edge(label="again")]` | a label on the edge, for the diagram |

The edges are checked when the class is defined: a `run()` without a return
annotation, one that returns something other than nodes, `End` and
`InputRequired`, or an edge to a node not in `nodes` raises
`GraphSetupError`. While it runs, a node that returns a node outside the
graph raises `GraphRuntimeError`.

## Where to call it

From wherever the decision is made: a tool of the conversation agent, when
the model decides; a handler, when a backend decides; a task hook, when the
work was scheduled or handed off. If it finishes in seconds and needs
nothing from the person, run it inline and return its result from the tool,
as the text the model reads, or from the handler as the reply. If a node
needs the person, await it from a tool. If it takes longer, submit a task
and run the graph in its `on_enter`: a graph holds the conversation while
it runs, and a task does not.

## Running inline

```python
result = await Renewal.run(WriteEmail(), state=Draft(policy))
result.output                                         # Email
result.state                                          # the Draft as the run left it

async with Renewal.iter(WriteEmail(), state=Draft(policy)) as run:    # node by node
    async for node in run: ...                        # the start node, each next node, the End
    run.result                                        # the GraphRunResult once at End
```

`run.next()` runs one node by hand instead of iterating. An inline graph
executes where the enclosing run executes, node after node in one process,
until `End`; a node that returns `InputRequired` raises `GraphRuntimeError`.
Nothing is stored between nodes. Its state is the object passed to
`run(state=)`, not the thread's; a node that must remember something for the
conversation writes it into the enclosing `ctx.state` when the graph ends.
`deps=` is whatever the nodes need for the run, a progress callback
included: a node reaches the conversation only through what it is given.
Agents a node calls with no `message_history` run a fresh conversation whose
messages are not the thread's history.

## Waiting on the person

```python
@dataclass
class LookUpBooking(BaseNode[Ticket, None, TriageResult]):
    async def run(self, ctx: GraphRunContext[Ticket]) -> EU261 | Baggage | InputRequired:
        if ctx.reply:                                      # the answer to this node's question
            ctx.state.booking_ref = parse_ref(ctx.reply)
        if not ctx.state.booking_ref:
            return InputRequired("What is your booking reference?")
        ...

class Triage(Graph[TriageResult, Ticket]):
    nodes = [Classify, LookUpBooking, EU261, Baggage, FareRules, DraftReply, Review, Pay]

support = Agent(instructions="Skyline Air customer care.", llm=LLM)

@support.tool
async def triage(ctx: RunContext, email: Email) -> TriageResult:
    return await Triage(Classify(email), state=Ticket())

@server.a2a_session(endpoint="triage", description="Skyline Air customer care.")
async def text(ctx: A2ASessionContext[None]) -> RunResult[str]:
    return await support.run(ctx.message, message_history=ctx.message_history)
```

- A node waits by returning `InputRequired(question)`, the same transition a
  tool or a task returns. The turn ends input-required with the question as
  the reply.
- The reply runs the node that asked again, with the answer on `ctx.reply`.
  There is no separate node per question.
- In text the graph sits on the lane's agent stack like any awaited task.
  The handler is called for the reply as usual, and `support.run(...)` runs
  the top of the stack: the waiting graph, which takes the message before
  any model. When the graph reaches `End`, its value is the output of the
  tool call that awaited it, and the agent below finishes its turn.
- On a call the graph holds the conversation in memory; the person's next
  turn runs the asking node again.
- A waiting graph has no deps: it is rebuilt on any worker from what was
  saved, so its nodes read the world from their module.
- A wait's `timeout=` is a task's, planned in
  [planned-agent-tasks.md](planned-agent-tasks.md).

### What is saved

While a graph waits, its stack entry holds the graph's class name, the node
that asked (`{"node": <class name>, "fields": <fields as JSON>}` as the
entry's arguments) and the state as JSON, like a task's class, arguments and
state. Nothing is saved between nodes that do not wait.

### A deploy changed the graph

Edges and node bodies may change while a graph waits: the next node is
whatever the rebuilt node's `run()` returns. A saved node that no longer
rebuilds, renamed, removed, or with changed fields, or a state that no
longer validates, fails the graph with `stale-graph` (`STALE_GRAPH`), and the
agent below reads it as a `ToolError` on the call that awaited it. A graph
never restarts silently, since its nodes have side effects. A graph class
that no longer exists drops from the stack like any task that cannot be
rebuilt.

## Drawing

```python
print(Renewal.mermaid_code(start_node=WriteEmail))
```

```
stateDiagram-v2
  [*] --> WriteEmail
  WriteEmail --> ReviewEmail
  ReviewEmail --> WriteEmail
  ReviewEmail --> [*]
```

A node that can return `InputRequired` loops on itself, labelled `asks`.

## What a graph is not

Not the conversation, whose next step the model picks from what the person
said. A graph is an agent task whose steps are nodes, chosen by code.

See [`examples/airline-support/`](../examples/airline-support/README.md) and
`examples/use-cases/15_renewal_email_graph.py`.
