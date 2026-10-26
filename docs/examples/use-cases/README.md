# Fourteen agents, on every channel

Fourteen complete files, one idea each, over the use cases LiveKit sees most on the phone:
conversational IVR replacement, appointments, verification, order status, intake, billing;
outbound reminders, speed-to-lead and patient outreach. Healthcare, financial services,
insurance, retail and sales, plus the summarizer that runs inside agent-service. The API they use
is documented one feature at a time in `../../README.md`.

## Built

These use the approved design; a code gap, where one remains, is in `../../TODO.md`.

| File | Vertical | The idea | New |
|---|---|---|---|
| `01_appointments_by_phone_and_text.py` | healthcare | book, confirm or move an appointment, by phone and by text | `Agent.run(message, message_history=, state=, deps=)`, `state_type=` and `deps_type=`, `@agent.instructions`, the handler returns the answer |
| `03_insurance_eligibility.py` | healthcare | one brain behind the phone and the text line; a slow expert on its own deployment | a thin voice agent whose one tool is `A2ADelegate(url)` in `tools=`; `A2ADelegate(url).run(..., progress=ctx.update)` from a tool, `record_delegation`; `public=False` |
| `04_a_claim_with_a_photo.py` | insurance | a first notice of loss needs a photo, maybe tonight | a tool returns `InputRequired(question)`: the turn waits, and the next message is the answer the model reads; `output_type=` |
| `07_hospital_main_line.py` | healthcare | billing, scheduling and nurse triage behind one number | delegation in a tool, handoff saved on the lane's agent stack, the voice door delegating to the same brain |
| `10_mortgage_prequalification.py` | financial | a runtime list of questions answered over days | `AgentTask` awaited from a tool in text: a handoff plus a deferred call; answers kept in the state |
| `15_renewal_email_graph.py` | insurance | a renewal email drafted and reviewed until compliant | a graph: `BaseNode`, `End`, `Graph`, edges as return annotations, `mermaid_code()`; stateless, inside one turn |

## Planned

These show design that is not built yet: background tasks, channels, the thread handle,
toolset composition. Each `planned-` file relies on at least one of them.

| File | Vertical | The idea | New |
|---|---|---|---|
| `planned-05_back_in_stock.py` | retail | order status, and a watch for an out-of-stock item | an `AgentTask` started in the background with `delay=` and `timeout=`: a pure code task returning `Completed`, and one that asks with `InputRequired(question, tool=self.decide)`; `on_timeout` |
| `planned-06_card_servicing_toolsets.py` | financial | cards, payments, rewards: composed, filtered, gated | `Toolset`, `FilteredToolset`, `PrefixedToolset`, `requires_approval`, `InputRequired` from a tool, MCP |
| `planned-08_summarizing_the_simulations.py` | internal | the summarizer that scores these agents' simulations | sixty submitted tasks, a join with `after=[handles]` and `ctx.task(id).result()`, `Retry`, `concurrency`, `@agent.output_validator` |
| `planned-09_speed_to_lead.py` | sales | the CRM calling a new lead in minutes, a rep resuming | `lk.threads.*`, a dispatch carrying the thread id, a `message/send` starting a task with no model, a rep resuming with `AgentClient.send(thread_id=)` |
| `planned-11_post_procedure_check_in.py` | healthcare | check on a patient on day 1, 3 and 7, escalate at once | an `AgentTask` started for a date whose `on_enter` returns `InputRequired(question, tool=self.report, timeout=)`, and that starts the next one |
| `planned-12_many_chats.py` | any | one assistant, many people, several threads each, with a sidebar | a thread per chat with the user as `scope`, `lk.threads.create(scope=)`, `lk.threads.list(scope=)`, `thread.update(attributes=)` |
| `planned-13_prescription_refill.py` | healthcare | a refill waits on the prescriber's system, then maybe on the patient | `Working(tool=self.x, timeout=)` waiting on a callback; a `DataPart` calling the tool with no model; `Completed \| InputRequired` from a tool |
| `planned-14_returns_by_email.py` | retail | a returns desk over email, attachments in, a warehouse that inspects days later | email as a channel: `email:` key, attachments as items, the reply in the thread, a task's question answered by `In-Reply-To`; a link where an attachment out is wanted |

## Where things live

- **The thread** is the person's conversation across weeks and channels. Every thread has its
  own database on agent-db, created with the thread. `AgentServer(storage=AgentDBStorage())` is
  the one line that names it; the tests run against a local agent-db. A call's thread is
  `ctx.thread`, the dispatch's `thread_id` or a new thread, given as `LiveSession(thread=)`.
- **That database holds the history, the journal, the state and the tasks.** The framework loads
  them before a run and commits them after it. Nothing in these files opens a database
  connection.
- **`state` is saved, `deps` are not.** The typed object declared with `state_type=` and opened
  with `a2a_session(state=X)` or `LiveSession(state=X())` is written by tools through `ctx.state`
  and saved with the thread after every run. `deps`, declared with `deps_type=` and passed to
  `run(deps=)`, are what the tools need for one run: HTTP clients, a read budget. Whoever
  starts the run builds them and passes them in, one shape everywhere:
  `deps = Deps(clinic=httpx.AsyncClient(base_url=CLINIC))`, then `LiveSession(deps=deps)`,
  `run(deps=deps)`, or `super().__init__(deps=deps)` in an `AgentTask`'s `__post_init__`
  (05, 08, 11, 13, 14). Tools and hooks read `ctx.deps`. They never reach storage.
- **A business's own systems** (a clinic's scheduling, a bank's accounts, a store's orders, a CRM)
  are reached only where something is genuinely theirs, as one `httpx` call where it happens.
- **Nothing lives in the process.** Every file is complete on its own.

## Two brains, two APIs

- A call is a `LiveSession`: the fast brain. It either runs a small agent itself (01) or stays
  thin and hands every request to the same brain the text door runs, `tools=[A2ADelegate(url)]`
  (03, 07).
- A text turn is `Agent.run(message, message_history=, state=, deps=)`: pure, it appends to the
  history you pass and returns a typed result. The handler returns the answer; the framework
  commits the history, the state and the journal and delivers it.

## Conventions

- An `Agent` is configuration: built once at module level, stateless, and it holds no history.
  The thread's history in voice, `run(message_history=)` in text. No `message_history` means a
  fresh conversation.
- The names: `Agent[Deps, State, Output]` with `deps_type=`, `state_type=` and
  `output_type=`, `RunContext[Deps, State]`, `run(prompt, deps=, state=, message_history=)`,
  `result.output`, `@agent.output_validator` raising `ModelRetry` (08). The model is `llm=`,
  because an `Agent` also carries `stt=` and `tts=` for a call. `RunContext[None, Patient]` says an agent has no deps.
- Agents are `Agent(...)` instances named after their variable. Tools are `@agent.tool`
  functions, or `@toolset.tool` on a `Toolset`; they read per-run data from `ctx.deps` and
  `ctx.state`, never `self`. `@agent.instructions` puts state in front of the model (01, 12).
- The classes a developer writes are `AgentTask`s (05, 08, 10, 11, 13, 14): a dataclass whose
  fields are the task's arguments, JSON-shaped, and whose tools return `Completed`. What a task
  learns goes into `self.state` (typed by `AgentTask[Output, TaskState]`), `ctx.state`, its own
  history, or a child task's arguments, never into other attributes of `self`.
- Idempotency keys: `ctx.function_call.call_id` in a tool, `ctx.task_id` in a task.
- A graph (15) is `livekit.agents.graph`. It runs
  inside a turn or a task, stores nothing, and is for code steps with known edges. It never
  waits; a workflow with waits is a task whose code between waits is graphs.
- A delegate is a toolset in `tools=`, by text and on a call. Agents that serve text carry `llm=`; a call brings its
  model on the thread.

## Everything resumes

A run, a tool call without its output, a task with its own stream. Nothing waits in memory and
nothing is replayed: long work is task rows in the thread, run one at a time, and a wait is a
due time the platform keeps. Two messages in a
burst, on two workers, give one reply: by SMS the inbox batches them after a debounce, in a web
chat the second interrupts the first (`../../text-endpoints.md`). None of the files say anything
about it; `inbox` and `debounce` on `a2a_session` are the only knobs, and
`concurrent` is there for a stateless door. New code reads old rows, which is the point. Not shown, because you never write it: thread ids, routing, persistence,
the A2A wire, the agent card, streams, expiry, timers.

## The whole vocabulary

- In a tool: `ctx.state`, `ctx.deps`, `ctx.update()`, `return InputRequired(question)`,
  and the thread (`update(attributes=)`, `lane(agent).history()`).
- A background task is an A2A task started without waiting for its reply, a delegate's
  `send()` from a tool, a handler or a task, or a `message/send` carrying `livekit/task` from
  outside (planned, [planned-agent-tasks.md](../../planned-agent-tasks.md)). Its constructor
  arguments are JSON in its row; its progress is its own A2A history.
- In an `AgentTask`: `on_enter(ctx)`, `@function_tool` methods, `on_timeout(ctx)`, all taking a `RunContext`; `self.state`; the return
  value is the transition: `Completed(value)`, `Failed(error)`, `InputRequired(question, tool=,
  timeout=)`, `Working(tool=, timeout=)`, or a plain value for a tool result; a `send()` for
  what runs next; `ctx.update()`; `ctx.state`; `ctx.task_id`.
- In a handler: `agent.run()`, its return, `ctx.update()`.

## Human in the loop

The return value of a tool or a hook is the A2A state it moves its task to. From a conversation
tool, `InputRequired(question)`: the turn waits and the next message is the answer, which the
model reads (04, 06). From an `AgentTask`, out of `on_enter` or one of its tools, `InputRequired(question,
tool=self.x, timeout=)`: the question is delivered word for word, the task waits
`input-required`, and `tool=` names the task's tool that takes the answer; a form or a system
posts its arguments by task id and the tool is called directly, free text goes to the tool's
string parameter, or to a turn of the task's model when it has none (05, 11). `Working(tool=self.x, timeout=)` waits on a system
that answers by task id (13, 14).
`Completed(value)` ends the task and `Failed(error)` fails it; a plain return value is a tool
result the model continues from. A reply without a task id is a turn of the main agent, which
sees the task waiting and hands the conversation to it. A guided conversation inside the turn is
the same `AgentTask` awaited (10).

## One class, three uses

- **awaited on a call:** the task takes the conversation and hands it back on `Completed`.
- **awaited in text:** a handoff plus a deferred call; each reply is a turn of the task.
- **in the background:** a `send()` not awaited, with `delay=`, runs it as its own A2A task
  beside the conversation, with its own id, stream and history. The default `Tasks()` capability gives the
  submitting agent a snapshot of its tasks in its instructions and the tools `task_status`,
  `cancel_task` and the handoff to a waiting one; it is built from `@agent.instructions` and a
  `Toolset` over `ctx.tasks`, and any part of it can be replaced or the whole removed with
  `capabilities=[]`. A pure code task has `on_enter` and no model. `Completed(value)` ends it; the
  value is `handle.result()` and, when text, its last message to the person. `on_timeout(ctx)`
  runs at the deadline. `after=[handles]` and `ctx.task(id).result()` make a join.

## Three things called task, and a graph

- An **A2A task** is one run, the unit on the wire: a turn and a submitted `AgentTask` alike, in
  the thread's context. A stock client can `tasks/get`, `tasks/list`, `tasks/cancel` and
  `tasks/resubscribe` any of them.
- An **`AgentTask`** is the class: an agent with a job that completes with a value. Awaited, it
  is a sub-flow inside the conversation; submitted, it is a background task.
- The word **job** is cloud's: a worker process assigned to a room for a call.
- A **graph** is nodes, edges and `End`, run inside a turn or a
  task (15), never across a wait.
