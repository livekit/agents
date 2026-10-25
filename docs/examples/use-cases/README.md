# Fourteen agents, on every channel

Fourteen complete files, one idea each, over the use cases LiveKit sees most on the phone:
conversational IVR replacement, appointments, verification, order status, intake, billing;
outbound reminders, speed-to-lead and patient outreach. Healthcare, financial services,
insurance, retail and sales, plus the summarizer that runs inside agent-service. The API they use
is documented one feature at a time in `../../README.md`.

## Conversation examples

These examples cover tools, conversations, and structured results.

| File | Vertical | The idea | New |
|---|---|---|---|
| `01_appointments_by_phone_and_text.py` | healthcare | book, confirm or move an appointment, by phone and by text | `Agent.run(message, message_history=, state=, deps=)`, `state_type=` and `deps_type=`, `@agent.instructions`, the handler returns the answer |
| `03_insurance_eligibility.py` | healthcare | one brain behind the phone and the text line; a slow expert on its own deployment | a thin voice agent whose one tool is `A2ADelegate(url)` in `tools=`; `A2ADelegate(url).run(..., progress=ctx.update)` from a tool, `record_delegation`; `public=False` |
| `04_a_claim_with_a_photo.py` | insurance | a first notice of loss needs a photo, maybe tonight | a tool returns `InputRequired(question)`: the turn waits, the next message becomes the call's output; `output_type=` |
| `07_hospital_main_line.py` | healthcare | billing, scheduling and nurse triage behind one number | delegation in a tool, handoff saved on the lane's agent stack, the voice door delegating to the same brain |
| `10_mortgage_prequalification.py` | financial | a runtime list of questions answered over days | `AgentTask` awaited from a tool in text: a handoff plus a deferred call; answers kept in the state |
| `15_renewal_email_graph.py` | insurance | a renewal email drafted and reviewed until compliant | summarize/review task loop; graph-based review loop |

## Integration examples

These examples cover channels, thread handles, and toolset composition.

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
- A text turn is `Agent.run(message, message_history=, state=, deps=)`: it appends to the
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
- A delegate is a toolset in `tools=`, by text and on a call. Agents that serve text carry `llm=`; a call brings its
  model on the thread.

## Conversational agent tasks

The [agent task API](../../agent-tasks.md) uses Agent definitions,
persisted draft state, automatic InputRequired waits, child Call(agent, context=...),
and typed final output or output functions.

General code workflows and
graph authoring are [deferred](../../code-workflows.md). Durable execution,
automatic recovery, and external wake-ups are also outside the current phase.
