# Agent use cases

These examples combine shared agent definitions, application context, explicit
persistence, typed tool arguments, and voice or text entrypoints.

| File | Use case | API behavior |
| --- | --- | --- |
| [01_appointments_by_phone_and_text.py](01_appointments_by_phone_and_text.py) | Clinic scheduling | One context carries a runtime HTTP client and persisted appointment data; voice and text use the same tools |
| [03_insurance_eligibility.py](03_insurance_eligibility.py) | Payer eligibility | A remote expert reports progress through a remote agent tool |
| [04_a_claim_with_a_photo.py](04_a_claim_with_a_photo.py) | Claim intake | A tool returns InputRequired; the final result is a typed Claim |
| [07_hospital_main_line.py](07_hospital_main_line.py) | Hospital departments | Application code explicitly selects an agent from the saved department; ordinary nested runs provide direct delegation |
| [10_mortgage_prequalification.py](10_mortgage_prequalification.py) | Mortgage questionnaire | Answers are persisted between turns and the final application is validated against recorded answers |
| [planned-06_card_servicing_toolsets.py](planned-06_card_servicing_toolsets.py) | Card servicing | Filtered tools, approval, and runtime bank clients |
| [planned-08_summarizing_the_simulations.py](planned-08_summarizing_the_simulations.py) | Simulation analysis | Ordinary application code runs a digester and a reviewer with a bounded transcript-read budget |
| [planned-09_speed_to_lead.py](planned-09_speed_to_lead.py) | Opted-in sales calls | A CRM creates a conversation thread and dispatches an outbound call; a representative can resume the thread |
| [planned-12_many_chats.py](planned-12_many_chats.py) | Multiple chats | Each thread has its own persisted title and pinned facts |
| [planned-14_returns_by_email.py](planned-14_returns_by_email.py) | Email returns desk | Typed photo references, RMA creation, and explicit warehouse status queries |

## Context and persistence

`Agent[Context]` describes an agent's application context. Tools and instruction
functions receive `RunContext[Context]` and use `ctx.context`. The context can
contain HTTP clients and domain services alongside fields marked `persisted`.

The application constructs runtime dependencies, calls `store.load_context(app)`,
runs the selected agent with `context=app`, and calls `store.save_context(app)`
to retain changes. A marker selects fields; changing a marked field does not
save it. Voice tools save changed application data explicitly through `ctx.store`.

The framework manages the thread history, transport records, storage connection,
and leases. The application's service APIs remain authoritative for bookings,
claims, payments, and inventory.

## Agent invocations

`agent.run(message, message_history=history, context=app)` runs that agent.
A completed result exposes the typed `result.output`; an input-required result
exposes `result.question`. Check `result.status` before accessing either.
Another human message starts a new invocation with saved history and context.

A tool can explicitly await another agent's `run()` when its work completes
within that invocation. [AgentRouter](../../agent-router.md) specifies routing
frames for conversations that move between agents over multiple turns.

Effects require domain validation and stable idempotency contracts. Saving a
draft does not make an external operation atomic with the conversation store.
