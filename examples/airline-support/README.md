# Skyline Air support

A made-up airline's customer care, on the local stack of [`../README.md`](../README.md): a
front desk that talks to the customer, by text and on the phone, and a back office it reaches
over A2A, which holds the airline's systems, decides, asks for what is missing and pays.

| File                | What                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------ |
| `frontend_agent.py` | The front desk: `RemoteAgent(...).as_tool()` to the back office, the phone and text doors. |
| `backend_agent.py`  | The back office: `triage` and `continue_request`, the specialists and the payment guards.  |
| `skyline.py`        | The airline's own system: bookings, flights, baggage files, payments, the policy, in memory. |
| `main.py`           | One worker for both, each on its own lane.                                                  |
| `scenarios/*.yaml`  | Simulation scenarios: `compensation`, `missing_details`, `spam`.                            |

## The back office

```python
@backend.tool
async def triage(ctx: RunContext[SupportContext], email: Email) -> TriageResult | InputRequired:
    ...

@back_office.a2a_session(endpoint="backend", inbox="queue")
async def text(ctx: A2ASessionContext) -> RunResult[str]:
    app = SupportContext()
    await ctx.store.load_context(app)
    result = await backend.run(ctx.message, message_history=ctx.message_history, context=app)
    await ctx.store.save_context(app)
    return result
```

- **Questions wait across turns.** Without a booking reference of the sender's own, or a bag's
  PIR, the tool returns `InputRequired`; the ticket is saved with `persisted` context and
  `continue_request` resumes it on the customer's answer. After three questions a person takes
  over.
- **Agents called as functions** where judgment is needed: a classifier reads the email, a
  writer drafts the reply, a reviewer checks it against the facts and the policy, two rounds at
  most.
- **Code for everything with a right answer**: the booking lookup (only for the passenger's own
  address), EU261 by delay, cause and distance, the baggage allowance, the fare rules, and what
  was already paid.
- **Money moves last.** Payment runs only after an approved reply and only up to EUR 400.

`TriageResult.action` is `resolved` (answered, and paid when something was due), `to_human`
(above EUR 400, a reply failing review twice, or a booking never found) or `close` (spam).

## Run

With the stack and the environment of [`../README.md`](../README.md), from this directory:

```sh
uv run python main.py dev
```

The console (`../../console`) then talks to the front desk. Try: "Tom Okafor,
tom.okafor@example.com. You cancelled SK517 to Lisbon two days before departure", then the
booking reference it asks for, SKY5C1: EUR 400 is paid.

| The customer                    | Expected                                                                  |
| ------------------------------- | ------------------------------------------------------------------------- |
| Ana Garcia, SKY4F2              | SK204 3h20 late, 1240 km: EUR 250 paid                                    |
| Marcus Lee, SKY7Q9              | SK880 5h late, 5850 km: EUR 600 due, a colleague answers                  |
| Priya Nair, SKY2M8              | SK312 4h late, thunderstorm: nothing due                                  |
| Tom Okafor, SKY5C1              | SK517 cancelled two days ahead, 1850 km: EUR 400 paid                     |
| Eleanor Smith, SKY9B3           | a bag three days late, EUR 180 of receipts, PIR AMSSK10422: EUR 150 paid; or a refund of the flex fare: EUR 320 paid |
| anyone with another's reference | not found under their address, asked again                                |

The airline is seeded per worker, so a second claim on the same booking pays only the
difference; restart the worker to start over.

## Scenarios

```sh
lk agent simulate text --scenarios scenarios/missing_details.yaml --agent-name skyline-support
```

`compensation` covers a payment made on the spot, one above what is paid without a person, and
an extraordinary circumstance; `missing_details` the waits: no booking reference, a bag without
its PIR, someone else's booking; `spam` a message with nothing to act on.

## Tests

`tests/test_airline_support_example.py` runs the back office on scripted models: the EU261
payment, an earlier payment taken off, the over-limit hand-off, a reply failing review twice,
spam, and a missing booking reference asked, saved and continued on the answer.
