# Getaway

A travel concierge with three desks behind it, on the local stack of [`../README.md`](../README.md): it plans a trip to any city, books the flight and the hotel, and charges them, by text and on the phone. Nothing is seeded, so nothing needs to be known before the first message: fares and rooms are made up from the city and the date.

| File                | Agent           | What                                                                                                                    |
| ------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `frontend_agent.py` | `getaway`       | Talks to the traveler, on the phone and by text (the `frontend` endpoint). Its two tools are the flight and hotel desks. |
| `flights_agent.py`  | lane `flights`  | The flight desk: a slow search with two progress steps, and a booking paid through the payments desk.                   |
| `hotels_agent.py`   | lane `hotels`   | The hotel desk: a slow search the traveler can cancel, and a booking paid through the payments desk.                    |
| `payments_agent.py` | lane `payments` | The payments desk, reached only by the other two: it asks which card, reads the charge back and says the receipt.       |
| `travel.py`         |                 | The travel systems the desks share: fares, hotels and the payment ledger.                                               |
| `main.py`           |                 | The one entrypoint: the server, the four handlers added to it.                                                          |
| `scenarios/*.yaml`  |                 | Simulation scenarios: `trip`.                                                                                            |

## What the pattern is made of

- **Two experts on one front.** `Agent(tools=[A2ADelegate(FLIGHTS, name="flights"), A2ADelegate(HOTELS, name="hotels")])`: a trip goes to both in one step, both work at once, and their answers come back as each ends.
- **A chain of experts.** The flight and hotel desks each hold `A2ADelegate(PAYMENTS, name="payments")`. The payments desk's question, its progress, its read-back and its receipt reach the traveler through two hops, and the traveler's answer goes back down the same chain to the task that asked.
- **Several open questions.** When both desks charge in one step, both ask which card; the concierge asks once, and the one answer continues both tasks. `answer_detector=True` picks which open question a later reply answers.
- **Exact text two hops away.** The read-back is `requires_approval=` on `charge`, and the receipt is `ctx.say()`: both reach the traveler as written, never reworded by the desk or the concierge.
- **The lane's state** on the payments desk keeps the card the traveler picked, so the trip's second charge asks only for the yes.
- **Work in the background.** Every door takes the `live` inbox: a search's first `ctx.update()` ends its turn, a question to the same desk runs beside it, and the search's result comes on its own.

## Run

With the stack and the environment of [`../README.md`](../README.md), from this directory:

```sh
GETAWAY_NOW=2026-06-12 uv run python main.py dev
```

The console (`../../console`) lists this example and picks it once it is up; `../../dev.sh getaway` starts both. `GETAWAY_NOW` pins the clock, so the scenarios' dates stay put.

## Try it

By text, in order:

1. "Flight and hotel in Lisbon, please." Both desks get it at once; each asks for what it is missing.
2. "From Boston, June 19 to June 22, two of us." Both searches run, with progress from each.
3. "What time is check-in?" The hotel desk answers while its search still runs.
4. "The cheapest flight and the cheapest hotel." Both desks charge through payments, which asks which card once for both.
5. "The Visa." Each charge is read back as written; "yes" to each books it, and the receipts come as written.

Or say "never mind the hotel" while it searches: the hotel desk cancels the search.

## Scenarios

```sh
lk agent simulate text --scenarios scenarios/trip.yaml --agent-name getaway
```
