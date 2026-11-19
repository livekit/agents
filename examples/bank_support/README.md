# Bank Support Example

A voice agent for card and transfer support at a fictional bank. It shows how to keep a tool-calling voice agent on track when callers are vague, change their minds, or are hard to hear.

For setup instructions, see the [main examples README](../README.md).

## Overview

The caller verifies with a phone number and date of birth. The agent can then:

- block or unblock a card, and order a replacement,
- dispute one or more transactions,
- move money between the caller's accounts, within a daily limit,
- add a travel notice to a card,
- hand off to a person.

The data lives in an in-memory `FakeBank` (`bank_db.py`), so the example runs without a backend.

## The practices this example shows

Each practice is a mechanism in code, not a line in the prompt. The prompt stays short.

### 1. A contract table enforces the order of operations

`flows/bank.py` declares one `Rule` per tool:

| Field | Meaning | Example |
| --- | --- | --- |
| `needs` | An argument must come from an earlier tool result. | `block_card.card_id` comes from `get_card`. |
| `after` | Tools that must have succeeded first in this call. | Every read comes after `verify_identity`. |
| `after_any` | At least one of these must have happened. | `end_call` comes after a write or a hand-off. |
| `formats` | A regex for an argument, with a spoken description. | A phone number has ten digits. |
| `undo` | The tool that takes this write back. | `block_card` is undone by `unblock_card`. |
| `requires_closing_question` | The agent must have asked a question, and the call must quote the caller's answer. | `end_call`. |

`Contract.guarded` wraps each tool. Before the call, it checks the rule and returns a short `NOT DONE` reason when the call is out of order. The model reads the reason and corrects itself on the next step. A repeated write returns `ALREADY DONE` instead of running twice, unless a tool it depends on replied again since. For example, a second transfer of the same amount needs a second `transfer_check`.

Use `after` only for an order that must hold inside one call, such as verifying the caller first. A rule about the world, such as "only a blocked card can be replaced", belongs in the tool, which reads the current state. Then a card blocked on an earlier call can still be unblocked or replaced.

### 2. Tool results say where their values go

After a read, `Contract.narrate` appends which later tool uses each value, for example `(card_id for block_card: card_4421)`. After a write, it says that the write is done and which tool takes it back. The model does not need to guess an identifier or repeat a write to be sure.

### 3. One task per write

Each write flow (block, unblock, dispute, transfer, travel notice) runs as its own `AgentTask` with only the tools that flow needs. The parent agent starts a flow through a `start_<flow>` tool. The task stays open after a write, so the caller can add to the same request, for example a replacement after a block. The model calls `finish_request` when the caller is done with the request, and the task refuses to finish until one of its writes succeeded. The model calls `end_request` when the caller drops the request. Small tool sets keep a smaller model from reaching for the wrong tool.

### 4. Digits are read back from the transcript, not from memory

`annotate_user_turn` adds a `[heard: 4155550123]` note to each caller turn that contains spoken digits, including "double five" and "triple one". The STT gets the digit words as keyterms. The phone format rule rejects nine digits, so the agent reads the number back and asks for the missing part instead of guessing.

### 5. The agent speaks while slow tools run

If a guarded tool takes more than two seconds, the agent says "One moment." It repeats the filler every 3.5 seconds, at most three times. The caller does not hear dead air and does not talk over the tool call.

### 6. The call ends only after a closing question

`end_call` requires some handled request: a write, an answered question, a hand-off, or a request the caller dropped. The agent's last turn must also be a question, and `end_call` must quote the caller's answer in `caller_reply`. The hook checks that the quote matches the caller's last turn. The model decides whether the answer means "nothing else", so the check has no word lists and works in any language. The agent says a closing line and the session closes after the audio drains.

### Session configuration

`build_session` sets these values:

- Deepgram `nova-3` with digit keyterms for STT, through LiveKit Inference.
- An open-weight LLM through LiveKit Inference. The contract keeps a small model on track.
- `parallel_tool_calls=False`, so writes happen one at a time in a known order.
- Endpointing delays of 1.2 to 3.0 seconds, because callers pause while they read card numbers.
- Adaptive interruption, so a short "mm-hm" does not stop the agent.
- `max_tool_steps=8`.

## Adapt it to your domain

Most changes happen in `flows/bank.py`:

1. Replace the tools with your own reads and writes.
2. Write one `Rule` per tool in `CONTRACT`.
3. Add one `Flow` per write in `FLOWS`.

`contract.py` and `agent.py` have no bank-specific logic, except for the prompt and the digit words, which are English.

## Simulations

`scenarios.yaml` holds eight simulated callers: one per flow, a caller who takes a block back, a caller who drops a digit, a caller who is not the account holder, and a request that needs a person. All simulation glue is in `simulation.py`.

Each scenario's `userdata` seeds the bank and grades the result:

- `transferred_today`: an amount already moved today, to test the daily limit.
- `balances`: balance overrides by account kind.
- `expected`: the final bank state the run must reach (`card_status`, `replacement`, `transfers`, `disputes`, `travel_notices`). `on_simulation_end` fails the run when the state does not match.

## Tests

```bash
uv run pytest examples/bank_support
```

- `test_contract.py` tests the rule checks, narration, and digit parsing.
- `test_flows.py` tests the transfer limit and dispute totals.
- `test_agent.py` tests the agent wiring and the task tool sets.
