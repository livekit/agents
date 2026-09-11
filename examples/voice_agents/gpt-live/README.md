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

The model hands the work over as a `delegation_created` event and waits. Nothing on the wire can reach a framework tool, so the agent carries no tools at all — passing any raises `RealtimeError` when the session starts, rather than leaving an agent whose tools silently never run.

The tools live on an ordinary `llm.LLM` that `run_delegation` drives itself, and the answer goes back through `append_commentary(answer, delegation_id=...)`, which the voice model says in its own words, capped at 500 tokens.

The event arrives before the caller's turn reaches the chat context, so the words that triggered it ride on it as `pending_transcript`; everything before that is already history.

`delegation_created` is emitted from the plugin's read loop, so the handler starts a task and returns instead of blocking it.

### A simple example, and what it cannot do

Each delegation answers on its own, in its own task, knowing only its own request. So a later delegation cannot replace an earlier one: ask to book Monday, change your mind to Tuesday a moment later, and both run and both answer.

Superseding needs one expert that holds the whole conversation and sees the correction. The framework will support it.
