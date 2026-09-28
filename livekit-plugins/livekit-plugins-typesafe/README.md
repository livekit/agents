# TypeSafe decisions for LiveKit Agents

Jev evaluates probability questions, named choices, and ordered scores in one request.
The plugin supports the TypeSafe API and the OpenRouter Decisions API.

Use `typesafe.Jev()` with `TYPESAFE_API_KEY` for direct access.
Use `typesafe.Jev.with_openrouter()` with `OPENROUTER_API_KEY` for OpenRouter access.

## Run the receptionist example

From the repository root, install the workspace dependencies:

```sh
uv sync --all-extras --dev
```

Set `OPENROUTER_API_KEY` in `.env` for conversation and decisions.
For voice, also set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` for your LiveKit Cloud project.

Start the interactive console:

```sh
lk agent console examples/voice_agents/decision_receptionist.py
```

The console shows the transcript and decision results as they arrive.
Each completed user turn triggers callback-intent, request-category, and frustration checks.
Press `m` to mute the microphone, `Ctrl+T` to switch to typing, or `q` to quit voice mode.

With only an OpenRouter key, start in text mode:

```sh
lk agent console --text examples/voice_agents/decision_receptionist.py
```

For scripted testing, use the debugger:

```sh
lk agent debugger start examples/voice_agents/decision_receptionist.py
lk agent debugger say "I'd like a table for two tomorrow."
lk agent debugger say "Please ask a staff member to call me back."
lk agent debugger say "Yes, I still want that callback."
lk agent debugger logs --last 30
lk agent debugger stop
```

Jev detects explicit callback requests and acceptance of an offered callback. At a probability of 0.9, the example calls `log_callback_desire`.
This placeholder only writes to the console. A session-local guard limits it to one successful invocation, even when the caller repeats the request.
If the placeholder fails, a later decision can retry it.
The LLM can offer and acknowledge callbacks, but has no callback tool.
After the placeholder succeeds, the handler asks the LLM to confirm that it recorded the request.
A failed or timed-out decision produces no confirmation. Background detection does not guarantee callback capture.
All three decision checks continue throughout the conversation.
Voice uses Deepgram STT and Cartesia TTS through LiveKit inference.

## Background decisions

Set `AgentSession(decision_model=..., decision_options=...)` and `Agent(decisions=...)`.
The session emits `decisions_completed` with typed results, model identity, and the source message, agent, and activity IDs.
Decision requests run independently of conversational replies.

The options are:

| Option | Default | Meaning |
| --- | --- | --- |
| `turn_interval` | `1` | Run after every N committed, non-empty user text turns. |
| `max_context_turns` | `6` | Include the last N user turns and intervening assistant text, through the triggering message. |
| `timeout` | `10.0` | Limit each background request, including retries, in seconds. |
| `include_context_events` | `False` | Include tool calls, tool outputs, handoffs, and interruption metadata within the context window. |
| `allow_partial` | `False` | Emit valid results and an `errors` mapping for decisions with missing or invalid answers. |

By default, context contains user and assistant text only. Jev excludes system and developer messages, agent instructions, and non-text content in both modes.
Context can include earlier agents' conversation.
Each activity starts a new cadence count.

One request runs at a time. While it runs, eligible turns replace one pending snapshot with the newest snapshot.
Intermediate snapshots can be skipped. A completed result still identifies its original input, even when newer messages exist.
On handoff or close, the runner cancels pending work and suppresses late results from the old activity.
Request failures are logged and do not stop conversational replies or subsequent evaluations.

## On-demand evaluation

Pass an explicit context to evaluate a decision before an application action:

```python
from livekit.agents import decisions, llm, utils
from livekit.plugins import typesafe

async def evaluate_request() -> float:
    async with utils.http_context.open():
        model = typesafe.Jev.with_openrouter()
        context = llm.ChatContext.empty()
        context.add_message(role="user", content="Please let me speak to a person.")
        response = await model.evaluate(
            chat_ctx=context,
            decisions={"handoff": decisions.Probability("The caller requests a human.")},
        )
        result = response.results["handoff"]
        assert result.kind == "probability"
        return result.value
```

In `on_user_turn_completed`, add `new_message` to a copy of `turn_ctx` before evaluation.
The new message is not yet in the committed session history.

For post-session evaluations, pass the session history with `include_context_events=True`:

```python
response = await model.evaluate(
    chat_ctx=session.history,
    decisions={
        "resolved": decisions.Probability("The agent resolved the caller's request."),
        "intent": decisions.Choice(
            "What did the caller want?",
            options={"booking": "A reservation", "other": "Anything else"},
        ),
    },
    include_context_events=True,
    allow_partial=True,
)
for name, result in response.results.items():
    print(name, result.value)
for name, error in response.errors.items():
    print(name, error)
```

With `allow_partial=True`, each requested decision appears in either `results` or `errors`.
Without this option, one missing or invalid answer rejects the batch.
Request failures and invalid batch structures still raise exceptions in both modes.
Usage includes the full request, including decisions with invalid answers.

Responses and events expose `model` and `provider`.
The model ID comes from the provider response, with the configured model as a fallback.
Each evaluation creates a `decision_model.evaluate` trace span, including on-demand calls.

All results expose `kind` and `value`. Probability values do not apply a boolean threshold.
Choice results preserve the distribution across option names.
Score results preserve the distribution across level indices and the level descriptions.
A score is the expected zero-based level index, so it can be fractional.
Validation allows independent rounding to two decimal places. Results retain the provider's values and distributions.
Score validation requires a returned legend that matches the requested level descriptions and indices.
The Jev `provider_data["confidence"]` value describes distribution concentration, not the probability that the answer is correct.

Models declare supported decision kinds through `capabilities`.
The session validates decisions before startup or handoff. A missing model or unsupported kind raises `ValueError` without changing the active agent.
The session reports completed request duration and token usage through `metrics_collected` and `session.usage`.
For on-demand requests that belong to a session, use `session.decision_model.evaluate(...)`.
This session-bound model isolates usage, even when multiple sessions share one provider model.
Calls through the original provider model are standalone evaluations and do not contribute to session usage.
The original model's `metrics_collected` event still includes all requests.
OpenTelemetry exports reported tokens as `lk.agents.usage.decision_input_tokens` and
`lk.agents.usage.decision_output_tokens`, with provider and model attributes.
The current session protocol has no decision usage variant, so it carries decision token counts in its LLM usage variant.
The full session report retains the decision usage type and request count.
