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
Each completed user turn triggers a decision request. Decisions stop after the simulated handoff.
Press `m` to mute the microphone, `Ctrl+T` to switch to typing, or `q` to quit voice mode.

With only an OpenRouter key, start in text mode:

```sh
lk agent console --text examples/voice_agents/decision_receptionist.py
```

For scripted testing, use the debugger:

```sh
lk agent debugger start examples/voice_agents/decision_receptionist.py
lk agent debugger say "I'd like a table for two tomorrow."
lk agent debugger say "I've asked three times. Please get me a person."
lk agent debugger logs --last 30
lk agent debugger stop
```

The example logs three decisions for each completed request. At a human-request probability of 0.9, it switches to an acknowledgement agent.
This simulates a handoff without transferring a real call. The acknowledgement agent has no background decisions.
Voice uses Deepgram STT and Cartesia TTS through LiveKit inference.

## Background decisions

Set `AgentSession(decision_model=..., decision_options=...)` and `Agent(decisions=...)`.
The session emits `decisions_completed` with typed results and the source message, agent, and activity IDs.
Decision requests run independently of conversational replies.

The options are:

| Option | Default | Meaning |
| --- | --- | --- |
| `turn_interval` | `1` | Run after every N committed, non-empty user text turns. |
| `max_context_turns` | `6` | Include the last N user turns and intervening assistant text, through the triggering message. |
| `timeout` | `10.0` | Limit each background request, including retries, in seconds. |

Context excludes system instructions, tools, and non-text content. Context can include earlier agents' conversation.
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

All results expose `kind` and `value`. Probability values do not apply a boolean threshold.
Choice results preserve the distribution across option names.
Score results preserve the distribution across level indices and the level descriptions.
A score is the expected zero-based level index, so it can be fractional.
The Jev `provider_data["confidence"]` value describes distribution concentration, not the probability that the answer is correct.

Models declare supported decision kinds through `capabilities`.
The session validates decisions before startup or handoff. A missing model or unsupported kind raises `ValueError` without changing the active agent.
The session reports completed request duration and token usage through `metrics_collected` and `session.usage`.
The current session protocol has no decision usage variant, so it carries decision token counts in its LLM usage variant.
The full session report retains the decision usage type and request count.
