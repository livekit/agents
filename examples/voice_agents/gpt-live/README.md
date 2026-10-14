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

### Images

The voice model has no eyes, so an image goes to the backend model. There are two ways in, and they are the ordinary ones.

An `llm.ImageContent` in the chat context — a data URL, an external URL, or a `VideoFrame` — becomes a Responses image input item, with the message's own words as its caption. This is the path for a screenshot the caller is about to ask about, and the one that keeps the chat context and the backend's input telling the same story.

```python
chat_ctx.add_message(role="user", content=["what is on my screen?", ImageContent(image=shot)])
```

An image keeps the role that carried it, so a standing reference image on a `system` or `developer` message reaches the backend as one. Only `assistant` is left out: an assistant turn is output, and the API has no way to take one as input. A message carrying nothing but an image is fine — it adds no words to the conversation and its image still goes.

The item itself is built by the same converter the Responses plugin uses (`to_provider_format(format="openai.responses")`), so the detail level, an external url the backend can fetch itself, and the ordering of an image against its caption are decided in one place rather than twice.

`push_video(frame)` sends a single frame and records nothing, exactly as on `RealtimeModel`. `RoomInputOptions(video_enabled=True)` feeds the room's video track through it at the session's `video_sampler` rate, about 1 fps while the caller speaks — every one of those frames is input the backend keeps, so a track left on for a whole call costs far more than one image sent when it is wanted.

Either way nothing runs on its own: the image waits in the backend's input until the voice model next delegates, or until a tool result continues the backend. So send the screenshot, then let the caller ask about it.

An image belongs to the connection it was queued on. If that session ends before the image goes out, it is dropped and logged rather than replayed into the next session, whose backend knows nothing about it — send it again.

`delegation="client"` has no backend to look at anything. An image in the chat context is dropped with a warning while its words still reach the voice model, and `push_video` drops the frame with one warning per session, since video arrives on its own and must not break its own forwarding.

Asks that are not the caller's words — `generate_reply()` with no input, or `generate_reply(instructions=...)` — stay the voice model's under either setting.

## Client delegation — `client_delegation.py`

The model hands the work over as a `delegation_created` event and waits. Nothing on the wire can reach a framework tool, so the agent carries no tools at all — passing any raises `RealtimeError` when the session starts, rather than leaving an agent whose tools silently never run.

The tools live on an ordinary `llm.LLM` that `run_delegation` drives itself, and the answer goes back through `append_commentary(answer, delegation_id=...)`, which the voice model says in its own words, capped at 500 tokens.

The event arrives before the caller's turn reaches the chat context, so the words that triggered it ride on it as `pending_transcript`; everything before that is already history.

`delegation_created` is emitted from the plugin's read loop, so the handler starts a task and returns instead of blocking it.

### A simple example, and what it cannot do

Each delegation answers on its own, in its own task, knowing only its own request. So a later delegation cannot replace an earlier one: ask to book Monday, change your mind to Tuesday a moment later, and both run and both answer.

Superseding needs one expert that holds the whole conversation and sees the correction. The framework will support it.
