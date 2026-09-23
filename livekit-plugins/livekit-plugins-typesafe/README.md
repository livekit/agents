# TypeSafe reviewer plugin for LiveKit Agents

Reviews a voice agent's replies against its own system prompt and tool catalog, and
nudges it back on course when it drifts, using [TypeSafe](https://typesafe.ai)'s System
One model, Jev.

Jev does not generate text. It answers typed questions about a state with calibrated
probabilities in 70 to 500ms, and every question in a request is answered in parallel
against one ingestion of that state. A five-check review of a drafted reply is therefore
a single round trigger, which is what makes it usable on a live call.

## Install

```bash
pip install livekit-plugins-typesafe
```

Set `TYPESAFE_API_KEY` in your environment ([console](https://console.typesafe.ai/)).

## Usage

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import typesafe

session = AgentSession(...)
reviewer = typesafe.Reviewer()
reviewer.attach(session)

await session.start(agent=Agent(instructions="..."), room=ctx.room)
```

Those two lines are the whole integration. The checks read the agent's `instructions`
and tool catalog on every review, so there is nothing to write per agent, and a handoff
to a different agent is picked up automatically.

## What it judges

Five checks, all derived from the prompt rather than hand-authored per agent:

| Check | Type | Triggers when |
| --- | --- | --- |
| `follows_instructions` | Noul | the reply breaks a rule stated in the agent's own instructions |
| `unsupported_claim` | Noul | the reply asserts a concrete fact backed by nothing in the prompt, transcript, or tool results |
| `advances_task` | Noul | the reply has drifted off the task the prompt describes |
| `expected_tool` | Choice | a tool clearly should have been called and was not |
| `severity` | Score | the conversation is far enough off course to matter |

The defaults were measured against Jev 1.13 rather than guessed, on a small labelled
set of nine replies under one prompt. They sit in the gap between how Jev scores a
compliant reply and how it scores a violating one:

| Check | compliant | violating | default |
| --- | --- | --- | --- |
| `follows_instructions` | 0.46 to 0.93 | 0.01 to 0.03 | 0.25 |
| `unsupported_claim` | 0.06 to 0.29 | 0.93 to 0.99 | 0.6 |
| `advances_task` | 0.42 to 0.94 | 0.02 to 0.05 | 0.25 |
| `severity` | 0.05 to 1.87 | 2.51 to 2.83 | 2.2 |

The gap is asymmetric, which is worth knowing before you move a threshold. Jev is close
to certain about a violation and much less certain that nothing is wrong, because
spotting one broken rule is easier than confirming every rule held. That is why the
thresholds sit near the violating end rather than halfway. A threshold of 0.5 on
`follows_instructions` looks reasonable and fires on replies that are perfectly fine.

Because those numbers belong to one model version, the client pins `jev-1.13.0`
rather than following the `jev-latest` alias. An alias moves on TypeSafe's schedule,
and a version that scores replies differently would leave these thresholds quietly
mis-set. To track the alias instead, pass `model="jev-latest"` and re-measure when it
moves; the reviewer warns once if the model that answered is not the one the defaults
were measured on.

Nine replies under one prompt is a small sample, so re-measure on your own calls:

```python
reviewer = typesafe.Reviewer(
    checks=typesafe.default_checks(unsupported_claim=0.8, severity=2.5),
)
```

You can append your own `Check`. It goes out in the same request as the rest.

## Observe vs. gate

Observe is the default and costs nothing on the reply path. The review starts once the
assistant's message is committed, after the agent has finished speaking that turn, and
finishes well before the next generation because the user has to say something first. A
trigger appends a correction to the chat context, and the next reply corrects itself. The
caller hears nothing.

Gate holds a draft back until it clears. Opt in per check and delegate from `llm_node`:

```python
class MyAgent(Agent):
    def __init__(self, reviewer: typesafe.Reviewer) -> None:
        super().__init__(instructions="...")
        self.reviewer = reviewer

    async def llm_node(self, chat_ctx, tools, model_settings):
        return await self.reviewer.gate(self, chat_ctx, tools, model_settings)


reviewer = typesafe.Reviewer(checks=typesafe.default_checks(gated_check_ids=["unsupported_claim"]))
```

A gated trigger redrafts once with the correction attached, before any audio. Nothing bad
gets spoken, but the reply now waits for the full generation plus one evaluation instead
of streaming, so reserve it for checks worth paying that on every turn. Drafts carrying
tool calls pass straight through, because nothing is spoken for those.

## Metrics

`on_verdict` fires for every review, whether or not a check triggered:

```python
typesafe.Reviewer(on_verdict=lambda v: print(v.triggered_checks, v.answers, v.duration))
```

## Logs

Everything logs under `livekit.plugins.typesafe`. Turn it up to see a line per turn:

```python
logging.getLogger("livekit.plugins.typesafe").setLevel(logging.DEBUG)
```

```
DEBUG  reviewer attached
DEBUG  check !follows_instructions=0.04 !unsupported_claim=0.91 advances_task=0.88 \
             !expected_tool=lookup_repair@0.86 !severity=2.30@0.71 [180ms 412tok jev-1.13.0]
INFO   nudging the agent back on course
```

Keep the `check` line. A leading `!` marks a check that triggered, and every check
reports its value either way. Without that you cannot tell a check sitting at 0.51 from
one sitting at 0.99, which is what tells you whether a threshold is in the right place.
The bracket carries round-trigger latency, billed input tokens, and the versioned model
that answered. `jev-latest` moves under you, so a threshold is only valid against the
version it was tuned on.

| Level | When |
| --- | --- |
| `DEBUG` | attach/detach, one `check` line per turn, gate cleared / redrafting |
| `INFO` | a check triggered and the agent was nudged |
| `WARNING` | check failed and the turn went unjudged; gate gave up after redrafting; a check could not read its answer |
| `ERROR` | a check crashed, or your `on_verdict` callback raised |

A `WARNING` also fires once per reviewer if the model that answered is not the one the
default thresholds were measured against.

Reply text is tagged `lk.pii.reviewed_reply` and truncated to 500 characters, so
it travels with the repo's other PII fields and can be redacted the same way.

Structured fields ride on `extra`, so a JSON formatter gets `triggered_checks`, `reasons`,
`elapsed_ms`, `model`, `checks`, and `state_chars` as real fields instead of prose. For
per-turn metrics in your own pipeline, use `on_verdict` instead of parsing logs. If your
callback raises, the exception is caught and logged and never reaches the call.

## Tuning against how calls actually ended

The thresholds above are guesses until something grades them, and a caller hanging up
is one outcome label you get free on every call. `history` keeps the probabilities
behind every turn, so you can pair the two:

```python
from livekit.agents import CloseEvent, CloseReason

@session.on("close")
def _on_close(ev: CloseEvent) -> None:
    if ev.reason is CloseReason.PARTICIPANT_DISCONNECTED:
        store(call_id, [v.answers for v in reviewer.results])
```

Collect a few hundred calls, compare the distributions on calls that ended this way
against calls that ran to completion, and move each threshold to where it would have
separated them. The same exercise shows you which checks earn nothing and can be
dropped.

An early hangup is not a check. The call is already over when you see it, so there is
nothing left to correct. Its use is as the label you grade the checks against.

## Limitations

- It fails open. A timeout, rate limit or outage is logged and the turn goes unjudged
  instead of silencing the agent. This is a review layer and not an enforcement
  boundary, so do not make it the only thing between a caller and a harmful answer.
  Such a verdict carries `evaluated=False`, which keeps "nothing was wrong"
  distinguishable from "nobody looked". `ok` on its own cannot tell you which.
- Gating needs an `llm_node`. Realtime and speech-to-speech models have none, so they
  can only be observed.
- `expected_tool` assumes one tool per turn. An agent that legitimately calls two tools
  in a turn will show low confidence there. The check only triggers above a confidence
  floor, so it stays quiet instead of firing wrongly.
- A reply is judged against the agent that produced it, captured when the message is
  committed, and a correction is dropped rather than written into a different agent
  after a handoff. `conversation_item_added` does not name the message's author, so a
  handoff landing before that event is still attributed to the successor.
- Jev works best in English. It handles other languages less well, so test on your own
  traffic before trusting the thresholds.

## Cost

One review is roughly 1 to 3k input tokens at \$0.042 per million, and output tokens are
free. That works out to about a hundredth of a cent per turn. More checks cost more
tokens but no more round triggers.
