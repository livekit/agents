"""Live probe: how often does a Responses reply carry more than one message, and what is spoken?

Each target runs N conversations of several caller turns through its real plugin LLM,
one LLM per conversation, feeding every reply (and canned tool results) back into the
history the way an agent session does. For every reply the probe records
the raw output items (type, phase, text) next to what the shared engine sent the caller
(spoken text, tool calls, dropped item ids), and classifies the reply:

- ``single``: at most one message with text.
- ``commentary_chain``: several messages, each one before the last labelled ``commentary``.
- ``hosted_continuation``: several messages separated by a server-side tool call.
- ``multi_message``: several messages with neither between them. These are the replies the
  first-message rule trims, so the probe prints each one for review.

A reply with function tools also checks that every raw ``function_call`` reached the caller,
in order.

Usage (keys from the environment; a target whose key is missing is skipped):

    uv run python scripts/responses_multi_message_probe.py --n 100 --turns 4 --out /tmp/probe.jsonl
    uv run python scripts/responses_multi_message_probe.py --targets azure-http-gpt-4.1 \\
        --prompt back_to_back --n 25
    uv run python scripts/responses_multi_message_probe.py --n 50 \\
        --azure prod_dz=https://<resource>.openai.azure.com/openai/v1/,AZURE_KEY_ENV,gpt-4.1-datazone

Environment: OPENAI_API_KEY, XAI_API_KEY, AZURE_OPENAI_API_KEY and AZURE_OPENAI_BASE_URL
(an Azure ``.../openai/v1/`` URL).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from livekit.agents import llm
from livekit.agents.llm.tool_context import function_tool
from livekit.agents.types import NOT_GIVEN

# A system prompt that embeds an example transcript with speaker labels is what makes
# gpt-4.1 on Azure HTTP return alternate messages (~30%); short prompts without one did
# not in 440 tries. Keep the wording mild: staging moderation flags injury-heavy text.
CLAIMANT_SYSTEM = """You are role-playing Sam Lee, a claimant on a follow-up call from an \
insurance adjuster about a minor car accident two weeks ago. Sam's neck was a little \
stiff afterwards. Reply only with what Sam says next: one to three short spoken \
sentences, no speaker label, no stage directions.

Here is a transcript of Sam's previous call, for tone:

Adjuster: Hi, is this Sam Lee?
Claimant: Yes, this is Sam.
Adjuster: I'm calling about your claim from the accident on the 19th. How are you doing?
Claimant: I'm doing okay. My neck was stiff for a few days, but it's easing up.
Adjuster: Did you see a doctor?
Claimant: I went to urgent care the next day. They said it was a mild strain.
Adjuster: Have you missed any work?
Claimant: Just the Monday after. I've been back since.
Adjuster: Thanks, Sam. I'll call again in a couple of weeks.
Claimant: Sounds good, thanks for checking in."""
CLAIMANT_HISTORY = [
    ("user", "Hi Sam, it's Dana from Harbor Insurance, following up on your claim."),
    ("assistant", "Hi Dana, thanks for calling."),
]
CLAIMANT_QUESTIONS = [
    "How is your neck feeling today compared to last week?",
    "Have you been able to turn your head while driving?",
    "Are you still taking anything for the stiffness?",
    "Has it been bothering your sleep?",
    "Did urgent care give you any exercises?",
    "On a scale of one to ten, how stiff is it right now?",
    "Have you needed any follow-up appointments?",
    "How has work been going since you went back?",
    "Has the car been repaired yet?",
    "Is there anything that makes it worse?",
]

BACK_TO_BACK_SYSTEM = (
    "You are Riley, the voice assistant for Harbor Insurance, on a claims follow-up call. "
    "Speak naturally and briefly, one or two sentences per turn."
)
# agents emit consecutive assistant turns (say() then generate_reply()), so history
# with two assistant messages in a row is common in real sessions
BACK_TO_BACK_HISTORY = [
    ("assistant", "Hi, this is Riley from Harbor Insurance."),
    ("assistant", "I'm calling to follow up on the claim you filed after your accident."),
    ("user", "Oh, hi. Yeah, I have a minute."),
    ("assistant", "Great, thanks."),
    ("assistant", "First, I just want to make sure you're doing okay."),
]
BACK_TO_BACK_QUESTIONS = [
    "I'm alright, my neck's still a bit stiff.",
    "Doing better, thanks. What do you need from me?",
    "Pretty good. Is my claim approved yet?",
    "I'm okay. Did you get the photos I sent?",
    "Not bad. How long will the repair take?",
    "Fine, but I've been waiting a while to hear back.",
    "Okay I guess. Do I need to see your doctor?",
    "Good. Can I get a rental car?",
    "I'm fine. What happens next?",
    "Alright. Who do I talk to about the bill?",
]

ACK_THEN_ASK_SYSTEM = (
    "You are Riley, the voice assistant for Harbor Insurance, collecting details for a new "
    "auto claim. Every turn, first acknowledge what the caller said in a few words, then ask "
    "exactly one question to collect the next missing detail: date of the accident, location, "
    "other driver's insurer, injuries, and photos. Keep it conversational and short."
)
ACK_THEN_ASK_HISTORY = [
    ("assistant", "Hi, you've reached Harbor Insurance claims. How can I help?"),
]
ACK_THEN_ASK_QUESTIONS = [
    "Hi, I was rear-ended yesterday and need to file a claim.",
    "Someone backed into my car in a parking lot last Tuesday.",
    "I hit a pole this morning, nobody else was involved.",
    "My car got sideswiped on the highway on Friday.",
    "I'm calling because a guy ran a red light and hit me.",
    "There was a fender bender at the school pickup line today.",
    "I need to report an accident from last weekend.",
    "Somebody hit my parked car overnight.",
    "I got into a small accident on the freeway on-ramp.",
    "My daughter had an accident in my car this afternoon.",
]

LONG_PERSONA_SYSTEM = """# Role
You are Riley, a friendly, efficient voice agent for Harbor Insurance's claims team. You \
speak with callers on the phone. Everything you write is read aloud by a text-to-speech \
engine.

# Style
- One to three short sentences per turn. Never use lists, markdown, emoji or URLs.
- Sound warm and human. Use contractions. Mirror the caller's wording.
- Never read out claim numbers digit by digit unless asked.
- If the caller is upset, acknowledge it before moving on.

# Goals, in order
1. Confirm the caller's name and policy number.
2. Find out why they are calling: new claim, claim status, or billing.
3. For claim status, give the status and the next step.
4. Offer to help with anything else, then close the call politely.

# Known facts for this call
- Caller: Sam Lee, policy HB-448210.
- Open claim: rear-end collision on October 19. Status: approved, repair shop booked for \
November 6, rental car authorized for 10 days.

# Example conversation 1
Caller: Hi, I'm calling about my claim.
Riley: Sure, I can help with that. Can I get your name and policy number?
Caller: Sam Lee, HB-448210.
Riley: Thanks, Sam. I see your claim from October 19th. It's approved, and the repair shop \
has you booked for November 6th.
Caller: Great. Do I get a rental?
Riley: You do, a rental's authorized for ten days. Anything else I can help with?

# Example conversation 2
Caller: I've been waiting forever, this is ridiculous.
Riley: I'm sorry about the wait, that's frustrating. Let's get this sorted. Can I get your \
name and policy number?
Caller: Sam Lee, HB-448210.
Riley: Thanks, Sam. Good news, your claim is approved and the repairs are scheduled.
"""
LONG_PERSONA_HISTORY = [
    ("assistant", "Thanks for calling Harbor Insurance, this is Riley. How can I help?"),
    ("user", "Hi, it's Sam Lee, policy HB-448210."),
    ("assistant", "Thanks, Sam. What can I do for you today?"),
]
LONG_PERSONA_QUESTIONS = [
    "I'm calling to check on my claim.",
    "When is my car getting fixed?",
    "Do I get a rental car?",
    "I've been waiting forever, what's going on with my claim?",
    "Is my claim approved?",
    "Can you tell me what happens next?",
    "How long do I get the rental for?",
    "Which repair shop is it?",
    "Will my premium go up?",
    "Can I change the repair date?",
]

SCRIPT_LINES_SYSTEM = """You are the Harbor Insurance voice agent. Follow this call script \
closely, adapting the wording to what the caller says. Say only the agent's next line.

Agent: Hi, this is Harbor Insurance calling for Sam Lee.
Agent: Is now a good time to talk for two minutes?
Caller: <answers>
Agent: Thanks. I'm following up on your claim from October 19th.
Agent: First, how are you feeling since the accident?
Caller: <answers>
Agent: I'm glad to hear that. Next, have you picked a repair shop?
Caller: <answers>
Agent: Perfect. You'll get a text confirming the appointment.
Agent: Is there anything else I can help with today?"""
SCRIPT_LINES_HISTORY = [
    ("assistant", "Hi, this is Harbor Insurance calling for Sam Lee."),
    ("assistant", "Is now a good time to talk for two minutes?"),
]
SCRIPT_LINES_QUESTIONS = [
    "Yeah, sure, go ahead.",
    "I've got a minute, what's up?",
    "Um, okay, quickly though.",
    "Sure. Is this about my car?",
    "Yes, this is Sam. What's this about?",
    "I guess, I'm driving though.",
    "Yep, now's fine.",
    "Sure, I was hoping you'd call.",
    "Okay, but make it fast.",
    "Yes. Did my claim go through?",
]

PROMPTS: dict[str, tuple[str, list[tuple[str, str]], list[str]]] = {
    "claimant": (CLAIMANT_SYSTEM, CLAIMANT_HISTORY, CLAIMANT_QUESTIONS),
    "back_to_back": (BACK_TO_BACK_SYSTEM, BACK_TO_BACK_HISTORY, BACK_TO_BACK_QUESTIONS),
    "ack_then_ask": (ACK_THEN_ASK_SYSTEM, ACK_THEN_ASK_HISTORY, ACK_THEN_ASK_QUESTIONS),
    "long_persona": (LONG_PERSONA_SYSTEM, LONG_PERSONA_HISTORY, LONG_PERSONA_QUESTIONS),
    "script_lines": (SCRIPT_LINES_SYSTEM, SCRIPT_LINES_HISTORY, SCRIPT_LINES_QUESTIONS),
}
_prompt_name = "claimant"

SCHEDULER_SYSTEM = """You are the scheduling assistant for Riverside Clinic, speaking \
with a caller by phone. Today is Monday, November 2, 2026. Use the tools to look the \
caller up, check availability and book. Keep spoken replies short and say what you are \
doing before you call a tool.

Here is a transcript of a good call, for tone:

Caller: Hi, I'd like to book a checkup.
Assistant: Sure, can I have your full name and date of birth?
Caller: Alex Kim, July 8th 1985.
Assistant: Thanks, Alex. Let me pull up your record.
Assistant: Found you. What day works best?
Caller: Wednesday morning.
Assistant: Let me check Wednesday. I have 9:30 or 11. Which would you like?
Caller: 9:30.
Assistant: You're booked for Wednesday at 9:30. Anything else?"""
SCHEDULER_REQUESTS = [
    "Hi, this is Sam Lee, born March 3rd 1990. Is Tuesday the 10th open? If so book me at 9.",
    "Sam Lee here, date of birth 3/3/1990. Can you check Thursday the 12th and book the "
    "earliest slot?",
    "It's Sam Lee, birthday March 3 1990. I need to move my appointment to Monday the 9th, "
    "anything after 2 works.",
    "Hello, Sam Lee, 1990-03-03. Please check Friday the 13th and also tell me the clinic address.",
]


@function_tool
async def lookup_patient(name: str, date_of_birth: str) -> str:
    """Find a patient record by full name and date of birth (YYYY-MM-DD)."""
    return "never called"


@function_tool
async def check_availability(date: str) -> str:
    """List open appointment times on a date (YYYY-MM-DD)."""
    return "never called"


@function_tool
async def book_appointment(patient_id: str, date: str, time: str) -> str:
    """Book an appointment for a patient at a date (YYYY-MM-DD) and time (HH:MM)."""
    return "never called"


@function_tool
async def get_clinic_info(topic: str) -> str:
    """Answer a question about the clinic, such as its address or opening hours."""
    return "never called"


SCHEDULER_TOOLS = [lookup_patient, check_availability, book_appointment, get_clinic_info]


@dataclass
class Target:
    name: str
    make: Callable[[], llm.LLM]
    tools: bool = False
    key_env: tuple[str, ...] = ()


def _server_default_reasoning(model: llm.LLM) -> llm.LLM:
    # the plugin injects a low effort for known reasoning models; clearing it sends no
    # `reasoning` field, so the provider applies the model's own default
    model._opts.reasoning = NOT_GIVEN  # type: ignore[attr-defined]
    return model


def _openai(model: str, default_reasoning: bool = False) -> Callable[[], llm.LLM]:
    def make() -> llm.LLM:
        from livekit.plugins import openai

        made = openai.responses.LLM(model=model, use_websocket=False)
        return _server_default_reasoning(made) if default_reasoning else made

    return make


def _azure(
    model: str,
    default_reasoning: bool = False,
    base_url_env: str = "AZURE_OPENAI_BASE_URL",
    key_env: str = "AZURE_OPENAI_API_KEY",
) -> Callable[[], llm.LLM]:
    def make() -> llm.LLM:
        from livekit.plugins import azure

        made = azure.responses.LLM(
            model=model,
            base_url=os.environ[base_url_env],
            api_key=os.environ[key_env],
            api_version="preview",
        )
        return _server_default_reasoning(made) if default_reasoning else made

    return make


def _azure_target(spec: str) -> Target:
    """Build a target from ``NAME=BASE_URL,KEY_ENV,DEPLOYMENT[,tools]``."""
    name, _, rest = spec.partition("=")
    base_url, key_env, deployment, *flags = rest.split(",")
    url_env = f"_PROBE_URL_{name}"
    os.environ[url_env] = base_url
    return Target(
        name,
        _azure(deployment, base_url_env=url_env, key_env=key_env),
        tools="tools" in flags,
        key_env=(key_env,),
    )


def _xai(model: str) -> Callable[[], llm.LLM]:
    def make() -> llm.LLM:
        from livekit.plugins import xai

        return xai.responses.LLM(model=model)

    return make


AZURE_KEYS = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL")
TARGETS = [
    Target("openai-gpt-5.4", _openai("gpt-5.4"), key_env=("OPENAI_API_KEY",)),
    Target("azure-http-gpt-5.4", _azure("gpt-5.4"), key_env=AZURE_KEYS),
    Target("azure-http-gpt-4.1", _azure("gpt-4.1"), key_env=AZURE_KEYS),
    Target("openai-gpt-5.6-sol", _openai("gpt-5.6-sol", True), key_env=("OPENAI_API_KEY",)),
    Target("azure-http-gpt-5.6-sol", _azure("gpt-5.6-sol", True), key_env=AZURE_KEYS),
    Target("openai-gpt-6-luna", _openai("gpt-6-luna", True), key_env=("OPENAI_API_KEY",)),
    Target("azure-http-gpt-6-luna", _azure("gpt-6-luna", True), key_env=AZURE_KEYS),
    Target("openai-gpt-6.1-sol", _openai("gpt-6.1-sol", True), key_env=("OPENAI_API_KEY",)),
    Target(
        "openai-gpt-6.1-sol-tools",
        _openai("gpt-6.1-sol", True),
        tools=True,
        key_env=("OPENAI_API_KEY",),
    ),
    Target("xai-grok-4-1-fast", _xai("grok-4-1-fast-non-reasoning"), key_env=("XAI_API_KEY",)),
    Target("xai-grok-4.3", _xai("grok-4.3"), key_env=("XAI_API_KEY",)),
    Target("azure-http-gpt-4.1-tools", _azure("gpt-4.1"), tools=True, key_env=AZURE_KEYS),
    Target("azure-http-gpt-5.4-tools", _azure("gpt-5.4"), tools=True, key_env=AZURE_KEYS),
]


@dataclass
class Reply:
    target: str
    index: int
    turn: int
    prompt: str
    items: list[dict[str, Any]] = field(default_factory=list)
    spoken: str = ""
    tool_calls: list[str] = field(default_factory=list)
    calls: list[dict[str, str]] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    kind: str = ""
    tools_intact: bool | None = None
    error: str | None = None


def _stream_class() -> Any:
    # the shared engine, or the openai plugin's stream on releases that predate it
    try:
        from livekit.agents.llm import _responses_engine

        return _responses_engine.ResponsesLLMStream
    except ImportError:
        from livekit.plugins.openai.responses import llm as openai_responses

        return openai_responses.LLMStream


_streams: dict[int, list[Any]] = {}
_STREAM_CLASS = _stream_class()
_original_process_event = _STREAM_CLASS._process_event


def _tee_process_event(self: Any, event: Any) -> Any:
    _streams.setdefault(id(self), []).append(event)
    return _original_process_event(self, event)


_STREAM_CLASS._process_event = _tee_process_event


def _message_text(item: Any) -> str:
    return "".join(
        getattr(part, "text", "") for part in (getattr(item, "content", None) or [])
    ).strip()


def _summarize_items(events: list[Any]) -> list[dict[str, Any]]:
    for event in reversed(events):
        if getattr(event, "type", "") == "response.completed":
            output = event.response.output
            break
    else:
        output = [e.item for e in events if getattr(e, "type", "") == "response.output_item.done"]
    items = []
    for item in output:
        entry: dict[str, Any] = {"type": item.type, "id": getattr(item, "id", None)}
        if item.type == "message":
            entry["phase"] = getattr(item, "phase", None)
            entry["text"] = _message_text(item)
        elif item.type == "function_call":
            entry["name"] = item.name
            entry["call_id"] = item.call_id
        items.append(entry)
    return items


def classify(items: list[dict[str, Any]]) -> str:
    """Name the reply's shape from its raw output items."""
    texts = [i for i in items if i["type"] == "message" and i.get("text")]
    if len(texts) <= 1:
        return "single"
    kind = "commentary_chain"
    seen_text = False
    hosted_since_text = False
    previous_phase: str | None = None
    for item in items:
        if item["type"] == "message" and item.get("text"):
            if seen_text and not hosted_since_text and previous_phase != "commentary":
                return "multi_message"
            if seen_text and hosted_since_text:
                kind = "hosted_continuation"
            seen_text, hosted_since_text, previous_phase = True, False, item.get("phase")
        elif item["type"] not in ("message", "function_call", "reasoning"):
            hosted_since_text = seen_text
    return kind


SCHEDULER_FOLLOW_UPS = [
    "Yes, please book that.",
    "Actually, can you make it a later time?",
    "Great. What's the clinic's address?",
    "Thanks, that's everything.",
]
TOOL_RESULTS = {
    "lookup_patient": '{"patient_id": "p_1042", "name": "Sam Lee"}',
    "check_availability": '{"open": ["09:00", "11:30", "14:00"]}',
    "book_appointment": '{"confirmed": true}',
    "get_clinic_info": '{"address": "12 Harbor Street, Suite 3"}',
}
# a model that keeps calling tools without speaking must not loop forever
MAX_TOOL_ROUNDS = 3


def _opening(target: Target) -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    if target.tools:
        ctx.add_message(role="system", content=SCHEDULER_SYSTEM)
        return ctx
    system, history, _ = PROMPTS[_prompt_name]
    ctx.add_message(role="system", content=system)
    # the agent is the assistant, so the other party's lines arrive as user turns
    for role, text in history:
        ctx.add_message(role=role, content=text)  # type: ignore[arg-type]
    return ctx


def _caller_line(target: Target, index: int, turn: int) -> str:
    if target.tools:
        if turn == 0:
            return SCHEDULER_REQUESTS[index % len(SCHEDULER_REQUESTS)]
        return SCHEDULER_FOLLOW_UPS[(turn - 1) % len(SCHEDULER_FOLLOW_UPS)]
    questions = PROMPTS[_prompt_name][2]
    return questions[(index + turn) % len(questions)]


async def _run_turn(
    model: llm.LLM, target: Target, ctx: llm.ChatContext, index: int, turn: int, prompt: str
) -> Reply:
    reply = Reply(target=target.name, index=index, turn=turn, prompt=prompt)
    stream = model.chat(chat_ctx=ctx, tools=SCHEDULER_TOOLS if target.tools else [])
    try:
        async with stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    reply.spoken += chunk.delta.content
                for call in (chunk.delta.tool_calls if chunk.delta else None) or []:
                    reply.tool_calls.append(call.call_id)
                    reply.calls.append(
                        {"call_id": call.call_id, "name": call.name, "arguments": call.arguments}
                    )
    except Exception as e:  # recorded, not fatal: one bad reply must not stop the run
        reply.error = f"{type(e).__name__}: {e}"
    reply.items = _summarize_items(_streams.pop(id(stream), []))
    reply.dropped = list(getattr(stream, "_dropped_item_ids", []))
    reply.kind = "error" if reply.error else classify(reply.items)
    raw_calls = [i["call_id"] for i in reply.items if i["type"] == "function_call"]
    if target.tools and not reply.error:
        reply.tools_intact = raw_calls == reply.tool_calls
    return reply


def _append_reply(ctx: llm.ChatContext, reply: Reply) -> None:
    """Add what the caller heard and the calls it ran, as an agent session would."""
    if reply.spoken:
        ctx.add_message(role="assistant", content=reply.spoken)
    for call in reply.calls:
        ctx.items.append(llm.FunctionCall(**call))
        ctx.items.append(
            llm.FunctionCallOutput(
                call_id=call["call_id"],
                name=call["name"],
                output=TOOL_RESULTS.get(call["name"], '{"ok": true}'),
                is_error=False,
            )
        )


async def run_conversation(
    target: Target,
    index: int,
    turns: int,
    store: bool | None = None,
    temperature: float | None = None,
    chain: bool = True,
) -> list[Reply]:
    """Run one call of `turns` caller lines, feeding every reply back into the history.

    Each conversation gets its own LLM so the plugin chains `previous_response_id` the
    way a live session does. A turn that calls tools gets canned results, and the reply
    after them is recorded as its own turn.
    """
    model = target.make()
    _apply_overrides(model, store, temperature)
    ctx = _opening(target)
    replies: list[Reply] = []
    try:
        for turn in range(turns):
            prompt = _caller_line(target, index, turn)
            ctx.add_message(role="user", content=prompt)
            for _ in range(MAX_TOOL_ROUNDS + 1):
                if not chain:
                    # a ZDR organization refuses previous_response_id, and the gateway
                    # bridge never chains; send the whole history instead
                    model._prev_resp_id = ""  # type: ignore[attr-defined]
                reply = await _run_turn(model, target, ctx, index, len(replies), prompt)
                replies.append(reply)
                if reply.error:
                    return replies
                _append_reply(ctx, reply)
                if not reply.calls:
                    break
                prompt = "<tool results>"
    finally:
        await model.aclose()
    return replies


def _apply_overrides(model: llm.LLM, store: bool | None, temperature: float | None) -> None:
    # read from the options on every request, so they apply to any plugin, including
    # azure.responses.LLM, which takes no `store` argument
    opts = model._opts  # type: ignore[attr-defined]
    if store is not None:
        opts.store = store
    if temperature is not None:
        opts.temperature = temperature


async def run_target(
    target: Target,
    n: int,
    concurrency: int,
    turns: int,
    store: bool | None = None,
    temperature: float | None = None,
    chain: bool = True,
) -> list[Reply]:
    sem = asyncio.Semaphore(concurrency)

    async def bounded(i: int) -> list[Reply]:
        async with sem:
            return await run_conversation(target, i, turns, store, temperature, chain)

    conversations = await asyncio.gather(*(bounded(i) for i in range(n)))
    return [reply for conversation in conversations for reply in conversation]


def _print_summary(results: dict[str, list[Reply]]) -> None:
    kinds = ["single", "commentary_chain", "hosted_continuation", "multi_message", "error"]
    print("\n| target | replies | " + " | ".join(kinds) + " | dropped | tools intact |")
    print("|---|---|" + "---|" * (len(kinds) + 2))
    for name, replies in results.items():
        counts = Counter(r.kind for r in replies)
        dropped = sum(1 for r in replies if r.dropped)
        tool_replies = [r for r in replies if r.tools_intact is not None]
        intact = (
            f"{sum(r.tools_intact for r in tool_replies)}/{len(tool_replies)}"
            if tool_replies
            else "-"
        )
        cells = " | ".join(str(counts.get(k, 0)) for k in kinds)
        print(f"| {name} | {len(replies)} | {cells} | {dropped} | {intact} |")
    for name, replies in results.items():
        for r in replies:
            if r.kind == "multi_message":
                texts = [i.get("text") for i in r.items if i["type"] == "message" and i.get("text")]
                print(f"\n[{name} #{r.index} turn {r.turn}] {r.prompt}")
                for j, t in enumerate(texts, 1):
                    print(f"  m{j}: {t}")


JUDGE_INSTRUCTIONS = """You review replies a voice agent produced. Each reply came back as \
several separate messages. Decide whether the later messages are ALTERNATES (they \
restate or re-answer what an earlier message already said, so speaking them repeats \
the agent) or a CONTINUATION (they add new content, such as a question or a next step, \
that the caller would miss if only the first message were spoken). Answer with one \
word: alternate or continuation."""


async def judge_file(path: str, model: str = "gpt-5.4") -> None:
    """Label every multi-message reply in a probe JSONL, writing `<path>.judged.jsonl`."""
    import openai

    client = openai.AsyncOpenAI()
    rows = [json.loads(line) for line in open(path)]
    sem = asyncio.Semaphore(8)

    async def judge(row: dict[str, Any]) -> None:
        texts = [i["text"] for i in row["items"] if i["type"] == "message" and i.get("text")]
        listing = "\n".join(f"message {n}: {t}" for n, t in enumerate(texts, 1))
        async with sem:
            response = await client.responses.create(
                model=model,
                instructions=JUDGE_INSTRUCTIONS,
                input=f"Caller said: {row['prompt']}\n{listing}",
                reasoning={"effort": "low"},
                store=False,
            )
        verdict = response.output_text.strip().lower()
        row["verdict"] = verdict if verdict in ("alternate", "continuation") else "unclear"

    multi = [r for r in rows if r["kind"] == "multi_message"]
    await asyncio.gather(*(judge(r) for r in multi))
    with open(path + ".judged.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    counts = Counter((r["target"], r["verdict"]) for r in multi)
    for (target, verdict), count in sorted(counts.items()):
        print(f"{target}\t{verdict}\t{count}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=100, help="conversations per target")
    parser.add_argument("--turns", type=int, default=4, help="caller lines per conversation")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--targets", default="", help="comma-separated target names")
    parser.add_argument("--out", default="", help="write every reply as JSONL")
    parser.add_argument(
        "--azure",
        action="append",
        default=[],
        metavar="NAME=BASE_URL,KEY_ENV,DEPLOYMENT[,tools]",
        help="probe an Azure deployment by name; repeatable, replaces the built-in targets",
    )
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="claimant")
    parser.add_argument(
        "--no-chain",
        action="store_true",
        help="send the full history every turn instead of previous_response_id",
    )
    parser.add_argument("--store", choices=["true", "false"], help="send `store` on every request")
    parser.add_argument("--temperature", type=float, help="send `temperature` on every request")
    parser.add_argument(
        "--judge",
        metavar="JSONL",
        help="label the multi-message replies in a probe JSONL as alternate or continuation",
    )
    args = parser.parse_args()
    if args.judge:
        await judge_file(args.judge)
        return
    store = None if args.store is None else args.store == "true"
    global _prompt_name
    _prompt_name = args.prompt

    wanted = {t for t in args.targets.split(",") if t}
    targets = [_azure_target(spec) for spec in args.azure] or TARGETS
    results: dict[str, list[Reply]] = {}
    for target in targets:
        if wanted and target.name not in wanted:
            continue
        missing = [k for k in target.key_env if not os.environ.get(k)]
        if missing:
            print(f"skip {target.name}: {', '.join(missing)} unset", file=sys.stderr)
            continue
        print(f"running {target.name} x{args.n}", file=sys.stderr)
        results[target.name] = await run_target(
            target, args.n, args.concurrency, args.turns, store, args.temperature, not args.no_chain
        )

    if args.out:
        with open(args.out, "w") as f:
            for replies in results.values():
                for r in replies:
                    f.write(json.dumps(asdict(r)) + "\n")
    _print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
