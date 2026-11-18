"""Scripted models for `examples/delegation-voice`.

Three roles, told apart by the tools each request offers: the routing model (`route_*`), the
backend (`prepare_job`, `submit_job`, `look_up`) and the frontend (the backend's remote tool).
Each answers from its request's content, as a model would: the person's words, the
framework's notes and instructions, and its tools' outputs. Anything it was not written for
raises `Unexpected`, which the provider journals and the test fails on.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeGuard

from livekit.agents.llm import ChatContext, ChatMessage, FunctionCall, FunctionToolCall
from tests.fakes import Hang, tool_call

from ..framework.providers import Request, Unexpected

EXPORT, BACKUP, AUDIT = "the weekly export", "the nightly backup", "the audit"
JOBS = (EXPORT, BACKUP)
GREETING = "Hi, how can I help?"
ACKNOWLEDGEMENT = "On it."
SIDE_ANSWER = "It is noon."
CLARIFY = "Which job, the weekly export or the nightly backup?"
NOT_SUBMITTED = "Not submitted."
CANNOT_CANCEL = "I can't cancel it."
WORKING = "Working on it."
STALL = "hold on"  # the frontend is still thinking about this when the person goes on

# Each phrase is its own tone, 200 Hz apart; the agent speaks at 1 kHz.
PHRASES = {
    300: "run the weekly export",
    500: "run the nightly backup",
    700: "look up the audit",
    900: "what time is it",
    1300: "high",
    1500: "normal",
    1700: "yes",
    1900: "no",
    2100: "actually normal",
    2300: "Q3",
    2500: "high and Q3",
    2700: "high for the weekly export",
    2900: "cancel the weekly export",
    3100: "high hold on",
    3300: "look up the audit for Q3",
}

# The framework's turn instructions and notes the models answer.
PROGRESS = "Tell the person how their request is going"
ASK = "Ask the person this, in your own words"
OUTCOME = "Present this remote tool outcome to the person"
RESUME = "Continue the recognized human responses reserved for your remote tools"
RESPONDS = "This human message responds to these presented input requests"
NOT_RECOGNIZED = "This message was not recognized as a response to a presented input request"
GREET = "Greet the user and offer to help."
APPROVAL = re.compile(r"the person answered '(.*)' to its read-back")

# The example's tool outputs.
READY = " is ready at "
PROGRESS_OUTPUTS = ("Reserving capacity for it.", "Submitting the job.")
PROGRESS_PREFIXES = ("Checking what ", "Checking capacity at ", "Searching the ")
WAITING = "waiting for the person's answer"


def reply(request: Request) -> Any:
    if {"route_input_response", "route_remote_call"} & set(request.tools):
        return route(request)
    if "prepare_job" in request.tools:
        return backend(request.chat_ctx)
    # The frontend's remote tool, beside the framework's cancel_<tool> sibling.
    remote = [tool for tool in request.tools if not tool.startswith("cancel_")]
    if len(remote) == 1:
        return frontend(request.chat_ctx, remote[0])
    raise Unexpected(f"no role offers the tools {request.tools}")


# -- what the person said


def priority_in(text: str) -> str | None:
    words = re.findall(r"[a-z]+", text.lower())
    if "normal" in words:
        return "normal"
    return "high" if "high" in words else None


def quarter_in(text: str) -> str | None:
    match = re.search(r"\bQ[1-4]\b", text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def confirmation_in(text: str) -> str | None:
    word = text.lower().strip(" .!")
    return word if word in ("yes", "no") else None


def answers(text: str) -> bool:
    return bool(priority_in(text) or quarter_in(text) or confirmation_in(text))


def job_in(text: str) -> str | None:
    return next((job for job in JOBS if job.split()[-1] in text.lower()), None)


def is_human(item: object) -> TypeGuard[ChatMessage]:
    return isinstance(item, ChatMessage) and item.role == "user"


def latest_human(history: ChatContext) -> str:
    return next((item.text_content or "" for item in reversed(history.items) if is_human(item)), "")


def last_instruction(history: ChatContext) -> str:
    """The turn instruction the framework appended for this reply, if any."""
    last = history.items[-1]
    if last.type == "message" and last.role in ("system", "developer"):
        return last.text_content or ""
    return ""


def call_of(history: ChatContext, call_id: str) -> FunctionCall:
    return next(
        item for item in history.items if item.type == "function_call" and item.call_id == call_id
    )


def arguments(call: FunctionCall) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(call.arguments or "{}")
    return parsed


def calls_named(history: ChatContext, name: str) -> list[FunctionCall]:
    return [item for item in history.items if item.type == "function_call" and item.name == name]


# -- the backend


def backend(history: ChatContext) -> str | FunctionToolCall | list[FunctionToolCall]:
    if (outcome := output_to_act_on(history)) is not None:
        return outcome
    if (approval := approval_note(history)) is not None:
        answer, call = approval
        if confirmation_in(answer) == "yes":
            return tool_call("submit_job", **arguments(call))
        if (priority := priority_in(answer)) is not None:
            # A correction instead of a yes: prepare it again at the new priority.
            return tool_call("prepare_job", **{**arguments(call), "priority": priority})
        return NOT_SUBMITTED
    text = latest_human(history)
    if answered := answered_calls(history):
        calls = []
        for call in answered:
            given = arguments(call)
            if call.name == "prepare_job" and (priority := priority_in(text)) is not None:
                calls.append(tool_call("prepare_job", **{**given, "priority": priority}))
            elif call.name == "look_up" and (quarter := quarter_in(text)) is not None:
                calls.append(tool_call("look_up", **{**given, "quarter": quarter}))
        if calls:
            return calls if len(calls) > 1 else calls[0]
    if text.startswith("run "):
        return tool_call("prepare_job", description=text.removeprefix("run "))
    if text.startswith("look up "):
        # A batched turn answers every request merged into it.
        lookups = []
        for request in humans_since_reply(history):
            topic, _, quarter = request.removeprefix("look up ").partition(" for ")
            lookups.append(tool_call("look_up", topic=topic, quarter=quarter or None))
        return lookups if len(lookups) > 1 else lookups[0]
    if text.startswith("cancel "):
        return CANNOT_CANCEL
    if (priority := priority_in(text)) is not None and (job := job_in(text)) is not None:
        # A message about one job that answers no presented question still says its priority.
        return tool_call("prepare_job", description=job, priority=priority)
    if priority is not None and text.startswith("actually "):
        # A correction to the job last prepared: prepare it again, which reads it back again.
        last = arguments(calls_named(history, "prepare_job")[-1])
        return tool_call("prepare_job", description=last["description"], priority=priority)
    if answers(text):
        return WORKING  # an answer meant for another task's question: nothing to do here
    raise Unexpected(f"backend has no reply to {text!r}")


def humans_since_reply(history: ChatContext) -> list[str]:
    humans: list[str] = []
    for item in reversed(history.items):
        if item.type == "message" and item.role == "assistant":
            break
        if is_human(item):
            # A batched turn's merged messages arrive as one, a line each.
            humans[:0] = (item.text_content or "").splitlines()
    return humans


def output_to_act_on(history: ChatContext) -> str | FunctionToolCall | None:
    """What the tool outputs since the last human message ask for, if not yet done.

    A live backend's tool releases the floor with its progress, and its final output replaces
    that progress in place: the output to act on is not always the last item."""
    items = history.items
    finals: list[str] = []
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if is_human(item):
            break
        if item.type != "function_call_output":
            continue
        call = call_of(history, item.call_id)
        later = items[index + 1 :]
        if call.name == "prepare_job" and READY in item.output:
            if not any(i.type == "function_call" and i.name == "submit_job" for i in later):
                return tool_call("submit_job", **arguments(call))
        elif (final := final_reply(call, item.output)) is not None:
            if not any(i.type == "message" and final in (i.text_content or "") for i in later):
                finals.insert(0, final)
        elif item.output in PROGRESS_OUTPUTS or item.output.startswith(PROGRESS_PREFIXES):
            if index == len(items) - 1:
                return WORKING  # the tool released the floor; its work goes on
        elif not item.output.startswith(WAITING):
            raise Unexpected(f"backend has no reply to {call.name} output {item.output!r}")
    return " ".join(finals) if finals else None


def final_reply(call: FunctionCall, output: str) -> str | None:
    if call.name == "submit_job" and output.startswith("submitted "):
        done = arguments(call)
        return f"'{done['description']}' is submitted at {done['priority']} priority."
    if call.name == "look_up" and " documents about " in output:
        return output
    return None


def approval_note(history: ChatContext) -> tuple[str, FunctionCall] | None:
    """The framework's note that the person answered a read-back, and the call it gates."""
    for item in reversed(history.items):
        if item.type == "message" and item.role == "assistant":
            return None
        if item.type == "message" and item.role == "system":
            match = APPROVAL.search(item.text_content or "")
            if match is not None:
                return match.group(1), calls_named(history, "submit_job")[-1]
    return None


def answered_calls(history: ChatContext) -> list[FunctionCall]:
    """The tool calls whose questions the framework's note says this human message answers."""
    for item in reversed(history.items):
        if item.type == "message" and item.role == "assistant":
            return []
        if item.type == "message" and (item.text_content or "").startswith(RESPONDS):
            text = item.text_content or ""
            return [
                call
                for call in history.items
                if call.type == "function_call" and call.id.split("/")[-1] in text
            ]
    return []


# -- the frontend


def frontend(history: ChatContext, remote: str) -> str | FunctionToolCall | list[Any] | Hang:
    said = last_instruction(history)
    if said.endswith(GREET):
        return GREETING
    if said.startswith(PROGRESS):
        return "Update: " + said.split(": ", 1)[1]
    if said.startswith(ASK):
        return said.split(": ", 1)[1]
    if said.startswith(OUTCOME):
        return str(json.loads(said.split(": ", 1)[1])["output"])
    if said.startswith((RESUME, RESPONDS)):
        if latest_human(history).endswith(STALL):
            return Hang()  # still thinking when the person corrects themselves
        return forward(remote, latest_human(history)) or ACKNOWLEDGEMENT
    if said and not said.startswith(NOT_RECOGNIZED):
        raise Unexpected(f"frontend has no reply to the instruction {said!r}")
    if history.items[-1].type == "function_call_output":
        return ACKNOWLEDGEMENT
    text = latest_human(history)
    if text == "what time is it":
        return SIDE_ANSWER
    if text.endswith(STALL):
        return Hang()  # still thinking when the person corrects themselves
    if text.startswith(("run ", "look up ", "cancel ")):
        return tool_call(remote, request=text)
    if answers(text):
        # No question recognized as answered. A bare priority with two jobs asking is ambiguous.
        asked = [
            item
            for item in history.items
            if item.type == "message"
            and item.role == "assistant"
            and "normal or high priority" in (item.text_content or "")
        ]
        if priority_in(text) and job_in(text) is None and len(asked) > 1:
            return CLARIFY
        return forward(remote, text) or ACKNOWLEDGEMENT
    raise Unexpected(f"frontend has no reply to {text!r}")


def forward(remote: str, text: str) -> list[FunctionToolCall] | None:
    """One call per question the words answer, each forwarding the person's words."""
    kinds = [bool(priority_in(text) or confirmation_in(text)), bool(quarter_in(text))]
    calls = [tool_call(remote, request=text) for present in kinds if present]
    return calls or None


# -- the routing model


def route(request: Request) -> FunctionToolCall:
    message = request.chat_ctx.items[-1]
    if message.type != "message" or message.text_content is None:
        raise Unexpected("a routing request ends with its prompt")
    prompt = json.loads(message.text_content)
    if "route_remote_call" in request.tools:
        return route_call(prompt)
    return route_input(prompt)


def route_call(prompt: dict[str, Any]) -> FunctionToolCall:
    proposed = prompt["proposed_call"].get("request", "")
    for candidate in prompt["candidates"]:
        for response in candidate["responses"]:
            if response["held_response"] == proposed:
                return tool_call(
                    "route_remote_call",
                    action="respond",
                    candidate=candidate["index"],
                    response=response["index"],
                )
    if answers(proposed) or proposed.startswith("cancel "):
        # Words about existing work go to the one task they can be about, else unclear.
        job = job_in(proposed)
        about = [
            candidate
            for candidate in prompt["candidates"]
            if job is None or any(job in (m.get("text") or "") for m in candidate["messages"])
        ]
        if len(about) == 1:
            return tool_call(
                "route_remote_call", action="continue", candidate=about[0]["index"], response=None
            )
        return tool_call("route_remote_call", action="unclear", candidate=None, response=None)
    return tool_call("route_remote_call", action="new", candidate=None, response=None)


def route_input(prompt: dict[str, Any]) -> FunctionToolCall:
    text = prompt["latest_message"]
    picked: list[int] = []
    for kind, present in (
        ("priority", priority_in(text) is not None),
        ("quarter", quarter_in(text) is not None),
        ("confirm", confirmation_in(text) is not None),
    ):
        if not present:
            continue
        of_kind = [c for c in prompt["candidates"] if kind_of(c["question"]) == kind]
        if (job := job_in(text)) is not None:
            of_kind = [c for c in of_kind if job in c["presentation"]]
        if kind == "confirm" and len(of_kind) > 1:
            # A yes or no answers the read-back the person just heard, when it was the last.
            said = [m["text"] for m in prompt["history"] if m["role"] == "Assistant"]
            of_kind = [c for c in of_kind if said and c["presentation"] == said[-1]] or of_kind
        if len(of_kind) > 1:
            return tool_call("route_input_response", candidates=[], unclear=True)
        picked += [c["index"] for c in of_kind]
    return tool_call("route_input_response", candidates=picked, unclear=False)


def kind_of(question: str) -> str:
    if "normal or high priority" in question:
        return "priority"
    if question.startswith("Which quarter"):
        return "quarter"
    if question.startswith("Submit "):
        return "confirm"
    raise Unexpected(f"no kind for the question {question!r}")
