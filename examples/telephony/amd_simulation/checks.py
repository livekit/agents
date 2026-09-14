from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .scenarios import Scenario

MAX_EXTRA_UNCERTAIN = 2
MAX_EXTRA_WAIT = 5.0
MACHINE_SILENCE = 1.5
TIMING_TOLERANCE = 0.15


@dataclass
class Event:
    time: float
    kind: str
    step: int
    data: dict[str, Any] = field(default_factory=dict)


def extra_uncertain(scenario: Scenario, events: list[Event]) -> int:
    return len(
        {
            e.data["turn_id"]
            for e in events
            if e.kind in ("classification", "prediction")
            and e.data["category"] == "uncertain"
            and 0 <= e.step < len(scenario.steps)
            and scenario.steps[e.step].category != "uncertain"
        }
    )


def turnaround(scenario: Scenario, events: list[Event]) -> float:
    total = 0.0
    for i, step in enumerate(scenario.steps):
        current = [e for e in events if e.step == i]
        ends = [e.time for e in current if e.kind == "input_end"]
        if not ends:
            continue
        predictions = [
            e.time
            for e in current
            if e.kind == "prediction" and e.data["category"] == step.category
        ]
        actions = [
            e.time
            for e in current
            if (e.kind == "audio" and step.reply) or (e.kind == "dtmf" and step.dtmf)
        ]
        deadline = max((e.time for e in current if e.kind == "action_timeout"), default=ends[-1])
        ready = max(min(predictions, default=ends[-1]), min(actions, default=deadline))
        total += max(0.0, ready - ends[-1])
    return total


def extra_wait(scenario: Scenario, events: list[Event], *, now: float | None = None) -> float:
    """Accumulate unresolved uncertain/wait decisions, excluding scripted time and playback."""
    horizon = now if now is not None else max((e.time for e in events), default=0.0)
    waiting: list[tuple[float, float]] = []
    excluded: list[tuple[float, float]] = []
    started: float | None = None
    active: dict[str, float] = {}
    for event in sorted(events, key=lambda e: e.time):
        if event.kind == "prediction":
            expected = (
                scenario.steps[event.step].category
                if 0 <= event.step < len(scenario.steps)
                else None
            )
            unresolved = event.data.get("should_wait", False) or (
                event.data["category"] == "uncertain" and expected != "uncertain"
            )
            if unresolved and started is None:
                started = event.time
            elif not unresolved and started is not None:
                waiting.append((started, event.time))
                started = None
        elif event.kind in ("completed", "run_end") and started is not None:
            waiting.append((started, event.time))
            started = None
        for beginning, ending in (
            ("input_start", "clip_end"),
            ("speech_start", "speech_end"),
            ("pause_start", "pause_end"),
        ):
            if event.kind == beginning:
                active[beginning] = event.time
            elif event.kind == ending and beginning in active:
                excluded.append((active.pop(beginning), event.time))
        if event.kind == "input_end" and 0 <= event.step < len(scenario.steps):
            if (scenario.steps[event.step].category or "").startswith("machine-"):
                excluded.append((event.time, event.time + MACHINE_SILENCE))
    if started is not None:
        waiting.append((started, horizon))
    excluded.extend((start, horizon) for start in active.values())
    # Union the exclusions while subtracting them; overlapping speech and input count once.
    total = 0.0
    for start, end in waiting:
        cursor = start
        for lower, upper in sorted(excluded):
            if upper <= cursor or lower >= end:
                continue
            total += max(0.0, lower - cursor)
            cursor = max(cursor, min(upper, end))
        total += max(0.0, end - cursor)
    return total


def check(scenario: Scenario, events: list[Event]) -> list[str]:
    errors: list[str] = []
    completed = [e for e in events if e.kind == "completed"]
    if len(completed) != 1:
        errors.append(f"Expected one completion, received {len(completed)}")
    elif (completed[0].data["category"], completed[0].data["reason"]) != (
        scenario.category,
        scenario.reason,
    ):
        errors.append(f"Wrong completion: {completed[0].data}")
    elif scenario.voicemail_played is not None and (
        completed[0].data["voicemail_message_played"] != scenario.voicemail_played
    ):
        errors.append(f"Expected voicemail_message_played={scenario.voicemail_played}")

    for i, step in enumerate(scenario.steps):
        current = [e for e in events if e.step == i]
        prefix = f"{step.name}: "
        ends = [e.time for e in current if e.kind == "input_end"]
        if not ends:
            errors.append(prefix + "script step did not finish playing")
            continue
        predictions = [e for e in current if e.kind == "prediction"]
        inconclusive = {
            e.data["turn_id"]
            for e in current
            if e.kind == "classification" and e.data["category"] == "uncertain"
        }
        if step.category is None:
            if predictions:
                errors.append(prefix + "AMD emitted predictions after human completion")
        elif not any(e.data["category"] == step.category for e in predictions):
            errors.append(prefix + f"missing expected category {step.category}")
        for e in predictions:
            if e.data["reason"] in ("prediction", "late_prediction") and not any(
                raw.kind == "classification" and raw.data["turn_id"] == e.data["turn_id"]
                for raw in events
            ):
                errors.append(prefix + "missing raw classification evidence for uncertainty budget")
            if e.data["reason"] == "inference_error":
                errors.append(prefix + "classification failed")
            if e.data["category"] not in (step.category, "uncertain") and not (
                e.data["turn_id"] in inconclusive or e.data["reason"] == "inference_timeout"
            ):
                errors.append(prefix + f"unexpected stage {e.data['category']}")

        replies = [e for e in current if e.kind == "reply" and e.data["text"].strip()]
        if len(replies) != bool(step.reply):
            errors.append(prefix + f"expected {int(bool(step.reply))} reply, got {len(replies)}")
        audio = [e for e in current if e.kind == "audio"]
        if step.reply and not audio:
            errors.append(prefix + "no reply audio reached the callee")
        if not step.reply and audio:
            errors.append(prefix + "unexpected spoken reply")
        digits = "".join(e.data["digit"] for e in current if e.kind == "dtmf")
        if digits != step.dtmf:
            errors.append(prefix + f"expected DTMF {step.dtmf!r}, received {digits!r}")
        if step.advance_on_start and not any(e.data["interrupted"] for e in replies):
            errors.append(prefix + "the scripted human pickup did not interrupt voicemail")

        baseline = MACHINE_SILENCE if (step.category or "").startswith("machine-") else 0.0
        actions = [e for e in current if e.kind in ("audio", "dtmf")]
        machine_predictions = [e for e in predictions if e.data["category"].startswith("machine-")]
        for e in actions + machine_predictions:
            if e.time < ends[-1] + baseline - TIMING_TOLERANCE:
                errors.append(prefix + f"premature {e.kind}")
        if (
            step.category is not None
            and actions
            and not any(e.time <= actions[0].time for e in predictions)
        ):
            errors.append(prefix + "action started before an AMD prediction")

    unexpected = [e for e in events if e.step < 0 and e.kind in ("audio", "dtmf", "reply")]
    if unexpected:
        errors.append("Agent acted before the callee's first turn")
    if completed and any(e.kind == "prediction" and e.time > completed[0].time for e in events):
        errors.append("AMD predicted after completion")
    uncertain = extra_uncertain(scenario, events)
    if uncertain > MAX_EXTRA_UNCERTAIN:
        errors.append(f"Call uncertainty budget exceeded: {uncertain} > {MAX_EXTRA_UNCERTAIN}")
    waiting = extra_wait(scenario, events)
    if waiting > MAX_EXTRA_WAIT:
        errors.append(f"Call waiting budget exceeded: {waiting:.2f}s > {MAX_EXTRA_WAIT:.2f}s")
    listening = next((e.time for e in events if e.kind == "listening"), None)
    if completed and listening is not None and scenario.reason == "timeout":
        elapsed = completed[0].time - listening
        if abs(elapsed - scenario.timeout) > 1.0:
            errors.append(f"Overall deadline moved: completed at {elapsed:.2f}s")
    if completed and listening is not None and scenario.reason == "idle_timeout":
        anchor = max(
            (e.time for e in events if e.kind in ("input_end", "speech_end", "dtmf")),
            default=listening,
        )
        idle = 60.0 if scenario.category == "machine-vm" else scenario.idle_timeout
        if not idle - 1.0 <= completed[0].time - anchor <= idle + 5.0:
            errors.append("Idle completion did not respect the stage's idle deadline")
    return errors
