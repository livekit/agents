"""Scripted delivery correctness through real processes, two A2A hops and received room audio."""

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from livekit.agents.a2a import PROGRESS
from tests.fakes import until

from .apps.tool_delivery import (
    ACKNOWLEDGEMENT,
    EXACT,
    FINAL,
    PHRASES,
    REQUEST,
    RESULT,
    SILENT_STATUS,
)
from .framework.app import E2E
from .resources import RecordingService


def report(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "live-model-report.json").read_text())


def completed_voice_report(directory: Path) -> bool:
    if not (directory / "live-model-report.json").exists():
        return False
    snapshot = report(directory)
    return snapshot["observations"]["agent_state"] == "listening" and any(
        message["text"] == FINAL and not message["interrupted"]
        for message in snapshot["conversation"]
        if message["role"] == "assistant"
    )


def only_remote(snapshot: dict[str, Any], name: str) -> dict[str, Any]:
    assert set(snapshot["remote_agents"]) == {name}
    remote: dict[str, Any] = snapshot["remote_agents"][name]
    [task] = remote["tasks"].values()
    assert task["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    sends = [
        event["payload"]["message"]
        for event in remote["events"]
        if event["kind"] == "request_recorded" and "message" in event["payload"]
    ]
    assert len(sends) == 1, "progress and exact output must not resend business input"
    request, history = sends[0]["parts"]
    assert request == {"text": REQUEST}
    assert history["metadata"] == {"https://livekit.io/a2a/ext/agents/v1/kind": "chat_ctx"}
    assert set(history["data"]) == {"items"}
    return remote


async def test_unawaited_exact_output_crosses_two_remote_hops_and_is_heard_once(
    e2e: E2E,
    service: RecordingService,
) -> None:
    journey = "delivery-" + uuid.uuid4().hex
    directories = {role: e2e.directory / role for role in ("service", "coordinator", "voice")}

    def environment(role: str) -> dict[str, str]:
        return {
            "E2E_DELIVERY_ROLE": role,
            "E2E_ARTIFACT_DIR": str(directories[role]),
            "E2E_DELIVERY_ID": journey,
            "E2E_SERVICE_URL": service.url,
        }

    leaf = await e2e.start_app("tests.e2e.apps.tool_delivery", env=environment("service"))
    middle = await e2e.start_app(
        "tests.e2e.apps.tool_delivery",
        env={**environment("coordinator"), "E2E_REMOTE_URL": leaf.endpoint("booking")},
    )
    front = await e2e.start_app(
        "tests.e2e.apps.tool_delivery",
        voice=True,
        env={**environment("voice"), "E2E_REMOTE_URL": middle.endpoint("booking")},
    )
    try:
        async with e2e.voice_client(front, phrases=PHRASES) as caller:
            await caller.say(REQUEST)
            await service.wait_for_call(journey, "wait")
            acknowledgement = await caller.hear()
            assert acknowledgement.text == ACKNOWLEDGEMENT
            assert [call.operation for call in service.calls] == ["wait"]
            service.release(journey)

            exact = await caller.hear()
            final = await caller.hear()
            received = [acknowledgement, exact, final]
            (e2e.directory / "received-speech.json").write_text(
                json.dumps([asdict(speech) for speech in received], indent=2) + "\n"
            )
            assert [speech.text for speech in received] == [ACKNOWLEDGEMENT, EXACT, FINAL]
            assert all(speech.audio_seconds > 0 for speech in received)
            assert len({speech.segment_id for speech in received}) == 3
            await until(
                lambda: completed_voice_report(directories["voice"]),
                "the final reservation reply reached the native listening boundary",
                timeout=15,
            )

            front_state = report(directories["voice"])
            perceived = [
                message["text"]
                for message in front_state["conversation"]
                if message["role"] == "assistant" and not message["interrupted"]
            ]
            assert perceived == [ACKNOWLEDGEMENT, EXACT, FINAL]
            assert SILENT_STATUS not in perceived
            assert front_state["observations"]["model_requests"] == 3
            assert front_state["observations"]["routing_requests"] == 0
            only_remote(front_state, "travel")

        middle_state = report(directories["coordinator"])
        remote = only_remote(middle_state, "fulfilment")
        statuses = [
            event["payload"]["statusUpdate"]["status"]
            for event in remote["events"]
            if event["kind"] == "response_received" and "statusUpdate" in event["payload"]
        ]
        silent = [
            status
            for status in statuses
            if status.get("message", {}).get("parts") == [{"text": SILENT_STATUS}]
        ]
        assert len(silent) == 1, "silent progress is observable SDK status, not dropped text"
        assert silent[0]["state"] == "TASK_STATE_WORKING"
        assert silent[0]["message"].get("metadata", {}).get(PROGRESS) is not True
        exact_status = [
            status
            for status in statuses
            if status.get("message", {}).get("parts") == [{"text": EXACT}]
        ]
        assert len(exact_status) == 1 and exact_status[0]["state"] == "TASK_STATE_WORKING"

        for role in ("service", "coordinator"):
            observations = report(directories[role])["observations"]
            assert observations["result"] == RESULT
            assert observations["model_requests"] == 2
            assert observations["routing_requests"] == 0
            assert [item["output"] for item in observations["tool_results"]] == [RESULT]
        assert [call.operation for call in service.calls] == ["wait", "book"]
        [effect] = [call for call in service.calls if call.operation == "book"]
        assert effect.thread == journey and effect.terms == "R-204"
    finally:
        (e2e.directory / "service-ledger.json").write_text(
            json.dumps([call.model_dump() for call in service.calls], indent=2) + "\n"
        )
        service.release(journey)
