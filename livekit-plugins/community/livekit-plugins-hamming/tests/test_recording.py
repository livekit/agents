from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from livekit.plugins.hamming._plugin import (
    RECORDING_MODE_NONE,
    RECORDING_MODE_SESSION_AUDIO,
    HammingRuntime,
    RecordingResolutionContext,
    _build_recording_config,
    _extract_test_case_run_id,
    _SessionMonitor,
    build_config,
)


class _FakeSession:
    def __init__(self) -> None:
        self._events: dict[str, object] = {}

    async def start(self, *args: object, **kwargs: object) -> object:
        return {"args": args, "kwargs": kwargs}

    def on(self, event_name: str, handler: object) -> None:
        self._events[event_name] = handler


class _FakeReport:
    room = "room-123"
    started_at = 100.0
    timestamp = 125.0
    events: list[object] = []
    audio_recording_path = None

    def to_dict(self) -> dict[str, object]:
        return {"room": self.room, "events": []}


class RecordingConfigTests(unittest.TestCase):
    def test_default_configuration_disables_recording(self) -> None:
        config = build_config(api_key="test-key", external_agent_id="agent-123")

        self.assertEqual(config.recording.mode, RECORDING_MODE_NONE)

    def test_auto_record_audio_maps_to_session_audio_mode(self) -> None:
        config = _build_recording_config(
            recording=None,
            auto_record_audio=True,
            recording_mode=None,
        )

        self.assertEqual(config.mode, RECORDING_MODE_SESSION_AUDIO)

    def test_unsupported_modes_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            _build_recording_config(
                recording={"mode": "unsupported_mode"},
                auto_record_audio=False,
                recording_mode=None,
            )

    def test_participant_egress_requires_recording_resolver(self) -> None:
        with self.assertRaises(ValueError):
            build_config(
                api_key="test-key",
                external_agent_id="agent-123",
                recording={"mode": "participant_egress"},
            )

    def test_extract_test_case_run_id_falls_back_to_recording_context(self) -> None:
        value = _extract_test_case_run_id(
            participant_metadata_raw=None,
            recording_context={"customer_conversation_id": "conv-123"},
        )

        self.assertEqual(value, "conv-123")


class AttachSessionRecordingTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_audio_wraps_session_start_with_default_recording_options(
        self,
    ) -> None:
        runtime = HammingRuntime(
            build_config(
                api_key="test-key",
                external_agent_id="agent-123",
                recording={"mode": "session_audio"},
            )
        )

        session = _FakeSession()
        runtime.attach_session(session)

        result = await session.start(foo="bar")

        self.assertEqual(
            result,
            {
                "args": (),
                "kwargs": {
                    "foo": "bar",
                    "record": {
                        "audio": True,
                        "traces": False,
                        "logs": False,
                        "transcript": False,
                    },
                },
            },
        )
        self.assertIn("close", session._events)

    async def test_session_audio_does_not_override_explicit_record_argument(self) -> None:
        runtime = HammingRuntime(
            build_config(
                api_key="test-key",
                external_agent_id="agent-123",
                recording={"mode": "session_audio"},
            )
        )

        session = _FakeSession()
        runtime.attach_session(session)

        result = await session.start(record={"audio": False})

        self.assertEqual(
            result,
            {
                "args": (),
                "kwargs": {
                    "record": {
                        "audio": False,
                    },
                },
            },
        )

    async def test_room_composite_lookup_retries_after_transient_fetch_failure(self) -> None:
        runtime = HammingRuntime(build_config(api_key="test-key", external_agent_id="agent-123"))
        runtime.transport.fetch_test_case_run_recording_url = AsyncMock(
            side_effect=[RuntimeError("transient"), "https://recordings.example.com/room.ogg"]
        )

        monitor = _SessionMonitor(
            runtime=runtime,
            session=_FakeSession(),
            participant_identity=None,
            participant_metadata='{"conversation_id":"conv-123"}',
            external_agent_id="agent-123",
            job_ctx=None,
            session_key=1,
            recording_context=None,
        )
        resolution_context = RecordingResolutionContext(
            session=monitor._session,
            report=_FakeReport(),
            job_ctx=None,
            close_event=None,
            call_id="call-123",
            room_name="room-123",
            participant_identity=None,
            participant_metadata_raw='{"conversation_id":"conv-123"}',
            external_agent_id="agent-123",
            recording_context=None,
        )

        with patch("livekit.plugins.hamming._plugin.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            recording_url = await monitor._resolve_hamming_room_composite_recording_url(
                resolution_context=resolution_context
            )

        self.assertEqual(recording_url, "https://recordings.example.com/room.ogg")
        self.assertEqual(runtime.transport.fetch_test_case_run_recording_url.await_count, 2)
        sleep_mock.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
