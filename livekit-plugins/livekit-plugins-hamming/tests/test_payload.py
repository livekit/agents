from __future__ import annotations

import unittest

from livekit.plugins.hamming._payload import (
    CALL_ID_STRATEGY_CUSTOM,
    CALL_ID_STRATEGY_PARTICIPANT_METADATA,
    PayloadBuildConfig,
    build_livekit_monitoring_envelope,
)


class _FakeReport:
    room = "room-123"
    started_at = 100.0
    timestamp = 125.0
    events: list[object] = []
    audio_recording_path = None

    def to_dict(self) -> dict[str, object]:
        return {"room": self.room, "events": []}


class PayloadEnvelopeTests(unittest.TestCase):
    def test_participant_metadata_strategy_uses_metadata_call_id(self) -> None:
        envelope = build_livekit_monitoring_envelope(
            config=PayloadBuildConfig(
                external_agent_id="agent-123",
                plugin_api_version="1.0.0",
                plugin_version="0.1.0",
                payload_schema_version="2026-03-02",
                call_id_strategy=CALL_ID_STRATEGY_PARTICIPANT_METADATA,
            ),
            report=_FakeReport(),
            participant_identity=None,
            participant_metadata_raw='{"call_id":"call-456"}',
            recording_context=None,
            close_event=None,
        )

        self.assertEqual(envelope["payload"]["call_id"], "call-456")

    def test_custom_call_id_resolver_falls_back_to_room_name_when_blank(self) -> None:
        envelope = build_livekit_monitoring_envelope(
            config=PayloadBuildConfig(
                external_agent_id="agent-123",
                plugin_api_version="1.0.0",
                plugin_version="0.1.0",
                payload_schema_version="2026-03-02",
                call_id_strategy=CALL_ID_STRATEGY_CUSTOM,
                resolve_call_id=lambda _context: "  ",
            ),
            report=_FakeReport(),
            participant_identity="user-1",
            participant_metadata_raw=None,
            recording_context=None,
            close_event=None,
        )

        self.assertEqual(envelope["payload"]["call_id"], "room-123")

    def test_recording_context_contributes_test_case_run_id(self) -> None:
        envelope = build_livekit_monitoring_envelope(
            config=PayloadBuildConfig(
                external_agent_id="agent-123",
                plugin_api_version="1.0.0",
                plugin_version="0.1.0",
                payload_schema_version="2026-03-02",
            ),
            report=_FakeReport(),
            participant_identity=None,
            participant_metadata_raw=None,
            recording_context={"customer_conversation_id": "conv-123"},
            close_event=None,
        )

        self.assertEqual(envelope["payload"]["test_case_run_id"], "conv-123")


if __name__ == "__main__":
    unittest.main()
