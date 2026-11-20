from __future__ import annotations

import pytest

from livekit.plugins import resemble

pytestmark = pytest.mark.plugin("resemble")


class _FakeTransport:
    def __init__(self, *scores: float) -> None:
        self._scores = list(scores)
        self.calls: list[dict[str, float | int]] = []

    async def submit(
        self,
        _pcm16: bytes,
        *,
        frame_length: int,
        request_timeout: float,
    ) -> dict[str, object]:
        index = len(self.calls)
        score = self._scores[index]
        self.calls.append({"frame_length": frame_length, "request_timeout": request_timeout})
        return {
            "uuid": f"detect-{index}",
            "status": "completed",
            "metrics": {
                "score": [score],
                "aggregated_score": score,
                "label": "fake" if score >= 0.7 else "real",
                "consistency": 0.9,
            },
        }


async def test_resemble_detect_standard_preset_requires_agreement() -> None:
    monitor = resemble.ResembleDetect(transport=_FakeTransport(0.94, 0.1, 0.93))
    raw_hits: list[resemble.DetectionResult] = []
    policy_hits: list[resemble.DetectionResult] = []
    monitor.on("fake_detected", raw_hits.append)
    monitor.on("synthetic_detected", policy_hits.append)

    await monitor._analyze_window(
        b"\0\0",
        index=0,
        window_start=0.0,
        participant_identity="caller",
    )

    assert len(raw_hits) == 1
    assert policy_hits == []
    assert monitor.verdict.to_dict()["label"] == "inconclusive"

    await monitor._analyze_window(
        b"\0\0",
        index=1,
        window_start=4.0,
        participant_identity="caller",
    )
    await monitor._analyze_window(
        b"\0\0",
        index=2,
        window_start=8.0,
        participant_identity="caller",
    )

    assert len(raw_hits) == 2
    assert len(policy_hits) == 1
    assert policy_hits[0].to_dict() == {
        "label": "synthetic",
        "raw_label": "fake",
        "score": 0.93,
        "confidence": 0.9,
        "window_ts": 12.0,
        "scan_index": 3,
        "is_final": False,
        "recommended_action": "block",
        "participant_identity": "caller",
        "detection_uuid": "detect-2",
        "latency": policy_hits[0].latency,
        "forced": False,
    }
    assert monitor.verdict.to_dict()["label"] == "synthetic"


async def test_resemble_detect_custom_transport_receives_options() -> None:
    transport = _FakeTransport(0.1)
    monitor = resemble.ResembleDetect(
        security="high",
        transport=transport,
        frame_length=3,
        request_timeout=12.0,
        sample_interval_seconds=11.0,
        agreement_window=2,
        min_fake_results=1,
    )

    await monitor._analyze_window(
        b"\0\0",
        index=0,
        window_start=0.0,
        participant_identity="caller",
    )

    assert isinstance(monitor, resemble.DetectionMonitor)
    assert monitor.security == "high"
    assert monitor._opts.mode == "continuous"
    assert monitor._opts.sample_interval_seconds == 11.0
    assert transport.calls == [{"frame_length": 3, "request_timeout": 12.0}]


async def test_resemble_detect_verdict_preserves_confirmed_synthetic_alert() -> None:
    monitor = resemble.ResembleDetect(
        security="spot",
        transport=_FakeTransport(0.94, 0.03),
    )
    policy_hits: list[resemble.DetectionResult] = []
    monitor.on("synthetic_detected", policy_hits.append)

    await monitor._analyze_window(
        b"\0\0",
        index=0,
        window_start=0.0,
        participant_identity="caller",
    )
    await monitor._analyze_window(
        b"\0\0",
        index=1,
        window_start=4.0,
        participant_identity="caller",
    )

    assert len(policy_hits) == 1
    assert monitor.verdict.label == "fake"


def test_resemble_detect_rest_transport_defaults_to_zero_retention() -> None:
    transport = resemble.RestDetectTransport(api_key="test")
    assert transport._extra_form_fields["zero_retention_mode"] == "true"


def test_resemble_detect_rest_transport_normalizes_extra_fields() -> None:
    transport = resemble.RestDetectTransport(
        api_key="test",
        zero_retention_mode=False,
        extra_form_fields={"use_ood_detector": True, "start_region": 1.25},
    )

    assert transport._extra_form_fields == {
        "zero_retention_mode": "false",
        "use_ood_detector": "true",
        "start_region": "1.25",
    }
