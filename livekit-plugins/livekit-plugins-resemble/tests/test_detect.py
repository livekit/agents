from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any

import pytest

from livekit.plugins.resemble import DetectionMonitor, ResembleDetect

pytestmark = pytest.mark.unit


class _FakeTransport:
    def __init__(self, *scores: float, consistency: float = 92.0) -> None:
        self._scores = list(scores)
        self._consistency = consistency
        self.calls: list[dict[str, float | int]] = []

    async def submit(
        self,
        _pcm16: bytes,
        *,
        frame_length: int,
        request_timeout: float,
    ) -> dict[str, Any]:
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
                "consistency": self._consistency,
            },
        }


class _OutOfOrderTransport:
    async def submit(
        self,
        pcm16: bytes,
        *,
        frame_length: int,
        request_timeout: float,
    ) -> dict[str, Any]:
        del frame_length, request_timeout
        index = pcm16[0]
        await asyncio.sleep((4 - index) * 0.01)
        score = (0.92, 0.91, 0.05, 0.04)[index]
        return {
            "uuid": f"detect-{index}",
            "status": "completed",
            "metrics": {
                "score": [score],
                "aggregated_score": score,
                "label": "fake" if score >= 0.7 else "real",
                "consistency": 92.0,
            },
        }


class _FakeAudioStream:
    def __init__(self, *chunks: bytes) -> None:
        self._events = iter(SimpleNamespace(frame=SimpleNamespace(data=chunk)) for chunk in chunks)
        self.closed = False

    def __aiter__(self) -> _FakeAudioStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed = True


class DetectionMonitorPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_standard_requires_agreement_before_confirming_synthetic(self) -> None:
        transport = _FakeTransport(0.92, 0.05, 0.88)
        monitor = DetectionMonitor(transport=transport)
        raw_hits = []
        confirmed_hits = []
        monitor.on("fake_detected", raw_hits.append)
        monitor.on("synthetic_detected", confirmed_hits.append)

        await monitor._analyze_window(
            b"\0\0",
            index=0,
            window_start=0.0,
            participant_identity="caller",
        )

        self.assertEqual(len(raw_hits), 1)
        self.assertEqual(len(confirmed_hits), 0)
        self.assertEqual(monitor.verdict.label, "inconclusive")

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

        self.assertEqual(len(raw_hits), 2)
        self.assertEqual(len(confirmed_hits), 1)
        self.assertEqual(monitor.verdict.label, "fake")
        self.assertEqual(monitor.verdict.normalized_label, "synthetic")

    async def test_spot_confirms_from_one_fake_result(self) -> None:
        transport = _FakeTransport(0.91)
        monitor = DetectionMonitor(security="spot", transport=transport)
        confirmed_hits = []
        monitor.on("synthetic_detected", confirmed_hits.append)

        await monitor._analyze_window(
            b"\0\0",
            index=0,
            window_start=0.0,
            participant_identity="caller",
        )

        self.assertEqual(len(confirmed_hits), 1)
        self.assertEqual(monitor.verdict.label, "fake")

    async def test_forced_check_can_confirm_immediately_when_configured(self) -> None:
        transport = _FakeTransport(0.94)
        monitor = DetectionMonitor(transport=transport, force_immediate_fake=True)
        confirmed_hits = []
        monitor.on("synthetic_detected", confirmed_hits.append)

        await monitor._analyze_window(
            b"\0\0",
            index=0,
            window_start=0.0,
            participant_identity="caller",
            forced=True,
        )

        self.assertEqual(len(confirmed_hits), 1)
        self.assertTrue(confirmed_hits[0].forced)

    async def test_forced_check_only_marks_one_window_in_non_sampled_modes(self) -> None:
        window = b"\x01\x00" * 32000

        for mode in ("continuous", "first_n"):
            with self.subTest(mode=mode):
                transport = _FakeTransport(0.1, 0.1)
                monitor = DetectionMonitor(
                    transport=transport,
                    mode=mode,
                    window_seconds=2.0,
                    analysis_budget_seconds=4.0,
                    silence_rms_threshold=0,
                )
                stream = _FakeAudioStream(window, window)

                monitor.check_now()
                await monitor._consume(stream, "caller")  # type: ignore[arg-type]

                self.assertEqual([result.forced for result in monitor.results], [True, False])
                self.assertFalse(monitor._force_pending)
                self.assertTrue(stream.closed)

    async def test_concurrent_results_use_chronological_window_order(self) -> None:
        monitor = DetectionMonitor(
            transport=_OutOfOrderTransport(),
            agreement_window=2,
            min_fake_results=2,
        )
        confirmed_hits = []
        monitor.on("synthetic_detected", confirmed_hits.append)

        await asyncio.gather(
            *(
                monitor._analyze_window(
                    bytes((index, 0)),
                    index=index,
                    window_start=index * 4.0,
                    participant_identity="caller",
                )
                for index in range(4)
            )
        )

        self.assertEqual([result.window_index for result in monitor.results], [0, 1, 2, 3])
        self.assertEqual(confirmed_hits, [])
        self.assertEqual(monitor.verdict.label, "inconclusive")

    async def test_final_verdict_preserves_confirmed_synthetic_alert(self) -> None:
        transport = _FakeTransport(0.91, 0.92, 0.02)
        monitor = DetectionMonitor(security="spot", transport=transport)
        confirmed_hits = []
        monitor.on("synthetic_detected", confirmed_hits.append)

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
        await monitor._analyze_window(
            b"\0\0",
            index=2,
            window_start=8.0,
            participant_identity="caller",
        )

        self.assertEqual(len(confirmed_hits), 1)
        self.assertEqual(monitor.verdict.label, "fake")

    async def test_result_payload_uses_stable_developer_shape(self) -> None:
        transport = _FakeTransport(0.82, consistency=91.0)
        monitor = DetectionMonitor(security="spot", transport=transport)

        await monitor._analyze_window(
            b"\0\0",
            index=0,
            window_start=12.0,
            participant_identity="caller",
        )

        result = monitor.results[0]
        self.assertEqual(result.normalized_label, "synthetic")
        self.assertEqual(result.score, 0.82)
        self.assertEqual(result.confidence, 0.91)
        self.assertEqual(result.scan_index, 1)
        self.assertEqual(result.window_ts, 16.0)
        self.assertEqual(result.recommended_action, "block")
        self.assertEqual(
            result.to_dict(),
            {
                "label": "synthetic",
                "raw_label": "fake",
                "score": 0.82,
                "confidence": 0.91,
                "window_ts": 16.0,
                "scan_index": 1,
                "is_final": False,
                "recommended_action": "block",
                "participant_identity": "caller",
                "detection_uuid": "detect-0",
                "latency": result.latency,
                "forced": False,
            },
        )

    async def test_custom_transport_receives_runtime_options(self) -> None:
        transport = _FakeTransport(0.1)
        monitor = ResembleDetect(
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

        self.assertIsInstance(monitor, DetectionMonitor)
        self.assertEqual(monitor.security, "high")
        self.assertEqual(monitor._opts.mode, "continuous")
        self.assertEqual(monitor._opts.sample_interval_seconds, 11.0)
        self.assertEqual(transport.calls, [{"frame_length": 3, "request_timeout": 12.0}])

    def test_invalid_explicit_overrides_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "samples must be >= 1"):
            DetectionMonitor(transport=_FakeTransport(0.1), samples=0)

        with self.assertRaisesRegex(ValueError, "min_fake_results must be <= agreement_window"):
            DetectionMonitor(
                transport=_FakeTransport(0.1),
                agreement_window=1,
                min_fake_results=2,
            )


if __name__ == "__main__":
    unittest.main()
