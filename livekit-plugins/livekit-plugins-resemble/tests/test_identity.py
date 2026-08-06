from __future__ import annotations

import unittest
from typing import Any

import pytest

from livekit.plugins.resemble import ResembleIdentity

pytestmark = pytest.mark.unit


class _FakeTransport:
    def __init__(self, item: dict[str, Any]) -> None:
        self._item = item
        self.calls: list[dict[str, Any]] = []

    async def search(self, url: str, *, request_timeout: float) -> dict[str, Any]:
        self.calls.append({"url": url, "request_timeout": request_timeout})
        return self._item


class _FakeHost:
    def __init__(self, url: str = "https://clips.example.com/turn.wav") -> None:
        self.url = url
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, audio: bytes, filename: str) -> str:
        self.calls.append({"audio": audio, "filename": filename})
        return self.url


_ENROLLED = {
    "uuid-harold": {"name": "Harold", "distance": 91.4},
    "uuid-alex": {"name": "Alex", "distance": 40.0},
    "uuid-unscored": {"name": "Broken", "distance": None},
}


class ResembleIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_url_ranks_matches_and_applies_threshold(self) -> None:
        transport = _FakeTransport(_ENROLLED)
        identity = ResembleIdentity(transport=transport)

        result = await identity.search_url("https://clips.example.com/turn.wav")

        self.assertEqual([match.name for match in result.matches], ["Harold", "Alex", "Broken"])
        self.assertTrue(result.matched)
        self.assertEqual(result.name, "Harold")
        self.assertEqual(result.score, 91.4)
        self.assertEqual(result.threshold, 70.0)
        self.assertEqual(
            transport.calls,
            [{"url": "https://clips.example.com/turn.wav", "request_timeout": 60.0}],
        )

    async def test_below_threshold_is_not_matched(self) -> None:
        transport = _FakeTransport({"uuid-alex": {"name": "Alex", "distance": 40.0}})
        identity = ResembleIdentity(transport=transport)

        result = await identity.search_url("https://clips.example.com/turn.wav")

        self.assertFalse(result.matched)
        self.assertEqual(result.name, "Alex")

    async def test_per_call_threshold_overrides_configured_threshold(self) -> None:
        transport = _FakeTransport({"uuid-alex": {"name": "Alex", "distance": 40.0}})
        identity = ResembleIdentity(transport=transport, threshold=90.0)

        result = await identity.search_url(
            "https://clips.example.com/turn.wav",
            threshold=30.0,
        )

        self.assertTrue(result.matched)
        self.assertEqual(result.threshold, 30.0)

    async def test_no_enrollments_yields_unmatched_empty_result(self) -> None:
        identity = ResembleIdentity(transport=_FakeTransport({}))

        result = await identity.search_url("https://clips.example.com/turn.wav")

        self.assertEqual(result.matches, [])
        self.assertFalse(result.matched)
        self.assertIsNone(result.name)
        self.assertIsNone(result.score)

    async def test_result_payload_uses_stable_developer_shape(self) -> None:
        identity = ResembleIdentity(transport=_FakeTransport(_ENROLLED))

        result = await identity.search_url("https://clips.example.com/turn.wav")

        self.assertEqual(
            result.to_dict(),
            {
                "matched": True,
                "name": "Harold",
                "score": 91.4,
                "threshold": 70.0,
                "matches": [
                    {"uuid": "uuid-harold", "name": "Harold", "score": 91.4},
                    {"uuid": "uuid-alex", "name": "Alex", "score": 40.0},
                    {"uuid": "uuid-unscored", "name": "Broken", "score": None},
                ],
            },
        )

    async def test_search_hosts_audio_then_searches_hosted_url(self) -> None:
        transport = _FakeTransport(_ENROLLED)
        host = _FakeHost()
        identity = ResembleIdentity(transport=transport, audio_host=host)

        result = await identity.search(b"pcm-bytes", filename="turn-7.wav")

        self.assertTrue(result.matched)
        self.assertEqual(host.calls, [{"audio": b"pcm-bytes", "filename": "turn-7.wav"}])
        self.assertEqual(transport.calls[0]["url"], host.url)

    async def test_search_without_audio_host_explains_url_requirement(self) -> None:
        identity = ResembleIdentity(transport=_FakeTransport(_ENROLLED))

        with self.assertRaisesRegex(ValueError, "audio_host"):
            await identity.search(b"pcm-bytes")

    async def test_invalid_inputs_are_rejected(self) -> None:
        identity = ResembleIdentity(transport=_FakeTransport(_ENROLLED))

        with self.assertRaisesRegex(ValueError, "url is required"):
            await identity.search_url("   ")

        with self.assertRaisesRegex(ValueError, "audio is required"):
            await identity.search(b"")

        with self.assertRaisesRegex(ValueError, "threshold"):
            ResembleIdentity(transport=_FakeTransport(_ENROLLED), threshold=101.0)

        with self.assertRaisesRegex(ValueError, "threshold"):
            await identity.search_url("https://clips.example.com/turn.wav", threshold=-1.0)


if __name__ == "__main__":
    unittest.main()
