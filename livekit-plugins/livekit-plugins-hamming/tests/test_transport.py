from __future__ import annotations

import unittest
from os import environ
from types import SimpleNamespace
from unittest.mock import patch

from livekit.plugins.hamming._transport import ConnectionPolicy, HammingTransport


class _FakeEgress:
    async def list_egress(self, _request: object) -> object:
        return SimpleNamespace(items=[])


class _FakeLiveKitAPI:
    def __init__(self) -> None:
        self.egress = _FakeEgress()


class TransportPollingTests(unittest.IsolatedAsyncioTestCase):
    async def test_poll_continues_when_egress_not_visible_before_last_attempt(self) -> None:
        transport = HammingTransport(
            base_url="https://app.hamming.ai",
            api_key="test-key",
            policy=ConnectionPolicy(),
        )

        resolved_url, _status_name, should_continue = await transport._poll_plugin_managed_egress(
            lkapi=_FakeLiveKitAPI(),
            egress_id="egress-1",
            filepath="recordings/test.ogg",
            attempt=1,
            max_attempts=3,
            last_status_name=None,
        )

        self.assertIsNone(resolved_url)
        self.assertTrue(should_continue)

    async def test_poll_stops_with_fallback_url_on_final_missing_attempt(self) -> None:
        transport = HammingTransport(
            base_url="https://app.hamming.ai",
            api_key="test-key",
            policy=ConnectionPolicy(),
        )

        with patch.dict(environ, {"AWS_RECORDINGS_BUCKET": "bucket", "AWS_REGION": "us-east-1"}):
            (
                resolved_url,
                _status_name,
                should_continue,
            ) = await transport._poll_plugin_managed_egress(
                lkapi=_FakeLiveKitAPI(),
                egress_id="egress-1",
                filepath="recordings/test.ogg",
                attempt=3,
                max_attempts=3,
                last_status_name=None,
            )

        self.assertEqual(
            resolved_url,
            "https://bucket.s3.us-east-1.amazonaws.com/recordings/test.ogg",
        )
        self.assertFalse(should_continue)


if __name__ == "__main__":
    unittest.main()
