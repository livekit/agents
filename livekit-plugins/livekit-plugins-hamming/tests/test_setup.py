from __future__ import annotations

import unittest
from unittest.mock import patch

import livekit.plugins.hamming._plugin as plugin_mod
from livekit.plugins.hamming._setup import configure_hamming


class ConfigureHammingTests(unittest.TestCase):
    def setUp(self) -> None:
        plugin_mod._RUNTIME = None

    def tearDown(self) -> None:
        plugin_mod._RUNTIME = None

    def test_configure_hamming_builds_runtime_for_session_export(self) -> None:
        with patch("livekit.plugins.hamming._setup.configure_runtime", return_value="runtime"):
            result = configure_hamming(
                api_key="ham_test_key",
                external_agent_id="agent-123",
                recording={"mode": "session_audio"},
            )

        self.assertEqual(result, "runtime")

    def test_configure_hamming_preserves_session_first_options(self) -> None:
        configure_hamming(
            api_key="ham_test_key",
            external_agent_id="agent-123",
            capture={"interim_transcripts": True},
            recording={"mode": "session_audio"},
            streaming={"mode": "none"},
        )
        runtime = plugin_mod.get_runtime()
        assert runtime is not None
        self.assertTrue(runtime.config.capture.interim_transcripts)
        self.assertEqual(runtime.config.recording.mode, "session_audio")
        self.assertEqual(runtime.config.streaming.mode, "none")


if __name__ == "__main__":
    unittest.main()
