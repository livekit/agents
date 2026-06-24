"""
Regression tests for Nova Sonic turn-detection serialization in sessionStart.

Nova Sonic 2 (amazon.nova-2-sonic-v1:0) rejects the sessionStart event with a
ValidationException unless the turn-detection setting is nested under
turnDetectionConfiguration.  Nova Sonic 1 (amazon.nova-sonic-v1:0) predates
controllable endpointing and uses the legacy flat endpointingSensitivity field.

SonicEventBuilder serializes model-aware: Nova 2 (including cross-region
inference-profile ids such as us.amazon.nova-2-sonic-v1:0) → nested form;
Nova 1 → flat form.
"""

import json
import sys
from typing import Literal
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out the optional AWS Smithy/Bedrock SDK not installed in the base venv.
# Importing the realtime package pulls in realtime_model, which imports the SDK.
# ---------------------------------------------------------------------------
_AWS_STUBS = [
    "aws_sdk_bedrock_runtime",
    "aws_sdk_bedrock_runtime.client",
    "aws_sdk_bedrock_runtime.models",
    "aws_sdk_bedrock_runtime.config",
    "smithy_aws_core",
    "smithy_aws_core.identity",
    "smithy_aws_event_stream",
    "smithy_aws_event_stream.exceptions",
    "smithy_core",
    "smithy_core.aio",
    "smithy_core.aio.interfaces",
    "smithy_core.aio.interfaces.identity",
]
for _mod in _AWS_STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


def _session_start(model: str, sensitivity: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH") -> dict:
    from livekit.plugins.aws.experimental.realtime.events import SonicEventBuilder

    builder = SonicEventBuilder("prompt-name", "audio-content-name", model=model)
    payload = builder.create_session_start_event(endpointing_sensitivity=sensitivity)
    return json.loads(payload)["event"]["sessionStart"]


class TestSessionStartTurnDetection:
    """Model-aware serialization of the turn-detection setting."""

    def test_nova_sonic_2_nests_under_turn_detection_configuration(self):
        ss = _session_start("amazon.nova-2-sonic-v1:0")

        assert ss["turnDetectionConfiguration"]["endpointingSensitivity"] == "HIGH"
        # the legacy flat field must not be emitted for Nova 2
        assert "endpointingSensitivity" not in ss

    def test_nova_sonic_1_keeps_flat_field(self):
        ss = _session_start("amazon.nova-sonic-v1:0")

        assert ss["endpointingSensitivity"] == "HIGH"
        # the nested field must not be emitted for Nova 1
        assert "turnDetectionConfiguration" not in ss

    def test_cross_region_inference_profile_uses_nested_form(self):
        # Bedrock cross-region inference profiles prefix the model id with a
        # region group (us./eu./apac.); these are still Nova 2 and must nest.
        ss = _session_start("us.amazon.nova-2-sonic-v1:0")

        assert ss["turnDetectionConfiguration"]["endpointingSensitivity"] == "HIGH"
        assert "endpointingSensitivity" not in ss
