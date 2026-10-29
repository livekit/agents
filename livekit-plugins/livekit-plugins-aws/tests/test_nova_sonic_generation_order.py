"""
Regression test for #8260: in the issue's event sequence, completionStart arrives before the
first user transcript, and creating the generation there announced it before
input_speech_started.
"""

import base64
import importlib.util
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

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
    if _mod in sys.modules:
        continue
    try:
        _installed = importlib.util.find_spec(_mod) is not None
    except (ImportError, ValueError):
        _installed = False
    if not _installed:
        sys.modules[_mod] = MagicMock()


def _turn(n: int) -> list[dict]:
    audio = base64.b64encode(b"\x01\x00" * 160).decode()
    return [
        {"event": {"contentStart": {"contentId": f"u{n}", "type": "TEXT", "role": "USER"}}},
        {"event": {"textOutput": {"contentId": f"u{n}", "role": "USER", "content": "hola"}}},
        {
            "event": {
                "contentEnd": {"contentId": f"u{n}", "type": "TEXT", "stopReason": "PARTIAL_TURN"}
            }
        },
        {
            "event": {
                "contentStart": {
                    "contentId": f"a{n}",
                    "type": "TEXT",
                    "role": "ASSISTANT",
                    "additionalModelFields": '{"generationStage":"SPECULATIVE"}',
                }
            }
        },
        {"event": {"textOutput": {"contentId": f"a{n}", "role": "ASSISTANT", "content": "Hola."}}},
        {
            "event": {
                "contentEnd": {"contentId": f"a{n}", "type": "TEXT", "stopReason": "PARTIAL_TURN"}
            }
        },
        {"event": {"contentStart": {"contentId": f"s{n}", "type": "AUDIO", "role": "ASSISTANT"}}},
        {"event": {"audioOutput": {"contentId": f"s{n}", "role": "ASSISTANT", "content": audio}}},
        {
            "event": {
                "contentEnd": {"contentId": f"s{n}", "type": "AUDIO", "stopReason": "END_TURN"}
            }
        },
    ]


async def test_first_turn_announced_after_user_speech_like_later_turns():
    from livekit.plugins.aws.experimental.realtime.realtime_model import RealtimeModel

    session = RealtimeModel(model="amazon.nova-2-5-sonic", modalities="mixed").session()
    order: list[str] = []
    session.on("input_speech_started", lambda _ev: order.append("input_speech_started"))
    session.on("generation_created", lambda _ev: order.append("generation_created"))

    await session._handle_event({"event": {"completionStart": {"completionId": "c1"}}})
    assert order == []

    turns = []
    for n in (1, 2):
        order.clear()
        for ev in _turn(n):
            await session._handle_event(ev)
        turns.append(list(order))

    assert turns[0][0] == "input_speech_started"
    assert turns[0] == turns[1]
