"""Check Rime v1 codecs against fixed wire envelopes, independent of the installed schema."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import aiohttp
import pytest
from google.protobuf import json_format
from rime_api import text_to_speech_pb2 as proto

from livekit.plugins.rime._websocket_v1 import _codec_for_protocol

pytestmark = pytest.mark.unit

_FIXTURES = json.loads((Path(__file__).parent / "fixtures/rime_websocket_v1.json").read_text())


@pytest.mark.parametrize("fixture", _FIXTURES)
@pytest.mark.parametrize("protocol", ["binary", "json"])
async def test_fixed_wire_envelope(fixture: dict[str, Any], protocol: str) -> None:
    codec = _codec_for_protocol(protocol)
    binary = bytes.fromhex(fixture["binary"])
    expected = fixture["json"]
    if fixture["message"] == "WebSocketRequest":
        request = json_format.ParseDict(expected, proto.WebSocketRequest())
        websocket = AsyncMock(spec=aiohttp.ClientWebSocketResponse)
        await codec.send_request(websocket, request)
        if protocol == "binary":
            websocket.send_bytes.assert_awaited_once_with(binary)
        else:
            websocket.send_str.assert_awaited_once()
            assert json.loads(websocket.send_str.call_args.args[0]) == expected
        assert json_format.MessageToDict(proto.WebSocketRequest.FromString(binary)) == expected
        if request.HasField("start"):
            assert not request.start.HasField("arcana_parameters")
    else:
        message = aiohttp.WSMessage(
            aiohttp.WSMsgType.BINARY if protocol == "binary" else aiohttp.WSMsgType.TEXT,
            binary if protocol == "binary" else json.dumps(expected),
            "",
        )
        response = codec.decode_response(message)
        assert json_format.MessageToDict(response) == expected
        assert response.SerializeToString() == binary
        assert response.WhichOneof("payload") == next(key for key in expected if key != "contextId")
