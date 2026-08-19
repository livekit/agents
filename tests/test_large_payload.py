from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from livekit.agents.utils.large_payload import (
    ATTR_PAYLOAD_CONTENT_TYPE,
    ATTR_PAYLOAD_ID,
    ATTR_PAYLOAD_SHA256,
    ATTR_PAYLOAD_SIZE,
    ATTR_PAYLOAD_TOPIC,
    DESCRIPTOR_TYPE,
    LargePayloadChecksumError,
    LargePayloadDescriptorTooLargeError,
    LargePayloadError,
    LargePayloadTooLargeError,
    parse_large_payload_descriptor,
    publish_large_payload,
    read_large_payload_stream,
)

pytestmark = pytest.mark.unit


@dataclass
class _FakeWriter:
    chunks: list[bytes] = field(default_factory=list)
    closed: bool = False
    close_reason: str = ""
    close_attributes: dict[str, str] | None = None
    write_error: BaseException | None = None
    close_error: BaseException | None = None

    async def write(self, data: bytes) -> None:
        if self.write_error is not None:
            raise self.write_error
        self.chunks.append(data)

    async def aclose(self, *, reason: str = "", attributes: dict[str, str] | None = None) -> None:
        self.closed = True
        self.close_reason = reason
        self.close_attributes = attributes
        if self.close_error is not None:
            raise self.close_error


class _FakeParticipant:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []
        self.streams: list[dict[str, Any]] = []
        self.writer = _FakeWriter()

    async def publish_data(
        self,
        payload: bytes,
        *,
        reliable: bool = True,
        destination_identities: list[str] | None = None,
        topic: str = "",
    ) -> None:
        self.published.append(
            {
                "payload": payload,
                "reliable": reliable,
                "destination_identities": destination_identities,
                "topic": topic,
            }
        )

    async def stream_bytes(self, **kwargs: Any) -> _FakeWriter:
        self.streams.append(kwargs)
        return self.writer


class _FakeReader:
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_publish_large_payload_inlines_small_payload() -> None:
    participant = _FakeParticipant()

    info = await publish_large_payload(
        participant, b'{"status":"ok"}', topic="generic-status", payload_id="payload-1"
    )

    assert not participant.streams
    assert len(participant.published) == 1
    assert participant.published[0]["topic"] == "generic-status"

    descriptor = parse_large_payload_descriptor(participant.published[0]["payload"])
    assert descriptor.payload_id == "payload-1"
    assert descriptor.transfer == "inline"
    assert descriptor.content_type == "application/octet-stream"
    assert descriptor.decode_inline_payload() == b'{"status":"ok"}'
    assert info.descriptor == descriptor


@pytest.mark.asyncio
async def test_publish_large_payload_rejects_empty_content_type() -> None:
    participant = _FakeParticipant()

    with pytest.raises(ValueError, match="content_type"):
        await publish_large_payload(
            participant,
            b"data",
            topic="generic-status",
            content_type="",
        )

    assert not participant.streams
    assert not participant.published


@pytest.mark.asyncio
async def test_publish_large_payload_streams_when_descriptor_exceeds_inline_limit() -> None:
    participant = _FakeParticipant()

    info = await publish_large_payload(
        participant,
        b"x" * 64,
        topic="generic-payload",
        content_type="application/json",
        attributes={"source": "test"},
        destination_identities=["receiver"],
        payload_id="payload-2",
        max_inline_bytes=80,
    )

    assert len(participant.streams) == 1
    stream = participant.streams[0]
    assert stream["name"] == "payload-2.bin"
    assert stream["topic"] == "generic-payload.payload"
    assert stream["mime_type"] == "application/json"
    assert stream["total_size"] == 64
    assert stream["destination_identities"] == ["receiver"]
    assert stream["attributes"]["source"] == "test"
    assert stream["attributes"][ATTR_PAYLOAD_ID] == "payload-2"
    assert stream["attributes"][ATTR_PAYLOAD_TOPIC] == "generic-payload"
    assert stream["attributes"][ATTR_PAYLOAD_SIZE] == "64"
    assert stream["attributes"][ATTR_PAYLOAD_CONTENT_TYPE] == "application/json"
    assert stream["attributes"][ATTR_PAYLOAD_SHA256] == info.descriptor.sha256

    assert participant.writer.chunks == [b"x" * 64]
    assert participant.writer.closed

    assert len(participant.published) == 1
    descriptor = parse_large_payload_descriptor(participant.published[0]["payload"])
    assert descriptor.transfer == "byte_stream"
    assert descriptor.stream_topic == "generic-payload.payload"
    assert descriptor.stream_name == "payload-2.bin"
    assert descriptor.data is None
    assert descriptor.attributes == {"source": "test"}


@pytest.mark.asyncio
async def test_publish_large_payload_streams_when_inline_exceeds_descriptor_limit() -> None:
    participant = _FakeParticipant()

    info = await publish_large_payload(
        participant,
        b"x" * 200,
        topic="generic-payload",
        payload_id="payload-2a",
        max_inline_bytes=1000,
        max_descriptor_bytes=400,
    )

    assert info.descriptor.transfer == "byte_stream"
    assert participant.streams
    assert participant.published


@pytest.mark.asyncio
async def test_publish_large_payload_skips_inline_encoding_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participant = _FakeParticipant()

    def _fail_b64encode(data: bytes) -> bytes:
        raise AssertionError("inline base64 path should not run")

    monkeypatch.setattr("livekit.agents.utils.large_payload.base64.b64encode", _fail_b64encode)

    info = await publish_large_payload(
        participant,
        b"x" * 64,
        topic="generic-payload",
        payload_id="payload-2b",
        max_inline_bytes=0,
    )

    assert info.descriptor.transfer == "byte_stream"
    assert participant.streams
    assert participant.published


@pytest.mark.asyncio
async def test_publish_large_payload_rejects_payload_over_stream_limit() -> None:
    participant = _FakeParticipant()

    with pytest.raises(LargePayloadTooLargeError):
        await publish_large_payload(
            participant,
            b"x" * 17,
            topic="generic-payload",
            payload_id="payload-too-large",
            max_inline_bytes=0,
            max_stream_bytes=16,
        )

    assert not participant.streams
    assert not participant.published


@pytest.mark.asyncio
async def test_publish_large_payload_allows_custom_stream_limit() -> None:
    participant = _FakeParticipant()

    info = await publish_large_payload(
        participant,
        b"x" * 17,
        topic="generic-payload",
        payload_id="payload-custom-limit",
        max_inline_bytes=0,
        max_stream_bytes=17,
    )

    assert info.descriptor.transfer == "byte_stream"
    assert participant.streams
    assert participant.published


@pytest.mark.asyncio
async def test_publish_large_payload_uses_custom_stream_topic_and_name() -> None:
    participant = _FakeParticipant()

    await publish_large_payload(
        participant,
        "payload body",
        topic="generic-control",
        payload_id="payload-3",
        max_inline_bytes=0,
        stream_topic="generic-streams",
        stream_name="payload-3.json",
    )

    stream = participant.streams[0]
    assert stream["topic"] == "generic-streams"
    assert stream["name"] == "payload-3.json"

    descriptor = parse_large_payload_descriptor(participant.published[0]["payload"])
    assert descriptor.stream_topic == "generic-streams"
    assert descriptor.stream_name == "payload-3.json"


@pytest.mark.asyncio
async def test_publish_large_payload_aborts_stream_when_write_fails() -> None:
    participant = _FakeParticipant()
    participant.writer.write_error = RuntimeError("write failed")
    participant.writer.close_error = RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="write failed"):
        await publish_large_payload(
            participant,
            b"x" * 64,
            topic="generic-payload",
            payload_id="payload-3a",
            max_inline_bytes=0,
        )

    assert participant.writer.closed
    assert participant.writer.close_reason == "large payload write failed"
    assert not participant.published


@pytest.mark.asyncio
async def test_publish_large_payload_rejects_oversized_stream_descriptor() -> None:
    participant = _FakeParticipant()

    with pytest.raises(LargePayloadDescriptorTooLargeError):
        await publish_large_payload(
            participant,
            b"x" * 64,
            topic="generic-payload",
            attributes={"metadata": "x" * 128},
            payload_id="payload-4",
            max_inline_bytes=1,
            max_descriptor_bytes=80,
        )

    assert not participant.streams
    assert not participant.published


@pytest.mark.asyncio
async def test_read_large_payload_stream_validates_size_and_checksum() -> None:
    expected = "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7"

    payload = await read_large_payload_stream(
        _FakeReader(b"da", b"ta"),
        expected_size=4,
        expected_sha256=expected,
    )

    assert payload == b"data"


@pytest.mark.asyncio
async def test_read_large_payload_stream_rejects_payload_over_max_bytes() -> None:
    with pytest.raises(LargePayloadTooLargeError):
        await read_large_payload_stream(
            _FakeReader(b"abc", b"def"),
            max_bytes=4,
            require_expected_metadata=False,
        )


@pytest.mark.asyncio
async def test_read_large_payload_stream_rejects_expected_size_over_max_before_read() -> None:
    with pytest.raises(LargePayloadTooLargeError):
        await read_large_payload_stream(
            _FakeReader(b"a"),
            max_bytes=4,
            expected_size=5,
            expected_sha256="0" * 64,
        )


@pytest.mark.asyncio
async def test_read_large_payload_stream_rejects_checksum_mismatch() -> None:
    with pytest.raises(LargePayloadChecksumError):
        await read_large_payload_stream(
            _FakeReader(b"data"), expected_size=4, expected_sha256="bad"
        )


@pytest.mark.asyncio
async def test_read_large_payload_stream_rejects_missing_expected_metadata() -> None:
    with pytest.raises(LargePayloadError, match="expected_size"):
        await read_large_payload_stream(_FakeReader(b"data"), expected_sha256="bad")


@pytest.mark.asyncio
async def test_read_large_payload_stream_rejects_invalid_checksum_text() -> None:
    with pytest.raises(LargePayloadChecksumError, match="valid hex digest"):
        await read_large_payload_stream(
            _FakeReader(b"data"),
            expected_size=4,
            expected_sha256="x" * 63 + chr(9731),
        )


@pytest.mark.asyncio
async def test_read_large_payload_stream_accepts_uppercase_checksum() -> None:
    expected = "3A6EB0790F39AC87C94F3856B2DD2C5D110E6811602261A9A923D3BB23ADC8B7"

    payload = await read_large_payload_stream(
        _FakeReader(b"data"),
        expected_size=4,
        expected_sha256=expected,
    )

    assert payload == b"data"


def test_parse_large_payload_descriptor_rejects_non_descriptor() -> None:
    payload = json.dumps({"type": "other", "version": 1}).encode("utf-8")

    with pytest.raises(LargePayloadError):
        parse_large_payload_descriptor(payload)


def test_parse_large_payload_descriptor_rejects_invalid_utf8() -> None:
    with pytest.raises(LargePayloadError, match="UTF-8"):
        parse_large_payload_descriptor(b"\xff")


def test_parse_large_payload_descriptor_rejects_missing_required_fields() -> None:
    payload = json.dumps(
        {
            "type": DESCRIPTOR_TYPE,
            "version": 1,
            "transfer": "inline",
            "topic": "generic",
            "size": 1,
            "sha256": "digest",
            "content_type": "application/octet-stream",
        }
    )

    with pytest.raises(LargePayloadError, match="'id'"):
        parse_large_payload_descriptor(payload)
