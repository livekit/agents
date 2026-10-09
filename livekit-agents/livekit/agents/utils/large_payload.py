"""Utilities for publishing payloads that may be too large for a data packet.

The helper publishes a compact descriptor on a data topic. If the descriptor can
carry the payload inline, the bytes are base64-encoded into that descriptor. If
not, the bytes are sent through a byte stream and the descriptor points to that
stream by topic/name with size and checksum metadata.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
from collections.abc import AsyncIterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from livekit import rtc

from .misc import shortuuid

DEFAULT_INLINE_PAYLOAD_LIMIT = 15 * 1024
DEFAULT_STREAM_PAYLOAD_LIMIT = 64 * 1024 * 1024

DESCRIPTOR_TYPE = "lk.large_payload"
DESCRIPTOR_VERSION = 1

ATTR_PAYLOAD_ID = "lk.large_payload.id"
ATTR_PAYLOAD_TOPIC = "lk.large_payload.topic"
ATTR_PAYLOAD_SHA256 = "lk.large_payload.sha256"
ATTR_PAYLOAD_SIZE = "lk.large_payload.size"
ATTR_PAYLOAD_CONTENT_TYPE = "lk.large_payload.content_type"

PayloadTransfer = Literal["inline", "byte_stream"]
_SHA256_HEX_CHARS = frozenset("0123456789abcdef")


class LargePayloadError(ValueError):
    """Base exception for large payload helper errors."""


class LargePayloadTooLargeError(LargePayloadError):
    """Raised when a received stream exceeds its configured size limit."""


class LargePayloadDescriptorTooLargeError(LargePayloadError):
    """Raised when the descriptor itself is too large for a data packet."""


class LargePayloadChecksumError(LargePayloadError):
    """Raised when received bytes do not match the expected SHA-256 digest."""


@dataclass(frozen=True)
class LargePayloadDescriptor:
    """Control message describing a payload sent inline or through a byte stream."""

    payload_id: str
    topic: str
    transfer: PayloadTransfer
    size: int
    sha256: str
    content_type: str
    stream_topic: str | None = None
    stream_name: str | None = None
    data: str | None = None
    attributes: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable descriptor representation.

        The returned dictionary is suitable for sending as the control message
        on a data topic after JSON serialization.

        Returns:
            A JSON-serializable descriptor dictionary.
        """
        descriptor: dict[str, Any] = {
            "type": DESCRIPTOR_TYPE,
            "version": DESCRIPTOR_VERSION,
            "id": self.payload_id,
            "topic": self.topic,
            "transfer": self.transfer,
            "size": self.size,
            "sha256": self.sha256,
            "content_type": self.content_type,
        }
        if self.stream_topic is not None:
            descriptor["stream_topic"] = self.stream_topic
        if self.stream_name is not None:
            descriptor["stream_name"] = self.stream_name
        if self.data is not None:
            descriptor["data"] = self.data
            descriptor["encoding"] = "base64"
        if self.attributes:
            descriptor["attributes"] = dict(self.attributes)
        return descriptor

    @classmethod
    def from_dict(cls, descriptor: Mapping[str, Any]) -> LargePayloadDescriptor:
        """Validate and build a descriptor from parsed JSON data.

        Args:
            descriptor: Parsed JSON object received on a data topic.

        Returns:
            A validated large payload descriptor.

        Raises:
            LargePayloadError: If the object is not a supported descriptor.
        """
        if descriptor.get("type") != DESCRIPTOR_TYPE:
            raise LargePayloadError("not a LiveKit large payload descriptor")
        version = descriptor.get("version")
        if version != DESCRIPTOR_VERSION:
            raise LargePayloadError(f"unsupported large payload descriptor version: {version}")

        transfer = descriptor.get("transfer")
        if transfer not in ("inline", "byte_stream"):
            raise LargePayloadError(f"unsupported large payload transfer: {transfer}")

        payload_id = _required_str(descriptor, "id")
        topic = _required_str(descriptor, "topic")
        size = _required_int(descriptor, "size")
        sha256 = _required_str(descriptor, "sha256")
        content_type = _required_str(descriptor, "content_type")

        attributes_raw = descriptor.get("attributes")
        attributes = (
            {str(k): str(v) for k, v in attributes_raw.items()}
            if isinstance(attributes_raw, Mapping)
            else None
        )

        stream_topic = _optional_str(descriptor, "stream_topic")
        stream_name = _optional_str(descriptor, "stream_name")
        data = _optional_str(descriptor, "data")
        if transfer == "byte_stream" and (not stream_topic or not stream_name):
            raise LargePayloadError(
                "large payload descriptor must include 'stream_topic' and 'stream_name' "
                "for byte_stream transfer"
            )
        if transfer == "inline" and data is None:
            raise LargePayloadError(
                "large payload descriptor must include 'data' for inline transfer"
            )

        return cls(
            payload_id=payload_id,
            topic=topic,
            transfer=transfer,
            size=size,
            sha256=sha256,
            content_type=content_type,
            stream_topic=stream_topic,
            stream_name=stream_name,
            data=data,
            attributes=attributes,
        )

    def decode_inline_payload(self) -> bytes:
        """Decode and verify an inline payload carried by this descriptor."""
        if self.transfer != "inline" or self.data is None:
            raise LargePayloadError("descriptor does not contain an inline payload")
        try:
            data = base64.b64decode(self.data.encode("ascii"), validate=True)
        except Exception as e:
            raise LargePayloadError("inline payload is not valid base64") from e
        _verify_payload(data, expected_size=self.size, expected_sha256=self.sha256)
        return data


@dataclass(frozen=True)
class LargePayloadInfo:
    """Result returned after a payload descriptor has been published."""

    descriptor: LargePayloadDescriptor
    descriptor_bytes: int


async def publish_large_payload(
    participant: rtc.LocalParticipant,
    payload: bytes | str,
    *,
    topic: str,
    content_type: str = "application/octet-stream",
    attributes: Mapping[str, str] | None = None,
    reliable: bool = True,
    destination_identities: Sequence[str] | None = None,
    max_inline_bytes: int = DEFAULT_INLINE_PAYLOAD_LIMIT,
    max_descriptor_bytes: int = DEFAULT_INLINE_PAYLOAD_LIMIT,
    max_stream_bytes: int | None = DEFAULT_STREAM_PAYLOAD_LIMIT,
    payload_id: str | None = None,
    stream_topic: str | None = None,
    stream_name: str | None = None,
) -> LargePayloadInfo:
    """Publish a payload descriptor, using a byte stream when it cannot fit inline.

    The receiver should listen for descriptors on ``topic``. Small payloads are
    embedded in the descriptor as base64. Larger payloads are sent through
    ``stream_bytes`` and the descriptor identifies the stream topic/name and the
    expected size/checksum.

    Args:
        participant: Local participant used to publish data and byte streams.
        payload: Bytes or UTF-8 text to publish.
        topic: Data topic used for the descriptor.
        content_type: MIME type for the payload.
        attributes: Optional application metadata copied into the descriptor and
            stream attributes.
        reliable: Whether to publish the descriptor on the reliable data channel.
        destination_identities: Optional recipient identities.
        max_inline_bytes: Maximum descriptor size allowed for inline transfer.
            Set to ``0`` to always use a byte stream.
        max_descriptor_bytes: Maximum size of the data-channel descriptor.
        max_stream_bytes: Maximum byte-stream payload size. Set to ``None`` to
            disable this helper-level bound.
        payload_id: Optional stable payload identifier.
        stream_topic: Optional byte-stream topic for streamed payloads.
        stream_name: Optional byte-stream name for streamed payloads.

    Returns:
        Information about the descriptor that was published.

    Raises:
        ValueError: If a size limit or topic argument is invalid.
        LargePayloadTooLargeError: If the payload exceeds ``max_stream_bytes``.
        LargePayloadDescriptorTooLargeError: If the descriptor cannot fit in a
            data packet.
    """
    if not topic:
        raise ValueError("topic is required")
    if not content_type:
        raise ValueError("content_type is required")
    if max_inline_bytes < 0:
        raise ValueError("max_inline_bytes must be non-negative")
    if max_descriptor_bytes < 1:
        raise ValueError("max_descriptor_bytes must be positive")
    if max_stream_bytes is not None and max_stream_bytes < 0:
        raise ValueError("max_stream_bytes must be non-negative")
    inline_limit = min(max_inline_bytes, max_descriptor_bytes)

    data = _coerce_payload(payload)
    payload_id = payload_id or shortuuid("payload_")
    sha256 = hashlib.sha256(data).hexdigest()
    size = len(data)
    attrs = _string_attrs(attributes)
    destinations = list(destination_identities or [])

    if (
        inline_limit > 0
        and _estimate_inline_descriptor_size(
            payload_id=payload_id,
            topic=topic,
            size=size,
            sha256=sha256,
            content_type=content_type,
            attributes=attrs or None,
        )
        <= inline_limit
    ):
        inline_descriptor = LargePayloadDescriptor(
            payload_id=payload_id,
            topic=topic,
            transfer="inline",
            size=size,
            sha256=sha256,
            content_type=content_type,
            data=base64.b64encode(data).decode("ascii"),
            attributes=attrs or None,
        )
        inline_bytes = _descriptor_bytes(inline_descriptor)
        if len(inline_bytes) <= inline_limit:
            await participant.publish_data(
                inline_bytes,
                reliable=reliable,
                destination_identities=destinations,
                topic=topic,
            )
            return LargePayloadInfo(
                descriptor=inline_descriptor,
                descriptor_bytes=len(inline_bytes),
            )

    if max_stream_bytes is not None and size > max_stream_bytes:
        raise LargePayloadTooLargeError(
            f"large payload exceeded max_stream_bytes ({max_stream_bytes})"
        )

    stream_topic = stream_topic or f"{topic}.payload"
    stream_name = stream_name or f"{payload_id}.bin"
    stream_descriptor = LargePayloadDescriptor(
        payload_id=payload_id,
        topic=topic,
        transfer="byte_stream",
        size=size,
        sha256=sha256,
        content_type=content_type,
        stream_topic=stream_topic,
        stream_name=stream_name,
        attributes=attrs or None,
    )
    stream_descriptor_bytes = _descriptor_bytes(stream_descriptor)
    if len(stream_descriptor_bytes) > max_descriptor_bytes:
        raise LargePayloadDescriptorTooLargeError(
            f"large payload descriptor exceeded max_descriptor_bytes ({max_descriptor_bytes})"
        )

    stream_attrs = {
        **attrs,
        ATTR_PAYLOAD_ID: payload_id,
        ATTR_PAYLOAD_TOPIC: topic,
        ATTR_PAYLOAD_SHA256: sha256,
        ATTR_PAYLOAD_SIZE: str(size),
        ATTR_PAYLOAD_CONTENT_TYPE: content_type,
    }
    writer = await participant.stream_bytes(
        name=stream_name,
        topic=stream_topic,
        mime_type=content_type,
        attributes=stream_attrs,
        total_size=size,
        destination_identities=destinations,
    )
    try:
        await writer.write(data)
    except BaseException:
        with contextlib.suppress(Exception):
            await writer.aclose(reason="large payload write failed")
        raise
    else:
        await writer.aclose()

    await participant.publish_data(
        stream_descriptor_bytes,
        reliable=reliable,
        destination_identities=destinations,
        topic=topic,
    )
    return LargePayloadInfo(
        descriptor=stream_descriptor,
        descriptor_bytes=len(stream_descriptor_bytes),
    )


def parse_large_payload_descriptor(payload: bytes | str) -> LargePayloadDescriptor:
    """Parse a JSON large-payload descriptor received on a data topic.

    Args:
        payload: Descriptor bytes or text received through ``publish_data``.

    Returns:
        A validated descriptor.

    Raises:
        LargePayloadError: If the payload is not valid descriptor JSON.
    """
    try:
        raw = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    except UnicodeDecodeError as e:
        raise LargePayloadError("large payload descriptor is not valid UTF-8") from e
    try:
        descriptor = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LargePayloadError("large payload descriptor is not valid JSON") from e
    if not isinstance(descriptor, Mapping):
        raise LargePayloadError("large payload descriptor must be a JSON object")
    return LargePayloadDescriptor.from_dict(descriptor)


async def read_large_payload_stream(
    reader: AsyncIterable[bytes],
    *,
    max_bytes: int | None = DEFAULT_STREAM_PAYLOAD_LIMIT,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    require_expected_metadata: bool = True,
) -> bytes:
    """Read a byte stream with size and checksum validation.

    Args:
        reader: Async byte stream reader.
        max_bytes: Maximum bytes to buffer while reading. Defaults to
            ``DEFAULT_STREAM_PAYLOAD_LIMIT``. Set to ``None`` only for trusted
            senders or when another layer already enforces a bound.
        expected_size: Expected payload size from a descriptor or stream
            attribute. Required unless ``require_expected_metadata`` is ``False``.
            If it exceeds ``max_bytes``, reading fails before consuming the
            stream.
        expected_sha256: Expected SHA-256 hex digest. Required unless
            ``require_expected_metadata`` is ``False``.
        require_expected_metadata: Whether ``expected_size`` and
            ``expected_sha256`` are required. Defaults to ``True`` so untrusted
            streams are validated unless callers explicitly opt out.

    Returns:
        The complete stream payload.

    Raises:
        ValueError: If a size argument is invalid.
        LargePayloadTooLargeError: If the stream exceeds ``max_bytes``.
        LargePayloadChecksumError: If checksum validation fails.
        LargePayloadError: If size validation fails.
    """
    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    if expected_size is not None and expected_size < 0:
        raise ValueError("expected_size must be non-negative")
    if require_expected_metadata:
        if expected_size is None:
            raise LargePayloadError("expected_size is required")
        if expected_sha256 is None:
            raise LargePayloadError("expected_sha256 is required")
    if max_bytes is not None and expected_size is not None and expected_size > max_bytes:
        raise LargePayloadTooLargeError(
            f"large payload expected_size exceeds max_bytes ({max_bytes})"
        )

    chunks: list[bytes] = []
    total = 0
    async for chunk in reader:
        data = bytes(chunk)
        total += len(data)
        if max_bytes is not None and total > max_bytes:
            raise LargePayloadTooLargeError(
                f"large payload stream exceeded max_bytes ({max_bytes})"
            )
        chunks.append(data)

    payload = b"".join(chunks)
    _verify_payload(payload, expected_size=expected_size, expected_sha256=expected_sha256)
    return payload


def _coerce_payload(payload: bytes | str) -> bytes:
    return payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)


def _string_attrs(attributes: Mapping[str, str] | None) -> dict[str, str]:
    if not attributes:
        return {}
    return {str(k): str(v) for k, v in attributes.items()}


def _descriptor_bytes(descriptor: LargePayloadDescriptor) -> bytes:
    return json.dumps(descriptor.to_dict(), separators=(",", ":")).encode("utf-8")


def _base64_encoded_len(size: int) -> int:
    return ((size + 2) // 3) * 4


def _estimate_inline_descriptor_size(
    *,
    payload_id: str,
    topic: str,
    size: int,
    sha256: str,
    content_type: str,
    attributes: dict[str, str] | None,
) -> int:
    empty_descriptor = LargePayloadDescriptor(
        payload_id=payload_id,
        topic=topic,
        transfer="inline",
        size=size,
        sha256=sha256,
        content_type=content_type,
        data="",
        attributes=attributes,
    )
    return len(_descriptor_bytes(empty_descriptor)) + _base64_encoded_len(size)


def _required_str(descriptor: Mapping[str, Any], key: str) -> str:
    value = descriptor.get(key)
    if not isinstance(value, str) or not value:
        raise LargePayloadError(f"large payload descriptor field {key!r} must be a string")
    return value


def _optional_str(descriptor: Mapping[str, Any], key: str) -> str | None:
    value = descriptor.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise LargePayloadError(f"large payload descriptor field {key!r} must be a string")
    return value


def _required_int(descriptor: Mapping[str, Any], key: str) -> int:
    value = descriptor.get(key)
    if not isinstance(value, int) or value < 0:
        raise LargePayloadError(
            f"large payload descriptor field {key!r} must be a non-negative integer"
        )
    return value


def _verify_payload(
    payload: bytes,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
) -> None:
    if expected_size is not None and len(payload) != expected_size:
        raise LargePayloadError(
            f"large payload size mismatch: expected {expected_size}, got {len(payload)}"
        )
    if expected_sha256 is not None:
        digest = hashlib.sha256(payload).hexdigest()
        expected = expected_sha256.strip().lower()
        if len(expected) != 64 or any(c not in _SHA256_HEX_CHARS for c in expected):
            raise LargePayloadChecksumError("large payload sha256 is not a valid hex digest")
        if not hmac.compare_digest(digest, expected):
            raise LargePayloadChecksumError("large payload checksum mismatch")
