"""Send application payloads that may be too large for a data packet.

This example uses a data-channel descriptor for control metadata. Small payloads
are embedded in that descriptor. Larger payloads are sent through a byte stream,
and the descriptor points receivers to the stream topic/name.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from dotenv import load_dotenv

from livekit import api, rtc
from livekit.agents.utils import aio
from livekit.agents.utils.large_payload import (
    ATTR_PAYLOAD_SHA256,
    ATTR_PAYLOAD_SIZE,
    ATTR_PAYLOAD_TOPIC,
    LargePayloadError,
    parse_large_payload_descriptor,
    publish_large_payload,
    read_large_payload_stream,
)

load_dotenv()

logger = logging.getLogger("large-payload-example")
logging.basicConfig(level=logging.INFO)

CONTROL_TOPIC = "example.payload"
STREAM_TOPIC = "example.payload.bytes"
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024


class PayloadReceiver:
    def __init__(self, room: rtc.Room) -> None:
        self._room = room
        self._stream_tasks: set[asyncio.Task[None]] = set()

    def register(self) -> None:
        self._room.register_byte_stream_handler(STREAM_TOPIC, self._on_byte_stream)

        @self._room.on("data_received")
        def _on_data(packet: rtc.DataPacket) -> None:
            if packet.topic != CONTROL_TOPIC:
                return
            try:
                descriptor = parse_large_payload_descriptor(packet.data)
            except LargePayloadError:
                logger.exception("invalid payload descriptor")
                return

            if descriptor.transfer == "inline":
                payload = descriptor.decode_inline_payload()
                logger.info("received inline payload: %d bytes", len(payload))
            else:
                logger.info(
                    "payload %s is carried on byte stream topic %s",
                    descriptor.payload_id,
                    descriptor.stream_topic,
                )

    def _on_byte_stream(self, reader: rtc.ByteStreamReader, participant_identity: str) -> None:
        task = asyncio.create_task(self._read_stream(reader, participant_identity))
        self._stream_tasks.add(task)
        task.add_done_callback(self._stream_tasks.discard)

    async def aclose(self) -> None:
        await aio.cancel_and_wait(*self._stream_tasks)

    async def _read_stream(self, reader: rtc.ByteStreamReader, participant_identity: str) -> None:
        try:
            attributes = reader.info.attributes or {}
            expected_size = attributes.get(ATTR_PAYLOAD_SIZE)
            payload = await read_large_payload_stream(
                reader,
                max_bytes=MAX_PAYLOAD_BYTES,
                expected_size=int(expected_size) if expected_size else None,
                expected_sha256=attributes.get(ATTR_PAYLOAD_SHA256),
            )
            logger.info(
                "received streamed payload from %s on %s: %d bytes",
                participant_identity,
                attributes.get(ATTR_PAYLOAD_TOPIC, ""),
                len(payload),
            )
        except Exception:
            logger.exception("failed to read streamed payload from %s", participant_identity)


async def publish_example_payload(room: rtc.Room) -> None:
    payload = json.dumps(
        {
            "kind": "document",
            "title": "Quarterly update",
            "sections": [{"heading": "Summary", "body": "..." * 8_000}],
        },
        separators=(",", ":"),
    )

    info = await publish_large_payload(
        room.local_participant,
        payload,
        topic=CONTROL_TOPIC,
        stream_topic=STREAM_TOPIC,
        content_type="application/json",
        attributes={"kind": "document"},
    )
    logger.info(
        "published %s payload descriptor: %d bytes",
        info.descriptor.transfer,
        info.descriptor_bytes,
    )


async def main() -> None:
    room_name = os.environ.get("LIVEKIT_ROOM", "large-payload-example")
    identity = os.environ.get("LIVEKIT_IDENTITY", "payload-demo")
    url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_publish_data=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )

    room = rtc.Room()
    receiver = PayloadReceiver(room)
    receiver.register()
    await room.connect(url, token)
    try:
        await publish_example_payload(room)
    finally:
        await receiver.aclose()
        await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
