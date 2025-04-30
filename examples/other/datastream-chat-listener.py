import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from itertools import cycle
from typing import Optional

from dotenv import load_dotenv

from livekit import api, rtc
from livekit.agents import utils
from livekit.agents.types import (
    ATTRIBUTE_TRANSCRIPTION_FINAL,
    ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID,
    ATTRIBUTE_TRANSCRIPTION_TRACK_ID,
    TOPIC_TRANSCRIPTION,
)

logger = logging.getLogger("text-only")
logger.setLevel(logging.INFO)

load_dotenv()

## This example demonstrates a text-only agent.
## Send text input using TextStream to topic `lk.chat` (https://docs.livekit.io/home/client/data/text-streams)
## The agent output is sent through TextStream to the `lk.transcription` topic


# Add color constants
COLORS = [
    "\033[36m",  # Cyan
    "\033[32m",  # Green
    "\033[33m",  # Yellow
    "\033[35m",  # Magenta
    "\033[34m",  # Blue
]
RESET = "\033[0m"  # Reset color


@dataclass
class Chunk:
    stream_id: str
    participant_identity: str
    track_id: Optional[str]
    segment_id: str
    content: str
    final: Optional[bool] = None


class TextStreamPrinter:
    def __init__(self):
        self._text_chunk_queue = asyncio.Queue[Chunk | None]()
        self.running = True

        self._color_cycle = cycle(COLORS)
        self._color_map: dict[str, str] = {}
        self._current_segment_id: str | None = None
        # track the stream id for each segment id, overwrite if new stream with the same segment id
        self._segment_to_stream: dict[str, str] = {}

        self._tasks = set[asyncio.Task]()
        self._main_atask = asyncio.create_task(self._main_task())

    def _get_color(self, identity: str) -> str:
        if identity not in self._color_map:
            self._color_map[identity] = next(self._color_cycle)
        return self._color_map[identity]

    async def _main_task(self):
        header = "[{participant_identity}][{type}][{segment_id}][{overwrite}]"

        while self.running:
            chunk = await self._text_chunk_queue.get()
            if chunk is None:
                break

            color = self._get_color(chunk.participant_identity)
            if self._current_segment_id != chunk.segment_id:
                # in cli we don't actually overwrite the line, just add a flag
                overwrite = (
                    "overwrite"
                    if chunk.segment_id in self._segment_to_stream
                    and self._segment_to_stream[chunk.segment_id] != chunk.stream_id
                    else "new"
                )
                # type = "transcript" if chunk.track_id else "chat"
                type = str(chunk.track_id)

                # header: [participant_identity][type][segment_id]
                line_header = header.format(
                    participant_identity=chunk.participant_identity,
                    type=type,
                    segment_id=chunk.segment_id,
                    overwrite=overwrite,
                )
                print(f"\n{color}{line_header}:{RESET} ", end="", flush=True)
                self._current_segment_id = chunk.segment_id

            if chunk.final is not None:
                print(f" {color}[final={chunk.final}]{RESET}", end="", flush=True)
                self._current_segment_id = None
            else:
                print(chunk.content, end="", flush=True)

            self._segment_to_stream[chunk.segment_id] = chunk.stream_id

    def on_text_received(self, reader: rtc.TextStreamReader, participant_identity: str):
        async def _on_text_received():
            stream_id = reader.info.stream_id
            segment_id = reader.info.attributes.get(ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID, None)
            # new stream with the same segment_id should overwrite the previous one
            if not segment_id:
                logger.warning("No segment id found for text stream")
                return

            track_id = reader.info.attributes.get(ATTRIBUTE_TRANSCRIPTION_TRACK_ID, None)
            async for chunk in reader:
                await self._text_chunk_queue.put(
                    Chunk(stream_id, participant_identity, track_id, segment_id, content=chunk)
                )

            # update the final flag
            final = reader.info.attributes.get(ATTRIBUTE_TRANSCRIPTION_FINAL, "null")
            await self._text_chunk_queue.put(
                Chunk(
                    stream_id, participant_identity, track_id, segment_id, content="", final=final
                )
            )

        task = asyncio.create_task(_on_text_received())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self):
        self.running = False
        await self._text_chunk_queue.put(None)
        await self._main_atask
        await utils.aio.cancel_and_wait(self._tasks)


async def main(room_name: str):
    url = os.getenv("LIVEKIT_URL")
    if not url:
        print("Please set LIVEKIT_URL environment variable")
        return

    room = rtc.Room()
    token = (
        api.AccessToken()
        .with_identity("chat-listener")
        .with_name("Chat Listener")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
            )
        )
        .to_jwt()
    )
    print(f"Connecting to room: {room_name}")
    await room.connect(url, token)

    stop_event = asyncio.Event()

    try:
        text_printer = TextStreamPrinter()
        room.register_text_stream_handler(
            topic=TOPIC_TRANSCRIPTION, handler=text_printer.on_text_received
        )
        print("Listening for chat messages. Press Ctrl+C to exit...")

        await stop_event.wait()  # run forever
    finally:
        await text_printer.aclose()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python datastream-chat-listener.py <room-name>")
        sys.exit(1)

    room_name = sys.argv[1]
    asyncio.run(main(room_name=room_name))
