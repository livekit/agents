import asyncio
import os
from dataclasses import dataclass
from itertools import cycle
from typing import Dict, Optional

from livekit import api, rtc

tasks = set()

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
class StreamMessage:
    participant_identity: str
    track_id: Optional[str]
    stream_id: str
    content: str
    final: Optional[bool] = None


class StreamPrinter:
    def __init__(self):
        self.queue = asyncio.Queue[StreamMessage | None]()
        self.running = True

        self.color_cycle = cycle(COLORS)
        self._color_map: Dict[str, str] = {}
        self._current_stream_id: str | None = None

        self.printer_task = asyncio.create_task(self.printer_loop())

    def get_color(self, identity: str) -> str:
        if identity not in self._color_map:
            self._color_map[identity] = next(self.color_cycle)
        return self._color_map[identity]

    async def printer_loop(self):
        while self.running:
            msg = await self.queue.get()
            if not msg:
                break

            if self._current_stream_id != msg.stream_id:
                # print a new line if the stream id has changed
                color = self.get_color(msg.participant_identity)
                type = "transcript" if msg.track_id else "chat"
                print(
                    f"\n{color}[{msg.participant_identity}][{type}][{msg.stream_id}]: {RESET}",
                    end="",
                    flush=True,
                )
                self._current_stream_id = msg.stream_id

            if msg.final is not None:
                print(f" [final={msg.final}]", end="", flush=True)
                self._current_stream_id = None
            else:
                print(msg.content, end="", flush=True)

    async def stop(self):
        self.running = False
        await self.queue.put(None)
        await self.printer_task


async def main(room: rtc.Room, room_name: str):
    # Create access token with the API
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

    url = os.getenv("LIVEKIT_URL")
    if not url:
        print("Please set LIVEKIT_URL environment variable")
        return

    stream_printer = StreamPrinter()

    def on_text_received(reader: rtc.TextStreamReader, participant_identity: str):
        async def _on_text_received():
            stream_id = reader.info.stream_id
            track_id = reader.info.attributes.get("lk.transcribed_track_id", None)

            async for chunk in reader:
                await stream_printer.queue.put(
                    StreamMessage(participant_identity, track_id, stream_id, content=chunk)
                )

            final = reader.info.attributes.get("lk.transcription_final", "null")
            await stream_printer.queue.put(
                StreamMessage(participant_identity, track_id, stream_id, content="", final=final)
            )

        task = asyncio.create_task(_on_text_received())
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    # Connect to the room
    try:
        print("Connecting to LiveKit room...")
        await room.connect(url, token)
        print(f"Connected to room: {room.name}")

        # Register handler for text messages on lk.chat topic
        room.register_text_stream_handler("lk.chat", on_text_received)

        print("Listening for chat messages. Press Ctrl+C to exit...")
        # Instead of running the loop forever here, we'll use an Event to keep the connection alive
        stop_event = asyncio.Event()
        await stop_event.wait()

    except KeyboardInterrupt:
        print("\nDisconnecting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await stream_printer.stop()  # Clean up the printer
        await room.disconnect()
        print("Disconnected from room")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ds_chat_test.py <room-name>")
        sys.exit(1)

    room_name = sys.argv[1]

    # Use asyncio.new_event_loop() instead of get_event_loop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    room = rtc.Room(loop=loop)

    async def cleanup():
        await room.disconnect()
        loop.stop()

    try:
        loop.run_until_complete(main(room, room_name))
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(room.disconnect())
        loop.close()
