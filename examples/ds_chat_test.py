import asyncio
import os
import signal

from livekit import api, rtc

tasks = set()

def on_text_received(reader: rtc.TextStreamReader, participant_identity: str):
    """Callback for when text is received on the data stream"""
    async def _on_text_received():
        text = await reader.read_all()
        stream_id = reader.info.stream_id
        final = reader.info.attributes.get("lk.transcription_final", "null")
        print(f"[{participant_identity}][{stream_id}][final={final}]: '{text.replace('\n', '\\n')}'")

    task = asyncio.create_task(_on_text_received())
    tasks.add(task)
    task.add_done_callback(tasks.discard)


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
