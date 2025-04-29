import asyncio
import os
import sys

from dotenv import load_dotenv

from livekit import api, rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, DataStreamAudioReceiver

load_dotenv()


## This is the audio receiver. It receives audio from the audio sender and streams it to the room.
## It also handles the interruption and playback notification.


RECEIVER_IDENTITY = "agent-receiver"


async def main(room_name: str):
    url = os.getenv("LIVEKIT_URL")
    if not url:
        print("Please set LIVEKIT_URL environment variable")
        return

    room = rtc.Room()
    token = (
        api.AccessToken()
        .with_identity(RECEIVER_IDENTITY)
        .with_name("Agent Receiver")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    print(f"Connecting to room: {room_name}")
    await room.connect(url, token)

    # read audio from the datastream
    audio_receiver = DataStreamAudioReceiver(room=room)
    await audio_receiver.start()  # wait for the sender to join the room
    print(f"Audio receiver connected to {audio_receiver._remote_participant.identity}")

    audio_source = rtc.AudioSource(sample_rate=24000, num_channels=1, queue_size_ms=10_000)
    track = rtc.LocalAudioTrack.create_audio_track("audio_receiver_output", audio_source)
    await room.local_participant.publish_track(track)

    # stream audio to the room and handle the interruption and playback notification
    pushed_duration = 0
    interrupted_event = asyncio.Event()

    def _on_clear_buffer():
        if not pushed_duration:
            return
        print("clear buffer called")
        interrupted_event.set()

    async def _wait_for_playout():
        nonlocal pushed_duration

        wait_for_interruption = asyncio.create_task(interrupted_event.wait())
        wait_for_playout = asyncio.create_task(audio_source.wait_for_playout())
        await asyncio.wait(
            [wait_for_playout, wait_for_interruption],
            return_when=asyncio.FIRST_COMPLETED,
        )

        interrupted = wait_for_interruption.done()
        played_duration = pushed_duration
        if interrupted:
            played_duration = max(pushed_duration - audio_source.queued_duration, 0)
            audio_source.clear_queue()
            wait_for_playout.cancel()
        else:
            wait_for_interruption.cancel()

        interrupted_event.clear()
        pushed_duration = 0
        print(f"playback finished: {played_duration} interrupted: {interrupted}")
        await audio_receiver.notify_playback_finished(
            playback_position=played_duration, interrupted=interrupted
        )

    audio_receiver.on("clear_buffer", _on_clear_buffer)
    async for frame in audio_receiver:
        if isinstance(frame, AudioSegmentEnd):
            print("audio segment end")
            await _wait_for_playout()
            continue

        await audio_source.capture_frame(frame)
        if frame.duration and not pushed_duration:
            print("==========")
            print("new audio segment start")
        pushed_duration += frame.duration


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python audio_receiver.py <room-name>")
        sys.exit(1)

    room_name = sys.argv[1]
    asyncio.run(main(room_name=room_name))
