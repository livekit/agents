import asyncio
import json
import logging
from typing import Optional

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import AgentServer, AutoSubscribe, JobContext, cli
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT
from livekit.plugins import inworld

load_dotenv()

logger = logging.getLogger("inworld-tts-demo")
logger.setLevel(logging.INFO)

server = AgentServer()


@server.rtc_session()
async def entrypoint(job: JobContext):
    logger.info("starting tts example agent")

    tts = inworld.TTS(
        # voice="Alex",  # Voice ID (or custom cloned voice ID)
        # timestamp_type="WORD",  # CHARACTER or WORD
        # text_normalization="ON",  # ON or OFF
    )

    source = rtc.AudioSource(tts.sample_rate, tts.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    await job.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
    publication = await job.room.local_participant.publish_track(track, options)
    await publication.wait_for_subscription()

    # --- Example 1: Using synthesize() (HTTP streaming) ---
    text = "Hello from Inworld. I hope you are having a spectacular and /ˈwʌn.dɚ.fəl/ day."
    logger.info(f'synthesizing (HTTP): "{text}"')
    async for audio in tts.synthesize(text):
        # Print timestamp information if available
        timed_strings = audio.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])
        for ts in timed_strings:
            start = (
                f"{ts.start_time:.3f}s" if hasattr(ts, "start_time") and ts.start_time else "N/A"
            )
            end = f"{ts.end_time:.3f}s" if hasattr(ts, "end_time") and ts.end_time else "N/A"
            logger.info(f"  [{start} - {end}] {ts}")

        await source.capture_frame(audio.frame)

    logger.info("HTTP synthesis complete")

    await asyncio.sleep(1)

    # --- Example 2: Using stream() (WebSocket streaming) ---
    streamed_text = (
        "This is an example using WebSocket streaming for lower latency real-time synthesis."
    )
    logger.info(f'streaming (WebSocket): "{streamed_text}"')

    stream = tts.stream()

    # Simulate streaming input (e.g., from an LLM) by pushing chunks
    # The TTS internally buffers and tokenizes these into sentences
    chunks = [
        "This is an example ",
        "using WebSocket streaming ",
        "for lower latency ",
        "real-time synthesis.",
    ]

    for chunk in chunks:
        logger.debug(f"pushing chunk: {chunk!r}")
        stream.push_text(chunk)
        await asyncio.sleep(0.1)  # Simulate generation delay

    stream.flush()
    stream.end_input()

    # Consume streamed audio
    playout_q: asyncio.Queue[Optional[rtc.AudioFrame]] = asyncio.Queue()

    async def _synth_task():
        async for ev in stream:
            # Print timestamp information if available
            timed_strings = ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])
            for ts in timed_strings:
                start = (
                    f"{ts.start_time:.3f}s"
                    if hasattr(ts, "start_time") and ts.start_time
                    else "N/A"
                )
                end = f"{ts.end_time:.3f}s" if hasattr(ts, "end_time") and ts.end_time else "N/A"
                logger.info(f"  [{start} - {end}] {ts}")

            playout_q.put_nowait(ev.frame)

        playout_q.put_nowait(None)

    async def _playout_task():
        while True:
            frame = await playout_q.get()
            if frame is None:
                break
            await source.capture_frame(frame)

    synth_task = asyncio.create_task(_synth_task())
    playout_task = asyncio.create_task(_playout_task())

    await asyncio.gather(synth_task, playout_task)
    await stream.aclose()

    logger.info("WebSocket streaming complete")
    # List available voices
    try:
        voices = await tts.list_voices()
        logger.info(f"[Inworld TTS] {len(voices)} voices available in this workspace")
        if voices:
            logger.info(
                f"[Inworld TTS] Logging information for first voice: {json.dumps(voices[0], indent=2)}"
            )
    except Exception as e:
        logger.error(f"[Inworld TTS] Failed to list voices: {e}")


if __name__ == "__main__":
    cli.run_app(server)
