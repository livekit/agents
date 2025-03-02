import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import AsyncIterator, Optional

import cv2
import numpy as np
from bithuman_runtime import AsyncBithumanRuntime, BithumanRuntime
from livekit import rtc
from livekit.agents import utils
from livekit.agents.pipeline.avatar import AvatarRunner, MediaOptions
from livekit.agents.pipeline.datastream_io import AudioFlushSentinel

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger("avatar-example")


class MyVideoGenerator:
    def __init__(self, avatar_model: str, token: str):
        _runtime = BithumanRuntime(token=token)
        _runtime.set_avatar_model(avatar_model)

        self.runtime = AsyncBithumanRuntime(_runtime)

    @property
    def video_resolution(self) -> tuple[int, int]:
        frame = self.runtime._runtime.get_first_frame()
        if frame is None:
            raise ValueError("No frame found")
        return frame.shape[1], frame.shape[0]

    @property
    def video_fps(self) -> int:
        return self.runtime._runtime.settings.FPS

    @property
    def audio_sample_rate(self) -> int:
        return self.runtime._runtime.settings.INPUT_SAMPLE_RATE

    @utils.log_exceptions(logger=logger)
    async def push_audio(self, frame: rtc.AudioFrame | AudioFlushSentinel) -> None:
        if isinstance(frame, AudioFlushSentinel):
            await self.runtime.flush()
            return
        await self.runtime.push_audio(bytes(frame.data), frame.sample_rate, last_chunk=False)

    def clear_buffer(self) -> None:
        self.runtime.clear_buffer()

    async def stream(
        self,
    ) -> AsyncIterator[tuple[rtc.VideoFrame, Optional[rtc.AudioFrame]] | AudioFlushSentinel]:
        def create_video_frame(image: np.ndarray) -> rtc.VideoFrame:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            return rtc.VideoFrame(
                width=image.shape[1],
                height=image.shape[0],
                type=rtc.VideoBufferType.RGBA,
                data=image.tobytes(),
            )

        async for frame in self.runtime.stream():
            video_frame: rtc.VideoFrame | None = None
            if frame.bgr_image is not None:
                video_frame = create_video_frame(frame.bgr_image)

            audio_frame: rtc.AudioFrame | None = None
            if frame.audio_chunk is not None:
                audio_frame = rtc.AudioFrame(
                    data=frame.audio_chunk.bytes,
                    sample_rate=frame.audio_chunk.sample_rate,
                    num_channels=1,
                    samples_per_channel=len(frame.audio_chunk.array),
                )
            if video_frame is not None:
                yield video_frame, audio_frame

            if frame.end_of_speech:
                yield AudioFlushSentinel()

    async def stop(self) -> None:
        await self.runtime.stop()


async def main(room: rtc.Room, avatar_model: str, bithuman_token: str):
    """Main application logic for the avatar worker"""
    runner: AvatarRunner | None = None
    stop_event = asyncio.Event()

    try:
        # Initialize and start worker
        video_generator = MyVideoGenerator(avatar_model, bithuman_token)
        output_width, output_height = video_generator.video_resolution
        media_options = MediaOptions(
            video_width=output_width,
            video_height=output_height,
            video_fps=video_generator.video_fps,
            audio_sample_rate=video_generator.audio_sample_rate,
            audio_channels=1,
        )
        logger.info("Media options: %s", media_options)
        runner = AvatarRunner(room, video_generator=video_generator, media_options=media_options)
        await runner.start()

        # Set up disconnect handler
        async def handle_disconnect(participant: rtc.RemoteParticipant):
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                logging.info("Agent %s disconnected, stopping worker...", participant.identity)
                await video_generator.stop()
                stop_event.set()

        room.on(
            "participant_disconnected",
            lambda p: asyncio.create_task(handle_disconnect(p)),
        )
        room.on("disconnected", lambda _: stop_event.set())

        room.register_text_stream_handler("lk.chat", lambda *args: None)

        # Wait until stopped
        await stop_event.wait()

    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise
    finally:
        if runner:
            await runner.aclose()


async def run_service(url: str, token: str, avatar_model: str, bithuman_token: str):
    """Run the avatar worker service"""
    room = rtc.Room()
    try:
        # Connect to LiveKit room
        logging.info("Connecting to %s", url)
        await room.connect(url, token)
        logging.info("Connected to room %s", room.name)

        # Run main application logic
        await main(room, avatar_model, bithuman_token)
    except rtc.ConnectError as e:
        logging.error("Failed to connect to room: %s", e)
        raise
    finally:
        await room.disconnect()


if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    def parse_args():
        """Parse command line arguments"""
        parser = ArgumentParser()
        parser.add_argument("--url", required=True, help="LiveKit server URL")
        parser.add_argument("--token", required=True, help="Token for joining room")
        parser.add_argument("--room", help="Room name")
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Log level",
        )
        parser.add_argument(
            "--avatar-model",
            help="Avatar model path",
            default=os.getenv("BITHUMAN_AVATAR_MODEL"),
        )
        parser.add_argument(
            "--bithuman-token",
            help="Token for joining room",
            default=os.getenv("BITHUMAN_RUNTIME_TOKEN"),
        )
        return parser.parse_args()

    def setup_logging(room: Optional[str], level: str):
        """Set up logging configuration"""
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        if room:
            log_format = f"[{room}] {log_format}"

        logging.basicConfig(level=getattr(logging, level.upper()), format=log_format)

    args = parse_args()
    setup_logging(args.room, args.log_level)
    try:
        asyncio.run(run_service(args.url, args.token, args.avatar_model, args.bithuman_token))
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logging.error("Fatal error: %s", e)
        sys.exit(1)
    finally:
        logging.info("Shutting down...")
