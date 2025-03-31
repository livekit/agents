import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import dotenv
from simli import SimliClient, SimliConfig

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioReceiver,
    VideoGenerator,
)

dotenv.load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger("avatar-example")


@dataclass
class SimliAudioOptions:
    # default values of simli
    in_sample_rate: int = 16000
    in_channels: int = 1
    out_sample_rate: int = 48000
    out_channels: int = 2


class SimliVideoGenerator(VideoGenerator):
    def __init__(self, avatar_options: AvatarOptions):
        self._options = avatar_options
        self._simli_audio_options = SimliAudioOptions()

        self._input_resampler: Optional[rtc.AudioResampler] = None
        self._output_resampler: Optional[rtc.AudioResampler] = None
        if (
            self._options.audio_sample_rate != self._simli_audio_options.out_sample_rate
            or self._options.audio_channels != self._simli_audio_options.out_channels
        ):
            self._output_resampler = rtc.AudioResampler(
                input_rate=self._simli_audio_options.out_sample_rate,
                output_rate=self._options.audio_sample_rate,
                num_channels=self._options.audio_channels,
            )

        self._data_ch = asyncio.Queue[rtc.AudioFrame | rtc.VideoFrame | AudioSegmentEnd]()
        self._recv_audio_atask: asyncio.Task = None
        self._recv_video_atask: asyncio.Task = None
        self._first_frame_fut = asyncio.Future[None]()

    @utils.log_exceptions(logger=logger)
    async def start(self, api_key: str, face_id: str):
        self.config = SimliConfig(api_key, face_id, maxIdleTime=100)
        self._simli_client = SimliClient(
            self.config,
            # simliURL="http://127.0.0.1:8892",  # used for debugging on simli servers or to connect to a relay server
        )
        await self._simli_client.Initialize()

        self._recv_audio_atask = asyncio.create_task(self._recv_audio_task())
        self._recv_video_atask = asyncio.create_task(self._recv_video_task())

        await self._simli_client.sendSilence()

    @utils.log_exceptions(logger=logger)
    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            if self._input_resampler is not None:
                for resampled_frame in self._input_resampler.flush():
                    await self._simli_client.send(bytes(resampled_frame.data))
            # send the end of the audio segment sentinel if needed
            return

        if not self._input_resampler and (
            frame.sample_rate != self._simli_audio_options.in_sample_rate
            or frame.num_channels != self._simli_audio_options.in_channels
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=self._simli_audio_options.in_sample_rate,
                num_channels=self._simli_audio_options.in_channels,
            )
        if self._input_resampler is not None:
            for resampled_frame in self._input_resampler.push(frame):
                await self._simli_client.send(bytes(resampled_frame.data))
            return
        else:
            await self._simli_client.send(bytes(frame.data))

    async def clear_buffer(self) -> None:
        await self._simli_client.clearBuffer()

    def __aiter__(self):
        return self._stream_impl()

    async def _stream_impl(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        while True:
            frame = await self._data_ch.get()
            yield frame
            self._data_ch.task_done()

    async def wait_for_first_frame(self):
        await self._first_frame_fut

    @utils.log_exceptions(logger=logger)
    async def _recv_audio_task(self):
        async for frame in self._simli_client.getAudioStreamIterator():
            data = frame.to_ndarray()
            audio_frame = rtc.AudioFrame(
                data=data.tobytes(),
                sample_rate=frame.sample_rate,
                num_channels=2,
                samples_per_channel=data.size // 2,
            )
            if self._output_resampler is not None:
                for resampled_frame in self._output_resampler.push(audio_frame):
                    await self._data_ch.put(resampled_frame)
            else:
                await self._data_ch.put(audio_frame)

        if self._output_resampler is not None:
            for resampled_frame in self._output_resampler.flush():
                await self._data_ch.put(resampled_frame)

    @utils.log_exceptions(logger=logger)
    async def _recv_video_task(self):
        async for videoFrame in self._simli_client.getVideoStreamIterator("yuva420p"):
            if not self._first_frame_fut.done():
                self._first_frame_fut.set_result(None)
            await self._data_ch.put(
                rtc.VideoFrame(
                    width=videoFrame.width,
                    height=videoFrame.height,
                    type=rtc.VideoBufferType.I420A,
                    data=videoFrame.to_ndarray().tobytes(),
                )
            )

    async def aclose(self):
        await self._simli_client.stop()
        if self._recv_audio_atask:
            self._recv_audio_atask.cancel()
        if self._recv_video_atask:
            self._recv_video_atask.cancel()


async def main(room: rtc.Room, simli_api_key, simli_face_id):
    """Main application logic for the avatar worker"""
    avatar_runner = None
    stop_event = asyncio.Event()

    assert simli_api_key is not None, "SIMLI_API_KEY is not set"
    assert simli_face_id is not None, "SIMLI_FACE_ID is not set"

    try:
        # Initialize and start worker
        avatar_options = AvatarOptions(
            video_width=512,
            video_height=512,
            video_fps=30,
            audio_sample_rate=48000,
            audio_channels=2,
        )
        video_generator = SimliVideoGenerator(avatar_options)

        logger.info("starting simli client")
        await video_generator.start(simli_api_key, simli_face_id)

        logger.info("waiting for first frame from simli")
        await video_generator.wait_for_first_frame()
        logger.info("first frame received")

        avatar_runner = AvatarRunner(
            room,
            audio_recv=DataStreamAudioReceiver(room),
            video_gen=video_generator,
            options=avatar_options,
        )
        await avatar_runner.start()

        # Set up disconnect handler
        async def handle_disconnect(participant: rtc.RemoteParticipant):
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                logging.info("Agent %s disconnected, stopping worker...", participant.identity)
                stop_event.set()

        room.on(
            "participant_disconnected",
            lambda p: asyncio.create_task(handle_disconnect(p)),
        )
        room.on("disconnected", lambda _: stop_event.set())

        # Wait until stopped
        await stop_event.wait()

    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise
    finally:
        await avatar_runner.aclose()
        await video_generator.aclose()


async def run_service(url: str, token: str, apikey, faceid):
    """Run the avatar worker service"""
    room = rtc.Room()
    try:
        # Connect to LiveKit room
        logging.info("Connecting to %s", url)
        await room.connect(url, token)
        logging.info("Connected to room %s", room.name)

        # Run main application logic
        await main(room, apikey, faceid)
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
            "--simli-api-key",
            default=os.getenv("SIMLI_API_KEY"),
            help="Simli API key, get it from https://app.simli.com",
        )
        parser.add_argument(
            "--simli-face-id",
            default=os.getenv("SIMLI_FACE_ID"),
            help="Simli face id, create your own at https://app.simli.com or get a premade one from https://docs.simli.com/api-reference/available-faces",
        )
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Log level",
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
        asyncio.run(run_service(args.url, args.token, args.simli_api_key, args.simli_face_id))
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logging.error("Fatal error: %s", e)
        sys.exit(1)
    finally:
        logging.info("Shutting down...")
