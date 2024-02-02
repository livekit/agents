# Copyright 2024 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import asyncio
import json
import logging
import mediapipe as mp
import numpy as np

from livekit import agents, rtc
from PIL import Image, ImageDraw

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class MediaPipe:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        instance = MediaPipe(ctx)
        await instance.start()

    def __init__(self, ctx: agents.JobContext):
        self.ctx: agents.JobContext = ctx
        self.options = FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path="./blaze_face_short_range.tflite"
            ),
            running_mode=VisionRunningMode.VIDEO,
        )

    async def start(self):
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            self.ctx.create_task(self.process_track(track))

        self.ctx.room.on("track_subscribed", on_track_subscribed)

    async def process_track(self, track: rtc.VideoTrack):
        video_stream = rtc.VideoStream(track)
        frame_timestamp_ms = int(time.time() * 1000)
        with FaceDetector.create_from_options(self.options) as detector:
            async for frame in video_stream:
                frame_timestamp_ms = int(time.time() * 1000)
                argb_frame = rtc.ArgbFrame.create(
                    format=rtc.VideoFormatType.FORMAT_RGBA,
                    width=frame.buffer.width,
                    height=frame.buffer.height,
                )
                frame.buffer.to_argb(dst=argb_frame)
                pil_image = Image.frombytes(
                    "RGBA", (argb_frame.width, argb_frame.height), argb_frame.data
                )
                pil_image = pil_image.resize((128, 128))
                pil_image = pil_image.convert("RGB")
                np_array = np.asarray(pil_image)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np_array,
                )
                face_detector_result = detector.detect_for_video(
                    mp_image, frame_timestamp_ms
                )
                print(face_detector_result)

    def update_state(self, state: str):
        metadata = json.dumps({"agent_state": state})
        self.ctx.create_task(self.ctx.room.local_participant.update_metadata(metadata))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Accepting job for MediaPipe")

        await job_request.accept(
            MediaPipe.create,
            identity="mediapipe_agent",
            auto_subscribe=agents.AutoSubscribe.VIDEO_ONLY,
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(job_request_cb)
    agents.run_app(worker)
