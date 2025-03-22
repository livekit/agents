"""
LiveKit agent that connects to a room and performs visual moderation on the video
of all participants using the Visual Content Moderation model from Hive
(https://docs.thehive.ai/docs/visual-content-moderation#visual-content-moderation).

The agent periodically sends a frame from the participant's video to Hive's API
for a moderation check. If the results of that check show a confidence score
of 0.9 or higher for any of the positive classes, it logs the result and adds a
message to the room's chat. This can easily be extended to take additional
actions like removing a participant or ending a livestream, etc.
"""

import asyncio
import logging
import os
import time
from io import BytesIO

import aiohttp
from dotenv import load_dotenv
from hive_data_classes import HiveResponse, from_dict
from PIL import Image

from livekit import agents, rtc

load_dotenv()

MOD_FRAME_INTERVAL = 5.0  # check 1 frame every 5 seconds
"""
How often to check a frame (in seconds)
"""

HIVE_HEADERS = {
    "Authorization": f"Token {os.getenv('HIVE_API_KEY')}",
    "accept": "application/json",
}
"""
The default headers included with every request to thehive.ai
"""

CONFIDENCE_THRESHOLD = 0.9
"""
THe threshold level for scores returned by thehive.ai.  See details in this doc:
https://docs.thehive.ai/docs/visual-content-moderation#choosing-thresholds-for-visual-moderation
"""


logger = logging.getLogger("hive-moderation-agent")
logger.setLevel(logging.INFO)


async def request_fnc(req: agents.JobRequest):
    """
    The request handler for the agent.  We use this to set the name of the
    agent that is displayed to users
    """
    # accept the job request and name the agent participant so users know what this is
    await req.accept(
        name="Moderator",
        identity="hive-moderator",
    )


async def entrypoint(ctx: agents.JobContext):
    """
    The entrypoint of the agent.  This is called every time the moderator
    agent joins a room.
    """

    # connect to the room and automatically subscribe to all participants' video
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.VIDEO_ONLY)
    chat = rtc.ChatManager(ctx.room)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        _publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """
        Event handler for video tracks.  We automatically subscribe to all video
        tracks when a participant joins the room.  This event is triggered
        once we have completed subscription to that video track.
        This creates a backgrond task to process frames from each track
        """
        asyncio.create_task(process_track(participant, track))

    async def process_track(participant: rtc.RemoteParticipant, track: rtc.VideoTrack):
        """
        This function is running in a background task once for each video track
        (i.e., once for each participant).  It handles processing a frame
        from the video once every MOD_FRAME INTERVAL seconds.
        """

        video_stream = rtc.VideoStream(track)
        last_processed_time = 0
        async for frame in video_stream:
            current_time = time.time()
            if (current_time - last_processed_time) >= MOD_FRAME_INTERVAL:
                last_processed_time = current_time
                await check_frame(participant, frame)

    async def check_frame(participant: rtc.RemoteParticipant, frame: rtc.VideoFrame):
        """
        Uses thehive.ai API to check the frame for any classifications we care about
        """

        # get the current frame and convert to png format
        argb_frame = frame.frame.convert(rtc.VideoBufferType.RGBA)
        image = Image.frombytes("RGBA", (argb_frame.width, argb_frame.height), argb_frame.data)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)  # reset buffer position to beginning after writing

        data = aiohttp.FormData()
        data.add_field("image", buffer, filename="image.png", content_type="image/png")

        # submit the image to Hive
        logger.info("submitting image to hive")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.thehive.ai/api/v2/task/sync",
                headers=HIVE_HEADERS,
                data=data,
            ) as response:
                response.raise_for_status()
                response_dict = await response.json()
                hive_response: HiveResponse = from_dict(HiveResponse, response_dict)
                if (
                    hive_response.code == 200
                    and len(hive_response.status) > 0
                    and len(hive_response.status[0].response.output) > 0
                ):
                    results = hive_response.status[0].response.output[0].classes
                    # filter to anything with a confidence score > threshold
                    for mod_class in results:
                        if mod_class.class_[0:4] == "yes_":
                            # TODO: should also include "general_nsfw" class
                            if mod_class.score >= CONFIDENCE_THRESHOLD:
                                class_name = mod_class.class_[4:]
                                message = f'FOUND {class_name} for participant "{participant.identity}" (confidence score: {mod_class.score:0.3f})'  # noqa: E501
                                logger.info(message)
                                await chat.send_message(message)

    await ctx.wait_for_participant()
    await chat.send_message(
        "I'm a moderation agent,"
        "I will detect and notify you of all inappropriate material in your video stream"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    agents.cli.run_app(agents.WorkerOptions(entrypoint, request_fnc=request_fnc))
