# Copyright 2023 LiveKit, Inc.
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

import asyncio
import livekit.rtc as rtc
from livekit import agents
from livekit.plugins import core
from livekit.plugins.vad import VADPlugin
from livekit.plugins.openai import WhisperLocalTranscriber
import logging
from typing import AsyncIterator


async def stt_agent(ctx: agents.JobContext):
    logging.info("starting stt agent")
    # agent is connected to the room as a participant

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant):
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return

        asyncio.create_task(process_track(track))


async def process_track(track: rtc.Track):
    audio_stream = rtc.AudioStream(track)
    input_iterator = core.PluginIterator.create(audio_stream)
    vad_plugin = VADPlugin(
        left_padding_ms=250, silence_threshold_ms=500)
    stt_plugin = WhisperLocalTranscriber()

    vad_results = vad_plugin \
        .set_input(input_iterator) \
        .filter(lambda data: data.type == core.VADPluginResultType.FINISHED) \
        .pipe(stt_plugin) \
        .map(lambda data: data.frames) \
        .unwrap()
    stt_results = stt_plugin.start(vad_results)

    async for event in stt_results:
        if event.type == core.STTPluginEventType.ERROR:
            continue

        text = event.data.text
        asyncio.create_task(ctx.room.local_participant.publish_data(text))


async def process_stt(text_stream: AsyncIterator[str], metadata: core.PluginIterator.ResultMetadata):
    complete_stt_result = ""
    async for stt_r in text_stream:
        complete_stt_result += stt_r.text
    msg = ChatGPTMessage(
        role=ChatGPTMessageRole.user, content=complete_stt_result)
    return msg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def available_cb(job_request: agents.JobRequest):
        await job_request.accept(
            stt_agent,
            should_subscribe=lambda track_pub, _: track_pub.kind == rtc.TrackKind.KIND_AUDIO,
        )

    worker = agents.Worker(available_cb=available_cb,
                           worker_type=agents.JobType.JT_ROOM)
    agents.run_app(worker)
