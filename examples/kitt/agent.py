import asyncio
from typing import List

from livekit import agents, rtc
from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import ChatMessage, ChatRole, LLMStream
from livekit.plugins.deepgram import STT
from livekit.plugins.elevenlabs import TTS
from livekit.plugins.openai import LLM


class Kitt:
    async def start(self, job: JobContext):
        tts = TTS()
        llm = LLM()
        stt = STT()
        stt_stream = stt.stream()
        committed_messages: List[agents.llm.ChatMessage] = []

        # Publish agent mic
        source = rtc.AudioSource(44100, 1)
        track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await job.room.local_participant.publish_track(track, options)

        audio_stream_future = asyncio.Future[rtc.AudioStream]()

        def on_track_subscribed(
            track: rtc.Track,
            pub: rtc.TrackPublication,
            rp: rtc.RemoteParticipant,
        ):
            audio_stream_future.set_result(rtc.AudioStream(track))

        job.room.on("track_subscribed", on_track_subscribed)

        # Wait for user audio
        audio_stream = await audio_stream_future

        llm_stream_queue = asyncio.Queue[LLMStream]()
        latest_llm_stream: LLMStream = None
        tts_stream_queue = asyncio.Queue[agents.tts.SynthesizeStream]()
        latest_tts_stream: agents.tts.SynthesizeStream = None
        working_text = ""
        agent_speaking = False

        async def audio_stream_task():
            async for audio_frame_event in audio_stream:
                # Ignore user input while the agent is speaking
                if agent_speaking:
                    continue
                stt_stream.push_frame(audio_frame_event.frame)

        async def stt_stream_task():
            nonlocal working_text, latest_llm_stream
            async for stt_event in self._stt_stream:
                # We eagerly try to run inference to keep the latency as low as possible.
                # If we get a new transcript, we update the working text, cancel in-flight inference,
                # and run new inference.
                if stt_event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
                    working_text += stt_event.alternatives[0].text
                    working_message = ChatMessage(role=ChatRole.USER, text=working_text)
                    chat_context = agents.llm.ChatContext(
                        messages=[committed_messages] + [working_message]
                    )
                    if latest_llm_stream is not None:
                        await latest_llm_stream.aclose(wait=False)
                    latest_llm_stream = await llm.chat(chat_context)
                    await llm_stream_queue.put(latest_llm_stream)

        async def chat_stream_task():
            nonlocal latest_tts_stream
            while True:
                llm_stream = await llm_stream_queue.get()
                if llm_stream is None:
                    break
                if latest_tts_stream is not None:
                    await latest_tts_stream.aclose(wait=False)
                latest_tts_stream = tts.stream()
                async for chunk in llm_stream:
                    content = chunk.choices[0].delta.content
                    if content is None:
                        continue
                    latest_tts_stream.push_text(content)
                await latest_tts_stream.aclose()

        async def tts_stream_task():
            nonlocal agent_speaking
            while True:
                tts_stream = await tts_stream_queue.get()
                if tts_stream is None:
                    break
                agent_speaking = True
                async for event in tts_stream:
                    if event.type == agents.tts.SynthesisEventType.AUDIO:
                        await source.capture_frame(event.audio.data)
                agent_speaking = False

        async def end_session_task():
            # Flush llm_queue
            await audio_stream.aclose()
            await stt_stream.aclose(wait=False)
            while True:
                try:
                    llm_stream_queue.get_nowait()
                except asyncio.QueueEmpty:
                    llm_stream_queue.put_nowait(None)
                    break

            while True:
                try:
                    tts_stream_queue.get_nowait()
                except asyncio.QueueEmpty:
                    tts_stream_queue.put_nowait(None)
                    break

        async with asyncio.TaskGroup() as tg:
            tg.create_task(audio_stream_task)
            tg.create_task(stt_stream_task)
            tg.create_task(chat_stream_task)
            tg.create_task(tts_stream_task)
            await asyncio.sleep(120)
            tg.create_task(end_session_task)


async def entrypoint(job: JobContext):
    agent = Kitt()
    await agent.start(job)


async def request_fnc(req: JobRequest) -> None:
    await req.accept(entrypoint, auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc=request_fnc))
