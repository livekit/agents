import asyncio
import logging
from typing import AsyncIterable, Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    transcription,
    cli,
    llm,
    stt,
    tokenize,
    tts,
)
from livekit.plugins import deepgram, elevenlabs, openai, silero

load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

word_tokenizer_without_punctuation: tokenize.WordTokenizer = (
    tokenize.basic.WordTokenizer(ignore_punctuation=True)
)

TRIGGER_PHRASE = "Hi Bob!"
trigger_phrase_words = word_tokenizer_without_punctuation.tokenize(text=TRIGGER_PHRASE)


async def _playout_task(
    playout_q: asyncio.Queue, audio_source: rtc.AudioSource
) -> None:
    # Playout audio frames from the queue to the audio source
    while True:
        frame = await playout_q.get()
        if frame is None:
            break

        await audio_source.capture_frame(frame)


async def _respond_to_user(
    stt_stream: stt.SpeechStream,
    tts: tts.TTS,
    agent_audio_source: rtc.AudioSource,
    local_llm: llm.LLM,
    llm_stream: llm.LLMStream,
    stt_forwarder: transcription.STTSegmentsForwarder,
):
    playout_q = asyncio.Queue[Optional[rtc.AudioFrame]]()
    tts_stream = tts.stream()

    async def _synth_task():
        async for ev in tts_stream:
            playout_q.put_nowait(ev.frame)

        playout_q.put_nowait(None)

    synth_task = asyncio.create_task(_synth_task())
    playout_task = asyncio.create_task(_playout_task(playout_q, agent_audio_source))

    async for ev in stt_stream:
        stt_forwarder.update(ev)
        if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            new_transcribed_text_words = word_tokenizer_without_punctuation.tokenize(
                text=ev.alternatives[0].text
            )
            for i in range(len(trigger_phrase_words)):
                if (
                    len(new_transcribed_text_words) < len(trigger_phrase_words)
                    or trigger_phrase_words[i].lower()
                    != new_transcribed_text_words[i].lower()
                ):
                    # ignore user speech by not sending it to LLM
                    break
                elif i == len(trigger_phrase_words) - 1:
                    # trigger phrase is validated
                    new_chat_context = llm_stream.chat_ctx.append(
                        text=ev.alternatives[0].text
                    )
                    llm_stream = local_llm.chat(chat_ctx=new_chat_context)
                    llm_reply_stream = _llm_stream_to_str_iterable(llm_stream)
                    async for seg in llm_reply_stream:
                        tts_stream.push_text(seg)
                    tts_stream.flush()
    await asyncio.gather(synth_task, playout_task)
    await tts_stream.aclose()


async def _llm_stream_to_str_iterable(stream: llm.LLMStream) -> AsyncIterable[str]:
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is None:
            continue
        yield content


async def entrypoint(ctx: JobContext):
    logger.info("starting trigger-phrase agent example")

    stt_local = deepgram.STT(keywords=[(TRIGGER_PHRASE, 3.5)])
    stt_stream = stt_local.stream()

    # publish agent track
    tts = elevenlabs.TTS(model_id="eleven_turbo_v2")
    agent_audio_source = rtc.AudioSource(tts.sample_rate, tts.num_channels)
    agent_track = rtc.LocalAudioTrack.create_audio_track(
        "agent-mic", agent_audio_source
    )
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    # setup LLM
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )
    local_llm = openai.LLM()
    llm_stream = local_llm.chat(chat_ctx=initial_ctx)

    async def subscribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stt_forwarder = transcription.STTSegmentsForwarder(
            room=ctx.room, participant=participant, track=track
        )
        asyncio.create_task(
            _respond_to_user(
                stt_stream=stt_stream,
                tts=tts,
                agent_audio_source=agent_audio_source,
                local_llm=local_llm,
                llm_stream=llm_stream,
                stt_forwarder=stt_forwarder,
            )
        )

        async for ev in audio_stream:
            stt_stream.push_frame(ev.frame)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            subscribe_task = asyncio.create_task(subscribe_track(participant, track))
            asyncio.gather(subscribe_task)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await ctx.room.local_participant.publish_track(agent_track, options)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
