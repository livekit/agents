"""v1 clients (LiveKit SDKs and UI kits built for Agents v1) against an Agents 2.0 agent: what
they say to it and read back (chat, transcription streams, legacy packets, interruptions,
expressions, pre-connect audio; ``COMPAT.md`` #6-13).

The client side is plain ``livekit.rtc`` doing what the cited v1 client does; the agent is a real
worker and job running a ``VoiceSession`` with scripted models (``v1_clients.py``)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid

import pytest

from livekit import rtc
from tests.fakes import until

from .v1_clients import (
    CHAT_DEFLATED,
    V1_CLIENTS,
    Transcription,
    V1Client,
    joined,
    next_transcription,
    play,
    publish_microphone,
    read_transcriptions,
    send_preconnect_buffer,
    speak,
    starter_token,
    tone,
    wait_for_agent,
    wait_for_agent_state,
)

pytestmark = pytest.mark.usefixtures("compat_worker")

REPLIES = {"hello there agent": "hi, how can I help you today", "what time is it": "it is noon"}
LONG = "this answer runs long enough for the caller to cut in before it ends"


def new_room() -> str:
    return "compat-" + uuid.uuid4().hex


def agent_streams_left(transcriptions: asyncio.Queue[Transcription], agent: str) -> int:
    return sum(transcriptions.get_nowait()[0] == agent for _ in range(transcriptions.qsize()))


@pytest.mark.parametrize("client", V1_CLIENTS)
async def test_chat_text_from_the_user_is_answered_aloud_and_in_the_transcript(
    sfu: str, client: V1Client
) -> None:
    """#6: each SDK broadcasts ``lk.chat`` as one text/plain stream without attributes, the JS SDK
    inline and deflated when every peer supports data streams v2 (``CHAT_DEFLATED``); the agent
    answers aloud, and the answer reaches the agent transcript the client renders.
    https://github.com/livekit/agents/issues/5178"""
    async with joined(sfu, starter_token(new_room(), metadata={"replies": REPLIES})) as room:
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")

        await room.local_participant.send_text(
            "what time is it", topic="lk.chat", compress=CHAT_DEFLATED[client]
        )
        await wait_for_agent_state(room, "speaking")

        _, reply = await next_transcription(transcriptions, agent.identity)
        assert await reply.read_all() == "it is noon"


async def test_chat_sent_while_the_agent_speaks_cuts_its_reply(sfu: str) -> None:
    """#6: a chat message sent mid-reply interrupts it: the reply's segment closes final with what
    was spoken so far, and the new answer follows."""
    script = {"replies": {"hello there agent": LONG, "what time is it": "it is noon"}}
    async with joined(sfu, starter_token(new_room(), metadata=script)) as room:
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")
        await room.local_participant.send_text("hello there agent", topic="lk.chat")
        _, long_reply = await next_transcription(transcriptions, agent.identity)
        first_word = await anext(aiter(long_reply))

        await room.local_participant.send_text("what time is it", topic="lk.chat")

        cut = first_word + await long_reply.read_all()
        assert (long_reply.info.attributes or {})["lk.transcription_final"] == "true"
        assert LONG.startswith(cut) and len(cut) < len(LONG), cut
        _, reply = await next_transcription(transcriptions, agent.identity)
        assert await reply.read_all() == "it is noon"


@pytest.mark.parametrize("client", V1_CLIENTS)
async def test_chat_from_a_participant_the_agent_does_not_serve_is_not_answered(
    sfu: str, client: V1Client
) -> None:
    """#6: only the linked participant's ``lk.chat`` reaches the agent."""
    name = new_room()
    async with joined(sfu, starter_token(name, metadata={"replies": REPLIES})) as room:
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")
        async with joined(sfu, starter_token(name, "bystander", agent=None)) as bystander:
            deflated = CHAT_DEFLATED[client]
            await bystander.local_participant.send_text(
                "what time is it", topic="lk.chat", compress=deflated
            )
            await room.local_participant.send_text(
                "hello there agent", topic="lk.chat", compress=deflated
            )

            _, reply = await next_transcription(transcriptions, agent.identity)
            assert await reply.read_all() == REPLIES["hello there agent"]
            await wait_for_agent_state(room, "listening")

        assert agent_streams_left(transcriptions, agent.identity) == 0


async def test_the_legacy_chat_packet_is_not_answered(sfu: str) -> None:
    """#12: components-js ``chat.ts:305`` also publishes the legacy ``lk-chat-topic`` packet and
    components-flutter ``chat_context.dart:74`` sends only that; like v1, the agent reads only the
    ``lk.chat`` copy."""
    async with joined(sfu, starter_token(new_room(), metadata={"replies": REPLIES})) as room:
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")
        packet = {"id": "1", "timestamp": int(time.time() * 1000), "message": "what time is it"}

        await room.local_participant.publish_data(json.dumps(packet), topic="lk-chat-topic")
        await room.local_participant.send_text("hello there agent", topic="lk.chat")

        _, reply = await next_transcription(transcriptions, agent.identity)
        assert await reply.read_all() == REPLIES["hello there agent"]
        await wait_for_agent_state(room, "listening")
        assert agent_streams_left(transcriptions, agent.identity) == 0


async def test_agent_transcripts_are_final_delta_streams_naming_the_agents_track(sfu: str) -> None:
    """#7: components-js ``useTranscriptions``, client-sdk-swift ``TranscriptionStreamReceiver``,
    client-sdk-flutter and components-android (whose ``rememberVoiceAssistant`` keeps only segments
    whose ``lk.transcribed_track_id`` is the agent's microphone, ``RememberVoiceAssistant.kt:72-79``)
    read one stream per reply from the agent: a unique ``SG_`` segment id, final "false" opening
    and "true" closing, and the spoken text."""
    async with joined(sfu, starter_token(new_room(), metadata={"replies": REPLIES})) as room:
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")
        microphone, _ = await publish_microphone(room)

        replies = []
        for said, answer in REPLIES.items():
            await speak(microphone, said)
            header, reply = await next_transcription(transcriptions, agent.identity)
            assert await reply.read_all() == answer
            replies.append((header, dict(reply.info.attributes or {})))
            await wait_for_agent_state(room, "listening")

        [voice] = [p.sid for p in agent.track_publications.values()]
        for header, trailer in replies:
            assert header["lk.segment_id"].startswith("SG_")
            assert header["lk.transcription_final"] == "false"
            assert header["lk.transcribed_track_id"] == voice
            assert trailer["lk.transcription_final"] == "true"
        assert replies[0][0]["lk.segment_id"] != replies[1][0]["lk.segment_id"]


async def test_user_transcripts_are_snapshots_of_one_segment_ending_final(sfu: str) -> None:
    """#8: components-js ``useSessionMessages.ts:65-111`` (sender is the local participant),
    client-sdk-swift, components-android ``RememberSessionMessages.kt:97`` and client-sdk-flutter
    read the user's words as streams sent on the user's behalf, each a whole snapshot of the
    interim text under one segment id naming the user's microphone, the last one final, before
    the agent's reply."""
    async with joined(sfu, starter_token(new_room(), metadata={"replies": REPLIES})) as room:
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")
        microphone, publication = await publish_microphone(room)

        await speak(microphone, "hello there agent")

        snapshots = []
        async with asyncio.timeout(15):
            while (opened := await transcriptions.get())[0] != agent.identity:
                sender, header, stream = opened
                snapshots.append((sender, header, await stream.read_all()))
        assert {sender for sender, _, _ in snapshots} == {"user"}
        assert len(snapshots) >= 2
        assert snapshots[-1][2] == "hello there agent"
        assert all("hello there agent".startswith(text) for _, _, text in snapshots)
        assert len({header["lk.segment_id"] for _, header, _ in snapshots}) == 1
        assert {header["lk.transcribed_track_id"] for _, header, _ in snapshots} == {
            publication.sid
        }
        finals = [header["lk.transcription_final"] for _, header, _ in snapshots]
        assert finals == ["false"] * (len(snapshots) - 1) + ["true"]


async def test_user_speech_cuts_the_agent_within_a_second(sfu: str) -> None:
    """#9: no client sends anything to interrupt; components-js, client-sdk-swift (which closes
    the segment itself) and client-sdk-flutter see the agent's segment end. Talking over the agent
    closes its segment final with only what was said, and the state leaves ``speaking`` within a
    second of the user's words being transcribed."""
    script = {"replies": {"hello there agent": LONG, "what time is it": "it is noon"}}
    async with joined(sfu, starter_token(new_room(), metadata=script)) as room:
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")
        microphone, _ = await publish_microphone(room)
        await speak(microphone, "hello there agent")
        _, long_reply = await next_transcription(transcriptions, agent.identity)
        first_word = await anext(aiter(long_reply))

        talking = asyncio.create_task(speak(microphone, "what time is it"))
        async with asyncio.timeout(10):
            while (
                "what" not in await (await next_transcription(transcriptions, "user"))[1].read_all()
            ):
                pass
        heard = time.monotonic()
        async with asyncio.timeout(5):
            while agent.attributes["lk.agent.state"] == "speaking":
                await asyncio.sleep(0.01)
        yielded = time.monotonic()
        await talking

        assert yielded - heard < 1.0
        cut = first_word + await long_reply.read_all()
        assert (long_reply.info.attributes or {})["lk.transcription_final"] == "true"
        assert LONG.startswith(cut) and len(cut) < len(LONG), cut
        _, reply = await next_transcription(transcriptions, agent.identity)
        assert await reply.read_all() == "it is noon"


@pytest.mark.parametrize(
    "markup", ['<expr type="expression" label="happy"/>', '<emotion value="happy"/>']
)
async def test_the_reply_carries_its_expression_for_the_agents_ui_avatar(
    sfu: str, markup: str
) -> None:
    """#10: components-js ``useAgentExpression.ts`` (agents-ui 1.0.8) reads ``lk.expression`` from
    the agent segment's attributes as ``{expression, mood}`` and shows the text without markup."""
    script = {"replies": {"what time is it": markup + "Great news, it worked"}}
    async with joined(sfu, starter_token(new_room(), metadata=script)) as room:
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")

        await room.local_participant.send_text("what time is it", topic="lk.chat")

        _, reply = await next_transcription(transcriptions, agent.identity)
        assert await reply.read_all() == "Great news, it worked"
        assert "lk.expression" in (reply.info.attributes or {}), reply.info.attributes
        expression = json.loads((reply.info.attributes or {})["lk.expression"])
        assert expression == {"expression": "happy", "mood": "happy"}


async def test_legacy_transcription_packets_mirror_the_final_segments(sfu: str) -> None:
    """#11: components-react ``useTrackTranscription`` (``useVoiceAssistant().agentTranscriptions``),
    client-sdk-flutter ``transcription_context.dart`` and client-sdk-swift's opt-in
    ``TranscriptionDelegateReceiver`` read ``TranscriptionReceived`` packets: the user's and the
    agent's segments end final, with the streams' segment ids and text."""
    async with joined(sfu, starter_token(new_room(), metadata={"replies": REPLIES})) as room:
        packets: list[tuple[str, rtc.TranscriptionSegment]] = []
        room.on(
            "transcription_received",
            lambda segments, participant, _: packets.extend(
                (participant.identity, s) for s in segments
            ),
        )
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")
        microphone, _ = await publish_microphone(room)

        await speak(microphone, "hello there agent")
        header, stream = await next_transcription(transcriptions, agent.identity)
        reply = await stream.read_all()
        await wait_for_agent_state(room, "listening")

        expected = (agent.identity, header["lk.segment_id"], reply)
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(5):  # packets travel apart from the stream
                while expected not in {(who, s.id, s.text) for who, s in packets if s.final}:
                    await asyncio.sleep(0.01)
        finals = {(who, s.id, s.text) for who, s in packets if s.final}
        assert expected in finals, finals
        assert any(who == "user" and text == "hello there agent" for who, _, text in finals)


async def test_legacy_transcription_packets_name_each_speakers_microphone(sfu: str) -> None:
    """#11: client-sdk-js ``CoreRoom.ts:2152-2161`` hands a ``TranscriptionReceived`` packet only
    to the publication its ``track_id`` names, and client-sdk-flutter ``room.dart:1038-1046``
    drops one whose participant it cannot resolve: the user's segments name the user and their
    microphone, the agent's name the agent and its microphone."""
    async with joined(sfu, starter_token(new_room(), metadata={"replies": REPLIES})) as room:
        packets: list[tuple[str | None, str | None, rtc.TranscriptionSegment]] = []
        room.on(
            "transcription_received",
            lambda segments, participant, publication: packets.extend(
                (participant and participant.identity, publication and publication.sid, s)
                for s in segments
            ),
        )
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")
        microphone, user_microphone = await publish_microphone(room)

        await speak(microphone, "hello there agent")
        _, stream = await next_transcription(transcriptions, agent.identity)
        reply = await stream.read_all()
        await until(
            lambda: any(s.final and s.text == reply for _, _, s in packets), "the reply's packet"
        )

        [agent_microphone] = [
            p.sid for p in agent.track_publications.values() if p.kind == rtc.TrackKind.KIND_AUDIO
        ]
        assert {(who, sid) for who, sid, s in packets if s.text == reply} == {
            (agent.identity, agent_microphone)
        }
        assert {(who, sid) for who, sid, s in packets if s.text == "hello there agent"} == {
            ("user", user_microphone.sid)
        }


SAID_EARLY = "early words before connect"


def said_early(sample_rate: int) -> bytes:
    """A recording of the user speaking before the agent joined."""
    return tone(900, 1.2, sample_rate) + bytes(sample_rate // 5 * 2)


def silence(sample_rate: int) -> bytes:
    """A recording of the user waiting quietly."""
    return bytes(sample_rate * 2)


async def stay_quiet(microphone: rtc.AudioSource, until: asyncio.Event) -> None:
    """A live microphone in a quiet room."""
    while not until.is_set():
        await play(microphone, bytes(960))


async def first_final_user_words(transcriptions: asyncio.Queue[Transcription], failure: str) -> str:
    try:
        async with asyncio.timeout(10):
            while True:
                sender, _, stream = await transcriptions.get()
                text = await stream.read_all()
                if (stream.info.attributes or {})["lk.transcription_final"] == "true":
                    break
    except TimeoutError:
        pytest.fail(failure)
    assert sender == "user"
    return text


@pytest.mark.parametrize("client", V1_CLIENTS)
async def test_words_spoken_before_the_agent_joined_are_heard_and_answered(
    sfu: str, client: V1Client
) -> None:
    """#13: the SDKs record the microphone before connecting, publish it flagged
    ``TF_PRECONNECT_BUFFER``, and once the agent is active stream the recording on
    ``lk.agent.pre-connect-audio-buffer`` (client-sdk-js ``LocalParticipant.ts:1080-1139``,
    client-sdk-swift ``PreConnectAudioBuffer.swift:190``, client-sdk-android
    ``PreconnectAudioBuffer.kt:141``, client-sdk-flutter ``pre_connect_audio_buffer.dart:322``).
    The agent transcribes those first words and answers them."""
    script = {"replies": {SAID_EARLY: "glad you started early"}}
    async with joined(sfu, starter_token(new_room(), metadata=script)) as room:
        transcriptions = read_transcriptions(room)
        microphone, publication = await publish_microphone(room, preconnect=True)
        done = asyncio.Event()
        quiet = asyncio.create_task(stay_quiet(microphone, done))
        try:
            agent = await wait_for_agent(room)

            await send_preconnect_buffer(client, room, agent.identity, publication.sid, said_early)

            said = await first_final_user_words(
                transcriptions, "the words recorded before connecting were never transcribed"
            )
            assert said == SAID_EARLY
            _, reply = await next_transcription(transcriptions, agent.identity)
            assert await reply.read_all() == "glad you started early"
        finally:
            done.set()
            await quiet


@pytest.mark.parametrize("client", V1_CLIENTS)
async def test_the_microphone_is_heard_after_a_preconnect_buffer(
    sfu: str, client: V1Client
) -> None:
    """#13: after the buffer, the live microphone is heard and answered, as in v1
    (``room_io/_input.py:436-485``)."""
    async with joined(sfu, starter_token(new_room(), metadata={"replies": REPLIES})) as room:
        transcriptions = read_transcriptions(room)
        microphone, publication = await publish_microphone(room, preconnect=True)
        agent = await wait_for_agent(room)
        await send_preconnect_buffer(client, room, agent.identity, publication.sid, silence)
        await wait_for_agent_state(room, "listening")

        await speak(microphone, "what time is it")

        said = await first_final_user_words(transcriptions, "the live microphone went unheard")
        assert said == "what time is it"
        _, reply = await next_transcription(transcriptions, agent.identity)
        assert await reply.read_all() == "it is noon"


@pytest.mark.parametrize("attributes", [{"sampleRate": "fast"}, {"channels": "0"}])
async def test_an_unreadable_preconnect_buffer_leaves_the_microphone_heard(
    sfu: str, attributes: dict[str, str]
) -> None:
    """#13: a buffer whose attributes cannot be read is dropped, as in v1; the call goes on."""
    async with joined(sfu, starter_token(new_room(), metadata={"replies": REPLIES})) as room:
        transcriptions = read_transcriptions(room)
        microphone, publication = await publish_microphone(room, preconnect=True)
        agent = await wait_for_agent(room)
        await send_preconnect_buffer(
            "js", room, agent.identity, publication.sid, silence, attributes
        )
        await wait_for_agent_state(room, "listening")

        await speak(microphone, "what time is it")

        said = await first_final_user_words(transcriptions, "the live microphone went unheard")
        assert said == "what time is it"


async def test_words_spoken_while_the_agent_waits_for_a_buffer_are_heard(sfu: str) -> None:
    """#13: a microphone flagged for pre-connect whose buffer never comes still delivers what the
    user says while the agent waits for it."""
    async with joined(sfu, starter_token(new_room(), metadata={"replies": REPLIES})) as room:
        transcriptions = read_transcriptions(room)
        microphone, _ = await publish_microphone(room, preconnect=True)
        await wait_for_agent_state(room, "listening")

        await speak(microphone, "what time is it")

        said = await first_final_user_words(transcriptions, "the words were discarded")
        assert said == "what time is it"
