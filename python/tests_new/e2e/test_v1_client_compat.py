"""v1 clients (LiveKit SDKs and UI kits built for Agents v1) against an Agents 2.0 agent:
finding it, reading its state, hearing it, dispatching it, calling it and seeing it leave (``COMPAT.md`` #1-5, #14-21).

The client side is plain ``livekit.rtc`` doing what the cited v1 client does; the agent is a real
worker and job running a ``VoiceSession`` with scripted models (``v1_clients.py``)."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from livekit import api, rtc
from livekit.protocol.agent_pb import agent_session as session_pb

from .sfu import KEY, SECRET, connected
from .v1_clients import (
    AGENT_NAME,
    AGENT_TONE,
    AVATAR_IDENTITY,
    V1_CLIENTS,
    V1Client,
    agent_server,
    joined,
    loudness_and_pitch,
    next_transcription,
    publish_microphone,
    read_transcriptions,
    speak,
    starter_token,
    wait_for_agent,
    wait_for_agent_state,
)

pytestmark = pytest.mark.usefixtures("compat_worker")

REPLIES = {"hello there agent": "hi, how can I help you today", "what time is it": "it is noon"}


def agent_left(
    participant: rtc.RemoteParticipant, agent: rtc.Participant, left: asyncio.Event
) -> None:
    if participant.identity == agent.identity:
        left.set()


def new_room() -> str:
    return "compat-" + uuid.uuid4().hex


async def test_the_agent_joins_as_the_one_active_participant_of_kind_agent(sfu: str) -> None:
    """#1: client-sdk-js ``Participant.isAgent``, components-js ``useAgent``, client-sdk-swift
    ``Room+Agent.swift:25``, components-android ``RememberAgent.kt:199`` and client-sdk-flutter
    ``room_agent.dart:32`` find the agent by kind alone, never by identity."""
    async with joined(sfu, starter_token(new_room())) as room:
        await wait_for_agent_state(room, "listening")

        agents = [
            p
            for p in room.remote_participants.values()
            if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        ]
        assert len(agents) == 1
        assert agents[0].state == rtc.ParticipantState.PARTICIPANT_STATE_ACTIVE
        assert "lk.publish_on_behalf" not in agents[0].attributes


async def test_the_agent_carries_its_registered_name(sfu: str) -> None:
    """#18: ``lk.agent.name`` (framework) and ``lk.agent_name`` (server) name the registered agent
    (client-sdk-swift ``Session.swift:56``, client-sdk-flutter ``constants.dart:18``)."""
    async with joined(sfu, starter_token(new_room())) as room:
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")

        assert agent.attributes["lk.agent.name"] == AGENT_NAME
        assert agent.attributes["lk.agent_name"] == AGENT_NAME


async def test_lk_agent_state_holds_only_known_values_and_follows_a_turn(sfu: str) -> None:
    """#2 #16 #17: components-js ``useAgent.ts:672-700``, client-sdk-swift
    ``AttributeTypings.swift:60`` and components-flutter decode the agent's attributes strictly: one
    unknown ``lk.agent.state`` value, or a string ``lk.agent.inputs``/``outputs``, blanks the state.
    The state is set, never removed, ``listening`` without user audio, then ``speaking`` and back to
    ``listening`` around a spoken turn."""
    async with joined(sfu, starter_token(new_room(), metadata={"replies": REPLIES})) as room:
        agent = await wait_for_agent(room)
        seen: list[dict[str, str]] = [dict(agent.attributes)]
        room.on("participant_attributes_changed", lambda _, p: seen.append(dict(p.attributes)))
        await wait_for_agent_state(room, "listening")

        microphone, _ = await publish_microphone(room)
        await speak(microphone, "hello there agent")
        await wait_for_agent_state(room, "speaking")
        await wait_for_agent_state(room, "listening")

        states = [attributes.get("lk.agent.state") for attributes in seen]
        assert set(states) <= {"initializing", "idle", "listening", "thinking", "speaking"}, states
        assert not {"lk.agent.inputs", "lk.agent.outputs", "lk.agent.input_mode"} & set().union(
            *seen
        )


async def test_the_agent_listens_within_ten_seconds_of_the_client_connecting(sfu: str) -> None:
    """#3: clients fail the call unless the agent is ready soon after they connect: 20 s in
    components-js ``useAgent.ts:318-337``, client-sdk-swift ``Session.swift:290`` and
    client-sdk-flutter ``session.dart:226``, 10 s in the agent-starter-embed popup
    (``session-view.tsx:67``)."""
    room = rtc.Room()
    started = asyncio.get_running_loop().time()
    try:
        await room.connect(sfu, starter_token(new_room()))
        await wait_for_agent_state(room, "listening", timeout=10)
    finally:
        await room.disconnect()

    assert asyncio.get_running_loop().time() - started < 10


async def test_the_agent_is_heard_on_its_one_microphone_track(sfu: str) -> None:
    """#5: clients pick the agent's audio by source, never by name (components-js
    ``useAgent.ts:548``, client-sdk-swift ``Agent.swift:198``, components-android
    ``RememberAgent.kt:218``, client-sdk-flutter ``agent.dart:194``); the ESP32 SDK plays the first
    remote audio track (``engine.c:156-175``). That one MICROPHONE track carries the agent's
    speech, audibly, while ``lk.agent.state`` is ``speaking``."""
    async with joined(sfu, starter_token(new_room(), metadata={"replies": REPLIES})) as room:
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")
        microphone, _ = await publish_microphone(room)

        await speak(microphone, "hello there agent")

        async with asyncio.timeout(10):
            while not (
                audio := [
                    p
                    for p in agent.track_publications.values()
                    if p.kind == rtc.TrackKind.KIND_AUDIO and p.track is not None
                ]
            ):
                await asyncio.sleep(0.01)
        [publication] = audio
        assert publication.track is not None
        heard = rtc.AudioStream(publication.track, sample_rate=48000, num_channels=1)
        speech_frames = 0
        async with asyncio.timeout(15):
            async for event in heard:
                rms, pitch = loudness_and_pitch(bytes(event.frame.data), 48000)
                if rms > 300 and pitch == AGENT_TONE:
                    assert agent.attributes["lk.agent.state"] == "speaking"
                    speech_frames += 1
                    if speech_frames == 20:
                        break
        await heard.aclose()

        assert publication.source == rtc.TrackSource.SOURCE_MICROPHONE
        assert publication.name == "roomio_audio"


async def test_token_dispatch_hands_the_job_the_tokens_metadata(sfu: str, tmp_path: Path) -> None:
    """#19a: ``roomConfig.agents[{agentName, metadata}]`` (client-sdk-js ``TokenSource``,
    components-js ``useSession``, the Swift/Android/Flutter token sources, agent-starter-react,
    -embed and -react-native ``agentName``) dispatches the named agent with that metadata."""
    metadata = {"marker": str(tmp_path / "metadata"), "user": {"plan": "pro", "café": True}}
    async with joined(sfu, starter_token(new_room(), metadata=metadata)) as room:
        await wait_for_agent(room)

        assert json.loads((tmp_path / "metadata").read_text()) == metadata


async def test_an_unnamed_worker_joins_rooms_whose_token_names_no_agent(
    sfu: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#19b: agent-starter-swift, -android and -flutter connect without naming an agent; an
    unnamed rooms-only 2.0 worker is dispatched to every new room."""
    monkeypatch.delenv("LIVEKIT_AGENT_NAME", raising=False)
    async with (
        agent_server(sfu, agent_name=""),
        joined(sfu, starter_token(new_room(), agent=None)) as room,
    ):
        await wait_for_agent_state(room, "listening")


async def test_a_named_worker_joins_only_when_dispatched_by_name(sfu: str) -> None:
    """#19c: a named worker (required once it also serves text endpoints) takes no room whose
    token names no agent, so the starters must send ``agentName``; dispatching it by name works."""
    name = new_room()
    async with (
        joined(sfu, starter_token(name, agent=None)) as room,
        api.LiveKitAPI(sfu, KEY, SECRET) as lk,
    ):
        assert [d.agent_name for d in await lk.agent_dispatch.list_dispatch(name)] == [""]

        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(room=name, agent_name=AGENT_NAME)
        )
        await wait_for_agent_state(room, "listening")

        kinds = [p.kind for p in room.remote_participants.values()]
        assert kinds == [rtc.ParticipantKind.PARTICIPANT_KIND_AGENT]


async def test_the_agent_leaves_when_its_user_leaves(sfu: str) -> None:
    """#21: the session closes when its user disconnects, and the agent leaves; seen by a hidden
    observer the agent never serves."""
    name = new_room()
    async with joined(sfu, starter_token(name)) as user:
        agent = await wait_for_agent(user)
        await wait_for_agent_state(user, "listening")
        async with connected(sfu, name, "observer", hidden=True) as observer:
            left = asyncio.Event()
            observer.on("participant_disconnected", lambda p: agent_left(p, agent, left))

            await user.disconnect()

            await asyncio.wait_for(left.wait(), 10)


async def test_the_room_is_deleted_when_its_user_leaves_if_configured(sfu: str) -> None:
    """#21: with ``delete_room_on_close``, the user leaving ends the session and deletes the room,
    disconnecting everyone left in it."""
    name = new_room()
    async with joined(sfu, starter_token(name, metadata={"delete_room": True})) as user:
        await wait_for_agent_state(user, "listening")
        async with connected(sfu, name, "observer", hidden=True) as observer:
            ended: list[rtc.DisconnectReason.ValueType] = []
            observer.on("disconnected", ended.append)

            await user.disconnect()

            async with asyncio.timeout(10):
                while not ended:
                    await asyncio.sleep(0.01)
            assert ended == [rtc.DisconnectReason.ROOM_DELETED]


async def test_a_session_closed_by_the_agent_shows_the_agent_leaving(sfu: str) -> None:
    """#21: components-js ``useAgent.ts:599`` ("Agent left the room unexpectedly") and
    client-sdk-swift/flutter ``failed(.left)`` rely on the agent participant leaving once the
    agent's own code ends the session."""
    async with joined(sfu, starter_token(new_room(), metadata={"shutdown_on": "goodbye"})) as room:
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")
        left = asyncio.Event()
        room.on("participant_disconnected", lambda p: agent_left(p, agent, left))

        await room.local_participant.send_text("goodbye", topic="lk.chat")

        await asyncio.wait_for(left.wait(), 10)
        assert room.isconnected()


@pytest.mark.parametrize("client", V1_CLIENTS)
async def test_a_tool_calls_the_rpc_method_the_device_registered(
    sfu: str, client: V1Client
) -> None:
    """#20: the ESP32 device registers ``get_cpu_temp`` (``example.c``) and its example agent calls
    it on the linked participant from a tool; the answer reaches the reply. Every SDK registers
    RPC methods the same way (``registerRpcMethod``), so each family is the same wire here."""
    script = {
        "tool_calls": {"what is my device temperature": "read_device_temperature"},
        "replies": {"42.5": "your device is at 42.5 degrees"},
    }
    async with joined(sfu, starter_token(new_room(), metadata=script)) as room:
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")
        callers: list[str] = []

        async def temperature(data: rtc.RpcInvocationData) -> str:
            callers.append(data.caller_identity)
            return "42.5"

        room.local_participant.register_rpc_method("get_cpu_temp", temperature)

        await room.local_participant.send_text("what is my device temperature", topic="lk.chat")

        _, reply = await next_transcription(transcriptions, agent.identity)
        assert await reply.read_all() == "your device is at 42.5 degrees"
        assert callers == [agent.identity]


@pytest.mark.parametrize("client", V1_CLIENTS)
async def test_the_client_calls_an_rpc_method_the_agent_registered(
    sfu: str, client: V1Client
) -> None:
    """#20: the agents-playground RPC panel, and any SDK's ``performRpc``, calls a method the
    application registered on the agent's room."""
    async with joined(sfu, starter_token(new_room(), metadata={"app_rpc": True})) as room:
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")

        answer = await room.local_participant.perform_rpc(
            destination_identity=agent.identity, method="app.echo", payload="ping"
        )

        assert answer == "echo:ping"


async def test_removed_client_event_rpcs_answer_unsupported_method(sfu: str) -> None:
    """#15: no shipped client calls the removed ``lk.agent.get_session_state``-style RPCs; one that
    did is told UNSUPPORTED_METHOD rather than left to time out."""
    async with joined(sfu, starter_token(new_room())) as room:
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")

        with pytest.raises(rtc.RpcError) as refused:
            await room.local_participant.perform_rpc(
                destination_identity=agent.identity,
                method="lk.agent.get_session_state",
                payload="{}",
            )

        assert refused.value.code == rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD


async def send_session_request_like_playground(
    room: rtc.Room, agent: str, messages: list[session_pb.AgentSessionMessage], **request: object
) -> session_pb.SessionResponse:
    """agents-playground ``useRemoteSession.ts:144-196``: one ``SessionRequest`` byte stream to the
    agent on ``lk.agent.session``, answered by ``request_id`` within its 5 s timeout."""
    request_id = "req_" + uuid.uuid4().hex
    message = session_pb.AgentSessionMessage(
        request=session_pb.SessionRequest(request_id=request_id, **request)  # type: ignore[arg-type]  # protobuf kwargs are per-field
    )
    writer = await room.local_participant.stream_bytes(
        "", topic="lk.agent.session", destination_identities=[agent]
    )
    await writer.write(message.SerializeToString())
    await writer.aclose()
    async with asyncio.timeout(5):
        while not (
            answers := [m.response for m in messages if m.response.request_id == request_id]
        ):
            await asyncio.sleep(0.01)
    return answers[0]


def read_session_messages_like_playground(room: rtc.Room) -> list[session_pb.AgentSessionMessage]:
    """agents-playground ``useRemoteSession.ts:89-121``: every message on ``lk.agent.session``."""
    messages: list[session_pb.AgentSessionMessage] = []

    def opened(reader: rtc.ByteStreamReader, sender: str) -> None:
        async def read() -> None:
            data = b"".join([chunk async for chunk in reader])
            messages.append(session_pb.AgentSessionMessage.FromString(data))

        asyncio.get_running_loop().create_task(read())

    room.register_byte_stream_handler("lk.agent.session", opened)
    return messages


async def test_the_playground_gets_rtc_stats_within_its_five_second_timeout(sfu: str) -> None:
    """#14: agents-playground polls ``get_rtc_stats`` every 5 s with a 5 s timeout while a call
    runs, and charts both peer connections."""
    async with joined(sfu, starter_token(new_room(), metadata={"replies": REPLIES})) as room:
        messages = read_session_messages_like_playground(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")
        microphone, _ = await publish_microphone(room)
        await speak(microphone, "hello there agent")
        await wait_for_agent_state(room, "speaking")

        stats = await send_session_request_like_playground(
            room, agent.identity, messages, get_rtc_stats={}
        )

        # The rtc SDK reports its peer connections' stats; it lists the subscriber's under the
        # publisher's (``subscriber_stats`` is empty for any participant with rtc 1.1.20).
        assert stats.get_rtc_stats.publisher_stats


async def test_the_playground_receives_the_calls_events_and_history(sfu: str) -> None:
    """#14: agents-playground renders the session's events (state changes, final transcripts,
    conversation items with their metrics, usage), each stamped ``created_at``, and reads the
    chat history."""
    async with joined(sfu, starter_token(new_room(), metadata={"replies": REPLIES})) as room:
        messages = read_session_messages_like_playground(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")

        microphone, _ = await publish_microphone(room)
        await speak(microphone, "hello there agent")
        await wait_for_agent_state(room, "speaking")
        await wait_for_agent_state(room, "listening")
        history = await send_session_request_like_playground(
            room, agent.identity, messages, get_chat_history={}
        )

        said = [
            i.message.content[0].text
            for i in history.get_chat_history.items
            if i.HasField("message")
        ]
        assert said == ["hello there agent", REPLIES["hello there agent"]]
        events = [m.event for m in messages if m.HasField("event")]
        kinds = {e.WhichOneof("event") for e in events}
        assert {
            "agent_state_changed",
            "user_input_transcribed",
            "conversation_item_added",
            "session_usage_updated",
        } <= kinds, kinds
        assert all(e.created_at.seconds > 0 for e in events)
        assert any(
            e.user_input_transcribed.is_final
            and e.user_input_transcribed.transcript == "hello there agent"
            for e in events
        )
        [reply] = [
            e.conversation_item_added.item.message
            for e in events
            if e.conversation_item_added.item.message.role == session_pb.ChatRole.ASSISTANT
        ]
        assert reply.HasField("metrics"), reply


async def test_only_the_jobs_primary_session_answers_the_playground(sfu: str) -> None:
    """#14: a job running a second session still answers each playground request once."""
    script = {"second_session": True}
    async with joined(sfu, starter_token(new_room(), metadata=script)) as room:
        messages = read_session_messages_like_playground(room)
        agent = await wait_for_agent(room)
        await wait_for_agent_state(room, "listening")

        first = await send_session_request_like_playground(room, agent.identity, messages, ping={})
        state = await send_session_request_like_playground(
            room, agent.identity, messages, get_session_state={}
        )

        assert state.get_session_state.agent_state == session_pb.AgentState.AS_LISTENING
        responses = [m.response.request_id for m in messages if m.HasField("response")]
        assert responses == [first.request_id, state.request_id]


async def test_an_avatar_worker_is_told_apart_from_the_agent_it_speaks_for(sfu: str) -> None:
    """#4: an avatar worker joins as kind AGENT with ``lk.publish_on_behalf`` naming the agent;
    components-js ``useAgent.ts:521-575``, client-sdk-swift ``Participant+Agent.swift:39``,
    components-android ``RememberAgent.kt:200-256`` and client-sdk-flutter ``agent.dart:194`` take
    the AGENT without that key as the agent. The agent sends its speech to the worker on
    ``lk.audio_stream``, signs its own transcripts, and never serves the worker as its user."""
    name = new_room()
    script = {"avatar": True, "replies": REPLIES}
    async with joined(sfu, starter_token(name, metadata=script)) as room:
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        worker_token = (
            api.AccessToken(KEY, SECRET)
            .with_identity(AVATAR_IDENTITY)
            .with_kind("agent")
            .with_attributes({"lk.publish_on_behalf": agent.identity})
            .with_grants(api.VideoGrants(room_join=True, room=name))
            .to_jwt()
        )
        async with joined(sfu, worker_token) as worker:
            speech: list[tuple[int, bytes]] = []

            def segment(reader: rtc.ByteStreamReader, sender: str) -> None:
                async def play_out() -> None:
                    rate = int((reader.info.attributes or {})["sample_rate"])
                    speech.append((rate, b"".join([chunk async for chunk in reader])))
                    await worker.local_participant.perform_rpc(
                        destination_identity=agent.identity,
                        method="lk.playback_finished",  # as every v1 worker reports
                        payload=json.dumps({"playback_position": 1.0, "interrupted": False}),
                    )

                asyncio.get_running_loop().create_task(play_out())

            worker.register_byte_stream_handler("lk.audio_stream", segment)
            camera = rtc.VideoSource(16, 16)
            await worker.local_participant.publish_track(
                rtc.LocalVideoTrack.create_video_track("avatar", camera),
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
            )
            await wait_for_agent_state(room, "listening")

            await worker.local_participant.send_text("what time is it", topic="lk.chat")
            await room.local_participant.send_text("hello there agent", topic="lk.chat")
            header, reply = await next_transcription(transcriptions, agent.identity)
            assert await reply.read_all() == REPLIES["hello there agent"]  # not the worker's
            await wait_for_agent_state(room, "listening")

            assert await wait_for_agent(room) is agent
            assert room.remote_participants[AVATAR_IDENTITY].attributes == {
                "lk.publish_on_behalf": agent.identity
            }
            assert speech and loudness_and_pitch(speech[0][1], speech[0][0])[1] == AGENT_TONE
            # The agent publishes no microphone here; as in v1 its stream then names no track.
            assert "lk.transcribed_track_id" not in header
            later = [transcriptions.get_nowait() for _ in range(transcriptions.qsize())]
            assert [t for t in later if t[0] == agent.identity] == []
