"""The shared harness exercises user-authored apps through real processes and public clients."""

import asyncio

from livekit.agents.a2a import TextPart

from .framework.app import E2E


async def test_a_bound_conversation_survives_an_application_restart(e2e: E2E) -> None:
    app = await e2e.start_app("tests.e2e.apps.conversation")
    async with e2e.text_client(app) as client:
        reply = await client.send("I am Alice")
        async for _ in reply:
            pass
        thread = reply.thread_id
    previous = app.child
    await app.restart()
    assert previous is not None and previous.reaped

    async with e2e.text_client(app) as client:
        reply = await client.send("recall", thread_id=thread)
        updates = [update async for update in reply]
    assert updates[-1].state == "completed"
    assert (
        "".join(
            part.text
            for artifact in updates[-1].artifacts
            for part in artifact.parts
            if isinstance(part, TextPart)
        )
        == "I am Alice"
    )


async def test_the_voice_client_observes_both_the_reply_and_received_audio(e2e: E2E) -> None:
    app = await e2e.start_app("tests.e2e.apps.voice_conversation", voice=True)
    async with e2e.voice_client(app) as caller:
        await caller.say("what time is it")
        reply = await caller.hear()
        assert reply.text == "it is noon"
        assert reply.audio_seconds > 0


async def test_a_spoken_interruption_reaches_the_caller_before_the_old_reply_would_finish(
    e2e: E2E,
) -> None:
    long = "This reply keeps speaking for six seconds unless the caller interrupts it with a new question."
    app = await e2e.start_app(
        "tests.e2e.apps.voice_conversation", voice=True, env={"GREETING": long}
    )
    async with e2e.voice_client(app) as caller:
        await caller.say("hello there agent")
        started = await caller.speech_started()
        hearing = asyncio.create_task(caller.hear())
        try:
            await caller.say("what time is it")
            cut = await hearing
        finally:
            if not hearing.done():
                hearing.cancel()
                await asyncio.gather(hearing, return_exceptions=True)
        assert long.startswith(cut.text) and len(cut.text) < len(long)
        answer = await caller.hear()
        assert answer.text == "it is noon"
        assert cut.audio_seconds > 0 and answer.audio_seconds > 0
        # The fake TTS would otherwise emit six seconds. The replacement is heard before
        # those frames could finish, not merely represented as an interrupted transcript.
        assert answer.ended_at - started < 4.5
