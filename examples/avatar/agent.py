import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from actions import ActionController, supports_actions
from dotenv import find_dotenv, load_dotenv
from hold_music import hold_beats
from personas import Persona, compose_instructions, resolve_persona

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
    llm as agents_llm,
)
from livekit.agents.llm import function_tool
from livekit.plugins import lemonslice
from livekit.rtc import RpcError, RpcInvocationData

load_dotenv(find_dotenv(usecwd=False))
logger = logging.getLogger("avatar")
server = AgentServer()


@dataclass
class State:
    persona: Persona
    avatar: lemonslice.AvatarSession
    session_id: str


class MotionAgent(Agent):
    def __init__(self, persona: Persona, actions: ActionController) -> None:
        super().__init__(
            instructions=compose_instructions(persona),
            tts=inference.TTS("cartesia/sonic-3.5", voice=persona.voice_id),
            chat_ctx=agents_llm.ChatContext.empty(),
        )
        self._actions = actions

    @function_tool
    async def wave(self) -> str:
        """Wave to the user. Only call when they explicitly ask you to wave."""
        return await self._actions.play("wave")

    @function_tool
    async def dance(self) -> str:
        """Dance for the user. Only call when they explicitly ask you to dance."""
        return await self._actions.play("dance")

    @function_tool
    async def turn(self) -> str:
        """Turn side to side. Only call when they explicitly ask you to turn."""
        return await self._actions.play("turn")


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    meta = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
    initial = resolve_persona(meta.get("set_avatar"))
    logger.info("starting session with persona %s", initial.id)

    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemini-3.5-flash"),
        turn_handling=TurnHandlingOptions(
            interruption={"resume_false_interruption": False},
        ),
    )

    def make_avatar(p: Persona) -> lemonslice.AvatarSession:
        return lemonslice.AvatarSession(
            agent_image_url=p.image_url,
            agent_prompt=p.speaking_prompt,
            agent_idle_prompt=p.idle_prompt,
            idle_timeout=120,
            response_done_timeout=2,
        )

    actions = ActionController()
    ctx.add_shutdown_callback(actions.shutdown)

    def make_agent(p: Persona) -> Agent:
        if supports_actions(p.id):
            return MotionAgent(p, actions)
        return Agent(
            instructions=compose_instructions(p),
            tts=inference.TTS("cartesia/sonic-3.5", voice=p.voice_id),
            chat_ctx=agents_llm.ChatContext.empty(),
        )

    avatar = make_avatar(initial)
    session_id = await avatar.start(session, room=ctx.room)
    state = State(persona=initial, avatar=avatar, session_id=session_id)
    await state.avatar.wait_for_join()

    if supports_actions(initial.id):
        actions.set_session(state.session_id, initial.id)

    await session.start(agent=make_agent(initial), room=ctx.room)

    if supports_actions(initial.id):
        await actions.opening_wave()

    session.generate_reply(
        instructions=(
            f"It's your turn to speak first. Open with a single short greeting in "
            f"character as {initial.name} and then stop."
            + (" Do not call wave — you already waved." if supports_actions(initial.id) else "")
        )
    )

    bg_audio = BackgroundAudioPlayer()
    await bg_audio.start(room=ctx.room, agent_session=session)

    @asynccontextmanager
    async def hold_music() -> AsyncIterator[None]:
        handle = bg_audio.play(AudioConfig(source=hold_beats(), fade_in=1.0, fade_out=0.4))
        try:
            yield
        finally:
            handle.stop()

    switch_lock = asyncio.Lock()

    @ctx.room.local_participant.register_rpc_method("set_avatar")
    async def set_avatar(data: RpcInvocationData) -> str:
        if switch_lock.locked():
            raise RpcError(
                RpcError.ErrorCode.APPLICATION_ERROR,
                "Still switching to the previous persona, please try again in a moment.",
            )

        new_persona = resolve_persona(json.loads(data.payload)["value"])

        async with switch_lock:
            if new_persona.id == state.persona.id:
                return json.dumps({"id": state.persona.id})

            logger.info("switching persona: %s -> %s", state.persona.id, new_persona.id)
            session.interrupt()

            async with hold_music():
                await actions.cancel()
                await state.avatar.aclose()
                state.avatar = make_avatar(new_persona)
                state.session_id = await state.avatar.start(session, room=ctx.room)
                await state.avatar.wait_for_join()
                session.update_agent(make_agent(new_persona))
                state.persona = new_persona
                # Lemonslice's video pipeline needs a beat after wait_for_join
                # before it actually consumes audio + emits frames.
                await asyncio.sleep(1.2)

            if supports_actions(new_persona.id):
                actions.set_session(state.session_id, new_persona.id)
                await actions.opening_wave()

            session.generate_reply(
                instructions=(
                    f"It's your turn to speak first. Open with a single short line in "
                    f"character as {state.persona.name} (acknowledge that you're who they "
                    "just picked) and then stop."
                    + (
                        " Do not call wave — you already waved."
                        if supports_actions(new_persona.id)
                        else ""
                    )
                )
            )

            return json.dumps({"id": state.persona.id})


if __name__ == "__main__":
    cli.run_app(server)
