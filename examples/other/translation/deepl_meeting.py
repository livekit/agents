"""A meeting interpreter on DeepL Voice: everyone speaks their own language and hears the others
in theirs.

The agent registers as "deepl-meeting" and joins only the rooms that dispatch it. Each
participant's language is the `language` attribute of their access token, which can dispatch the
agent as well:

    api.AccessToken().with_attributes({"language": "de"}).with_room_config(
        api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name="deepl-meeting")])
    )

For every speaker and every other language in the room, one DeepL Voice session interprets that
speaker into that language and publishes it as an audio track named "<speaker>-<language>". Track
permissions give each track only to the listeners of its language, so any LiveKit client plays just
the interpretation meant for it. Listeners still hear the speaker's own voice, as with a human
interpreter.

    DEEPL_API_KEY=... python deepl_meeting.py dev

Set PRIVATE_TRACKS=0 to let everyone hear every interpretation, for example to listen in from
Agent Console.
"""

import asyncio
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    CloseEvent,
    ConversationItemAddedEvent,
    JobContext,
    UserInputTranscribedEvent,
    cli,
    llm,
    room_io,
)
from livekit.plugins import deepl

logger = logging.getLogger("deepl-meeting")

load_dotenv()

AGENT_NAME = "deepl-meeting"
LANGUAGE_ATTRIBUTE = "language"
PRIVATE_TRACKS = os.getenv("PRIVATE_TRACKS", "1") != "0"


def _base(language: str) -> str:
    """`en` for `en-US`: source languages and same-language checks ignore the variant."""
    return language.split("-")[0].lower()


@dataclass(frozen=True)
class _Route:
    speaker: str
    source: str
    target: str

    @property
    def track_name(self) -> str:
        return f"{self.speaker}-{self.target}"

    def __str__(self) -> str:
        return f"{self.speaker} {self.source}->{self.target}"


class Interpreters:
    """One DeepL Voice session per speaker and target language, kept in step with the room."""

    def __init__(self, room: rtc.Room) -> None:
        self._room = room
        self._sessions: dict[_Route, AgentSession[None]] = {}
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._warned: set[str] = set()
        self._closed = False

    def start(self) -> None:
        self._room.on("participant_connected", self._on_room_changed)
        self._room.on("participant_disconnected", self._on_room_changed)
        self._room.on("participant_attributes_changed", self._on_room_changed)
        # a track can only be granted to its listeners once it is published
        self._room.on("local_track_published", self._on_track_published)
        self._update_permissions()
        self._on_room_changed()

    async def aclose(self) -> None:
        self._closed = True
        self._room.off("participant_connected", self._on_room_changed)
        self._room.off("participant_disconnected", self._on_room_changed)
        self._room.off("participant_attributes_changed", self._on_room_changed)
        self._room.off("local_track_published", self._on_track_published)
        # reconciles already queued run first: they start nothing now, and what one had already
        # started is closed below with the rest
        await asyncio.gather(*self._tasks, return_exceptions=True)
        async with self._lock:
            sessions, self._sessions = list(self._sessions.values()), {}
        await asyncio.gather(*(session.aclose() for session in sessions), return_exceptions=True)

    def _on_room_changed(self, *_: object) -> None:
        task = asyncio.create_task(self._reconcile())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_track_published(self, *_: object) -> None:
        self._update_permissions()

    def _languages(self) -> dict[str, str]:
        languages: dict[str, str] = {}
        for participant in self._room.remote_participants.values():
            if language := participant.attributes.get(LANGUAGE_ATTRIBUTE):
                languages[participant.identity] = language
            elif participant.identity not in self._warned:
                self._warned.add(participant.identity)
                logger.warning(
                    f"{participant.identity} has no '{LANGUAGE_ATTRIBUTE}' attribute, so they are "
                    "neither interpreted nor given an interpretation"
                )
        return languages

    def _wanted(self) -> set[_Route]:
        languages = self._languages()
        return {
            _Route(speaker=speaker, source=_base(source), target=target)
            for speaker, source in languages.items()
            for listener, target in languages.items()
            if listener != speaker and _base(target) != _base(source)
        }

    async def _reconcile(self) -> None:
        async with self._lock:
            if self._closed:
                return
            wanted = self._wanted()
            stale = [route for route in self._sessions if route not in wanted]
            new = list(wanted - self._sessions.keys())

            # closing waits for DeepL to flush, so sessions close together, then start together:
            # a replacement can reuse its predecessor's track name, so the two must not overlap
            for route in stale:
                logger.info(f"stopping {route}")
            await asyncio.gather(
                *(self._sessions.pop(route).aclose() for route in stale), return_exceptions=True
            )
            started = await asyncio.gather(
                *(self._start(route) for route in new), return_exceptions=True
            )
            for route, result in zip(new, started, strict=True):
                if isinstance(result, BaseException):
                    logger.error(f"failed to start {route}", exc_info=result)
                else:
                    self._sessions[route] = result
            self._update_permissions()

    async def _start(self, route: _Route) -> AgentSession[None]:
        logger.info(f"starting {route}")
        session: AgentSession[None] = AgentSession(
            llm=deepl.realtime.VoiceModel(
                source_language=route.source, target_language=route.target, voice="match"
            ),
            user_away_timeout=None,
        )

        @session.on("user_input_transcribed")
        def _on_heard(ev: UserInputTranscribedEvent) -> None:
            if ev.is_final:
                logger.info(f"[{route}] heard", extra={"lk.pii.transcript": ev.transcript})

        @session.on("conversation_item_added")
        def _on_said(ev: ConversationItemAddedEvent) -> None:
            if isinstance(ev.item, llm.ChatMessage) and ev.item.role == "assistant":
                logger.info(f"[{route}] said", extra={"lk.pii.transcript": ev.item.text_content})

        @session.on("close")
        def _on_close(ev: CloseEvent) -> None:
            # a session DeepL ended is dropped; the next change in the room starts it again
            if self._sessions.get(route) is session:
                logger.warning(f"{route} closed: {ev.reason}")
                del self._sessions[route]
                self._update_permissions()

        await session.start(
            agent=Agent(instructions=""),
            room=self._room,
            room_options=room_io.RoomOptions(
                participant_identity=route.speaker,
                # a room takes one chat and one pre-connect audio handler, not one per session
                text_input=False,
                audio_input=room_io.AudioInputOptions(pre_connect_audio=False),
                audio_output=room_io.AudioOutputOptions(track_name=route.track_name),
                # the speaker leaving ends the session through _reconcile
                close_on_disconnect=False,
            ),
        )
        return session

    def _update_permissions(self) -> None:
        local = self._room.local_participant
        if not PRIVATE_TRACKS:
            local.set_track_subscription_permissions(allow_all_participants=True)
            return

        # granted by the routes wanted, not the sessions started: a track publishes while its
        # session is still starting
        sids = {publication.name: sid for sid, publication in local.track_publications.items()}
        wanted = self._wanted()
        permissions = [
            rtc.ParticipantTrackPermission(
                participant_identity=identity,
                allow_all=False,
                allowed_track_sids=[
                    sids[route.track_name]
                    for route in wanted
                    if route.target == language and route.track_name in sids
                ],
            )
            for identity, language in self._languages().items()
        ]
        local.set_track_subscription_permissions(
            allow_all_participants=False, participant_permissions=permissions
        )


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    interpreters = Interpreters(ctx.room)
    ctx.add_shutdown_callback(interpreters.aclose)
    interpreters.start()


if __name__ == "__main__":
    cli.run_app(server)
