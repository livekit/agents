"""Realtime multi-user speech translation using a realtime translation model.

The backend is selectable via the ``TRANSLATION_PROVIDER`` env var ("openai" — the
default — or "google"), since both ``openai.realtime.RealtimeTranslationModel`` and
``google.realtime.RealtimeTranslationModel`` expose the same interface.

Every participant speaks their own language (declared via a ``language`` token
attribute) and hears every other participant translated into their own language.

Architecture (one *agent participant* per translation direction):
- A hidden "watcher" participant discovers who is in the room and what language
  each speaks. It is joined ``hidden=True`` so it never shows up for anyone.
- For each (source participant -> target language) pair, a dedicated agent
  participant joins the room (identity ``xlator-<source>-<lang>``) and runs an
  ``AgentSession`` whose ``llm`` is a :class:`RealtimeTranslationModel`. The
  session's RoomIO is bound to the source participant
  (``RoomOptions(participant_identity=...)``) for input and publishes the
  translated speech from that agent participant. In ``on_enter`` the agent sets
  **track-subscription permissions** so only the recipients (participants who
  speak the target language) may subscribe to that translated audio — so A->B is
  only audible to B and B->A only to A.

Original-voice ducking (``_ORIGINAL_AUDIO_VOLUME``):
- ``>= 1.0``: listeners hear each other's original audio directly at full volume
  (no ducking).
- ``0 < v < 1.0``: listeners are force-unsubscribed (admin API) from each other's
  raw mics, and each translator agent **relays** its source's audio at volume
  ``v`` — so the original is audible faintly *under* the translation.
- ``<= 0``: original is muted entirely (only translations are heard).
  NOTE: this server-side ducking only works because we control subscriptions via
  the admin API; with a purpose-built client the cleaner approach is to set the
  original track's playout volume client-side.

No STT/LLM/TTS plugins are used — the realtime model transcribes, translates and
synthesizes in one step. Each participant's token must carry a ``language``
attribute (e.g. ``en``/``de``); participants without one are assumed to speak
English.

How to run
----------
Set ``LIVEKIT_URL``, ``LIVEKIT_API_KEY`` and ``LIVEKIT_API_SECRET`` (plus the
provider key, e.g. ``GOOGLE_API_KEY`` / ``OPENAI_API_KEY``) in your env or a
``.env`` file, then pick a backend via ``TRANSLATION_PROVIDER`` ("openai" — the
default — or "google").

1) Connect to an already-existing room (the agent joins immediately and starts
   reconciling translation directions for whoever is in the room)::

       TRANSLATION_PROVIDER=google python examples/other/translation/realtime-multi-user-translator.py connect --room translate-test

2) Deploy as a worker that waits for incoming jobs from the LiveKit server.
   The agent is dispatched on demand (agent name ``realtime-translator``)::

       # development, with hot reload
       TRANSLATION_PROVIDER=google python examples/other/translation/realtime-multi-user-translator.py dev

       # production
       TRANSLATION_PROVIDER=google python examples/other/translation/realtime-multi-user-translator.py start
"""

import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv

from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobRequest,
    cli,
    llm,
    room_io,
    utils,
)
from livekit.plugins import google, openai

load_dotenv()

logger = logging.getLogger("realtime-translator")

# realtime translation backend: "openai" (default) or "google"
TRANSLATION_PROVIDER = os.getenv("TRANSLATION_PROVIDER", "openai").lower()


def _build_translation_model(target_language: str) -> llm.RealtimeModel:
    if TRANSLATION_PROVIDER == "google":
        return google.realtime.RealtimeTranslationModel(target_language=target_language)
    return openai.realtime.RealtimeTranslationModel(target_language=target_language)


# identity prefix for the per-direction agent participants we publish from
_XLATOR_PREFIX = "xlator-"
_WATCHER_IDENTITY = "xlator-watcher"
# how loud the other peer's *original* (untranslated) voice is, 0.0 - 1.0.
# >= 1.0 keeps the direct audio untouched; lower values route the original
# through the agent at reduced volume; 0 mutes it.
_ORIGINAL_AUDIO_VOLUME = 0.3
# the audio source frames the relay publishes at
_RELAY_SAMPLE_RATE = 48000


@dataclass
class Language:
    code: str
    name: str


_languages = [
    Language(code="ar", name="Arabic"),
    Language(code="cs", name="Czech"),
    Language(code="de", name="German"),
    Language(code="el", name="Greek"),
    Language(code="en", name="English"),
    Language(code="es", name="Spanish"),
    Language(code="fr", name="French"),
    Language(code="hi", name="Hindi"),
    Language(code="hr", name="Croatian"),
    Language(code="it", name="Italian"),
    Language(code="ja", name="Japanese"),
    Language(code="ko", name="Korean"),
    Language(code="nl", name="Dutch"),
    Language(code="pl", name="Polish"),
    Language(code="pt", name="Portuguese"),
    Language(code="ru", name="Russian"),
    Language(code="tr", name="Turkish"),
    Language(code="uk", name="Ukrainian"),
    Language(code="zh", name="Chinese"),
]

language_map: dict[str, Language] = {lang.code: lang for lang in _languages}


def language_name(code: str) -> str:
    lang = language_map.get(code)
    return lang.name if lang else code


def _scaled_frame(frame: rtc.AudioFrame, volume: float) -> rtc.AudioFrame:
    """Return a copy of ``frame`` with its PCM16 samples scaled by ``volume``."""
    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) * volume
    np.clip(samples, -32768, 32767, out=samples)
    return rtc.AudioFrame(
        data=samples.astype(np.int16).tobytes(),
        sample_rate=frame.sample_rate,
        num_channels=frame.num_channels,
        samples_per_channel=frame.samples_per_channel,
    )


class TranslationAgent(Agent):
    """Wraps the realtime translation model and locks its audio to an audience."""

    def __init__(self, *, target_language: str, audience: list[str]) -> None:
        super().__init__(
            instructions="",
            llm=_build_translation_model(target_language),
        )
        self._audience = list(audience)

    async def on_enter(self) -> None:
        self.reapply_permissions()

    def update_audience(self, audience: list[str]) -> None:
        self._audience = list(audience)
        self.reapply_permissions()

    def reapply_permissions(self) -> None:
        # only the recipients may subscribe to this direction's audio (translation
        # and the optional original relay are both published by this participant)
        if self.session is None:
            return
        local = self.session.room_io.room.local_participant
        local.set_track_subscription_permissions(
            allow_all_participants=False,
            participant_permissions=[
                rtc.ParticipantTrackPermission(participant_identity=identity, allow_all=True)
                for identity in self._audience
            ],
        )


class TranslationDirection:
    """One translation direction: a dedicated agent participant + AgentSession."""

    def __init__(
        self,
        *,
        server_url: str,
        api_key: str,
        api_secret: str,
        room_name: str,
        source_identity: str,
        source_language: str,
        target_language: str,
        audience: list[str],
    ):
        self._server_url = server_url
        self._api_key = api_key
        self._api_secret = api_secret
        self._room_name = room_name
        self._source_identity = source_identity
        self._source_language = source_language
        self._target_language = target_language
        self._audience = list(audience)

        self._room = rtc.Room()
        self._session = AgentSession(aec_warmup_duration=None)
        self._agent: TranslationAgent | None = None
        self._relay_task: asyncio.Task | None = None

    @property
    def identity(self) -> str:
        return f"{_XLATOR_PREFIX}{self._source_identity}-{self._target_language}"

    def _token(self) -> str:
        return (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(self.identity)
            .with_name(f"{language_name(self._target_language)} translator")
            .with_kind("agent")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self._room_name,
                    can_publish=True,
                    can_publish_data=True,
                    can_subscribe=True,
                    can_update_own_metadata=True,
                )
            )
            .to_jwt()
        )

    @utils.log_exceptions(logger=logger)
    async def start(self) -> None:
        await self._room.connect(self._server_url, self._token())
        self._agent = TranslationAgent(
            target_language=self._target_language, audience=self._audience
        )
        await self._session.start(
            room=self._room,
            agent=self._agent,
            room_options=room_io.RoomOptions(
                # listen only to the source participant; the model translates it
                participant_identity=self._source_identity,
                audio_input=room_io.AudioInputOptions(pre_connect_audio=False),
                video_input=False,
                close_on_disconnect=True,
            ),
        )
        logger.info(
            f"started {language_name(self._source_language)} -> "
            f"{language_name(self._target_language)} translator for "
            f"{self._source_identity}, audience={self._audience}"
        )
        if 0 < _ORIGINAL_AUDIO_VOLUME < 1.0:
            self._relay_task = asyncio.create_task(self._relay_original())

    @utils.log_exceptions(logger=logger)
    async def _relay_original(self) -> None:
        """Re-publish the source's original voice at reduced volume to the audience."""
        # wait for the source participant to be present in our room
        participant: rtc.RemoteParticipant | None = None
        for _ in range(50):
            participant = self._room.remote_participants.get(self._source_identity)
            if participant is not None:
                break
            await asyncio.sleep(0.1)
        if participant is None:
            logger.warning(f"relay: source {self._source_identity} not found")
            return

        stream = rtc.AudioStream.from_participant(
            participant=participant,
            track_source=rtc.TrackSource.SOURCE_MICROPHONE,
            sample_rate=_RELAY_SAMPLE_RATE,
            num_channels=1,
        )
        source = rtc.AudioSource(_RELAY_SAMPLE_RATE, 1)
        track = rtc.LocalAudioTrack.create_audio_track(f"original-{self._source_identity}", source)
        await self._room.local_participant.publish_track(track, rtc.TrackPublishOptions())
        # make sure the relay track is restricted to the audience too
        if self._agent is not None:
            self._agent.reapply_permissions()

        try:
            async for ev in stream:
                await source.capture_frame(_scaled_frame(ev.frame, _ORIGINAL_AUDIO_VOLUME))
        finally:
            await stream.aclose()

    def update_audience(self, audience: list[str]) -> None:
        if audience == self._audience:
            return
        self._audience = list(audience)
        if self._agent is not None:
            self._agent.update_audience(self._audience)

    async def aclose(self) -> None:
        if self._relay_task is not None:
            await utils.aio.cancel_and_wait(self._relay_task)
        with contextlib.suppress(Exception):
            await self._session.aclose()
        with contextlib.suppress(Exception):
            await self._room.disconnect()


class TranslationManager:
    """Hidden watcher that reconciles one :class:`TranslationDirection` per (source, target language)."""  # noqa: E501

    def __init__(self, *, server_url: str, api_key: str, api_secret: str, room_name: str):
        self._server_url = server_url
        self._api_key = api_key
        self._api_secret = api_secret
        self._room_name = room_name

        self._watcher = rtc.Room()
        self._directions: dict[str, TranslationDirection] = {}
        self._closed = asyncio.Event()
        self._lkapi: api.LiveKitAPI | None = None

    def _watcher_token(self) -> str:
        return (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(_WATCHER_IDENTITY)
            .with_kind("agent")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self._room_name,
                    hidden=True,  # invisible to the human participants
                    can_subscribe=False,
                    room_admin=True,  # needed to manage peer subscriptions for ducking
                )
            )
            .to_jwt()
        )

    async def start(self) -> None:
        self._lkapi = api.LiveKitAPI(self._server_url, self._api_key, self._api_secret)
        self._watcher.on("participant_connected", lambda _p: self._reconcile())
        self._watcher.on("participant_disconnected", lambda _p: self._reconcile())
        self._watcher.on("participant_attributes_changed", lambda *_a: self._reconcile())
        self._watcher.on("track_published", lambda *_a: asyncio.create_task(self._isolate_humans()))
        self._watcher.on("disconnected", lambda *_a: self._closed.set())
        await self._watcher.connect(
            self._server_url, self._watcher_token(), rtc.RoomOptions(auto_subscribe=False)
        )
        self._reconcile()

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def aclose(self) -> None:
        for direction in list(self._directions.values()):
            await direction.aclose()
        self._directions.clear()
        with contextlib.suppress(Exception):
            await self._watcher.disconnect()
        if self._lkapi is not None:
            await self._lkapi.aclose()

    def _humans(self) -> dict[str, str]:
        """identity -> language, for the human participants in the room."""
        humans: dict[str, str] = {}
        for participant in self._watcher.remote_participants.values():
            # ignore agents: our own translators/watcher and the dispatched job
            # agent (in `connect` mode the CLI joins the agent to the room).
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                continue
            if participant.identity.startswith(_XLATOR_PREFIX):
                continue
            humans[participant.identity] = participant.attributes.get("language") or "en"
        return humans

    async def _isolate_humans(self) -> None:
        """Force-unsubscribe each human from the others' raw mics (for ducking).

        Only needed when the original is ducked below full volume; the translator
        agents then relay the original at the configured volume instead.
        """
        if self._lkapi is None or _ORIGINAL_AUDIO_VOLUME >= 1.0:
            return

        audio_sids: dict[str, list[str]] = {}
        for participant in self._watcher.remote_participants.values():
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                continue
            if participant.identity.startswith(_XLATOR_PREFIX):
                continue
            sids = [
                pub.sid
                for pub in participant.track_publications.values()
                if pub.kind == rtc.TrackKind.KIND_AUDIO
            ]
            if sids:
                audio_sids[participant.identity] = sids

        for listener in audio_sids:
            others = [
                sid for owner, sids in audio_sids.items() if owner != listener for sid in sids
            ]
            if not others:
                continue
            with contextlib.suppress(Exception):
                await self._lkapi.room.update_subscriptions(
                    api.UpdateSubscriptionsRequest(
                        room=self._room_name,
                        identity=listener,
                        track_sids=others,
                        subscribe=False,
                    )
                )

    def _reconcile(self) -> None:
        humans = self._humans()
        speakers_by_language: dict[str, list[str]] = {}
        for identity, language in humans.items():
            speakers_by_language.setdefault(language, []).append(identity)

        desired: dict[str, tuple[str, str, str, list[str]]] = {}
        for source, source_lang in humans.items():
            for target_lang, listeners in speakers_by_language.items():
                if target_lang == source_lang:
                    continue
                audience = [i for i in listeners if i != source]
                if not audience:
                    continue
                key = f"{source}->{target_lang}"
                desired[key] = (source, source_lang, target_lang, audience)

        for key, (source, source_lang, target_lang, audience) in desired.items():
            existing = self._directions.get(key)
            if existing is None:
                direction = TranslationDirection(
                    server_url=self._server_url,
                    api_key=self._api_key,
                    api_secret=self._api_secret,
                    room_name=self._room_name,
                    source_identity=source,
                    source_language=source_lang,
                    target_language=target_lang,
                    audience=audience,
                )
                self._directions[key] = direction
                asyncio.create_task(direction.start())
            else:
                existing.update_audience(audience)

        for key in list(self._directions.keys()):
            if key not in desired:
                direction = self._directions.pop(key)
                asyncio.create_task(direction.aclose())

        asyncio.create_task(self._isolate_humans())


async def request_fnc(req: JobRequest) -> None:
    await req.accept(name="agent", identity="agent")


server = AgentServer()


@server.rtc_session(agent_name="realtime-translator", on_request=request_fnc)
async def entrypoint(ctx: JobContext) -> None:
    # NOTE: we deliberately do NOT call ctx.connect(): the dispatched agent should
    # not appear as a participant. Discovery happens via a hidden watcher, and each
    # translation direction joins as its own agent participant.
    manager = TranslationManager(
        server_url=os.environ["LIVEKIT_URL"],
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
        room_name=ctx.job.room.name,
    )
    try:
        await manager.start()
        await manager.wait_closed()
    finally:
        await manager.aclose()


if __name__ == "__main__":
    cli.run_app(server)
