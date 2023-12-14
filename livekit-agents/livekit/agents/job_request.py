import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Coroutine, Optional, TYPE_CHECKING
from .job_context import JobContext
from livekit import rtc, protocol, api
from livekit.protocol import agent as proto_agent

# TODO: refactor worker so we can avoid this circular import
if TYPE_CHECKING:
    from .worker import Worker


@dataclass
class SubscribeOptions:
    """
    SubscribeOptions is used to configure how the agent subscribes to tracks in the room.
    """

    predicate: Callable[[rtc.TrackPublication, rtc.RemoteParticipant], bool]
    _livekit_auto_subscribe = False

    @staticmethod
    def subscribe_all() -> "SubscribeOptions":
        """
        Subscribe to all tracks automatically. This will also set the LiveKit room option auto_subscribe to true as an optimization.
        """
        so = SubscribeOptions(predicate=lambda *_: True)
        so._livekit_auto_subscribe = True
        return so

    @staticmethod
    def subscribe_none() -> "SubscribeOptions":
        """Don't subscribe to any tracks automatically

        Returns:
            _type_: _description_
        """
        return SubscribeOptions(predicate=lambda *_: False)

    @staticmethod
    def video_only() -> "SubscribeOptions":
        """
        Subscribe to video tracks automatically
        """
        return SubscribeOptions(
            predicate=lambda p, _: p.kind == rtc.TrackKind.KIND_VIDEO
        )

    @staticmethod
    def audio_only() -> "SubscribeOptions":
        """
        Subscribe to audio tracks automatically
        """
        return SubscribeOptions(
            predicate=lambda p, _: p.kind == rtc.TrackKind.KIND_AUDIO
        )


@dataclass
class ShutdownOptions:
    """
    ShutdownOptions is used to configure when the agent should automatically disconnect from the room.
    """

    predicate: Callable[["JobContext"], bool]
    """
    Given a JobContext, decides whether the agent should automatically disconnect.
    This is called whenever the participants in the room change or the track
    publications in the room change. None indicates not to auto disconnect. Defaults to None.
    """

    task_timeout: Optional[float]
    """
    Grace period to wait before cancelling tasks created by JobContext.create_task().
    """

    @staticmethod
    def room_empty(
        task_timeout: Optional[float] = 25,
    ) -> "ShutdownOptions":
        def predicate(ctx: JobContext) -> Coroutine:
            # Hot path, if there are no participants, we don't need to check
            if len(ctx.room.participants) == 0:
                return True

            # Hot path, if there are more than 1 participants, we don't need to check
            if len(ctx.room.participants) > 1:
                return False

            # If only participant is the agent
            for p in ctx.room.participants.values():
                if p.identity == ctx.agent_identity:
                    return True

            return False

        return ShutdownOptions(predicate=predicate, task_timeout=10)

    @staticmethod
    def job_publisher_left(task_timeout: Optional[float] = 25):
        def predicate(ctx: "JobContext") -> Coroutine:
            if ctx.participant is None:
                logging.error(
                    "Incorrect usage of ShutdownOptions, the JobContext is not tied to a Participant"
                )
                return False

            return ctx.room.participants.get(ctx.participant.sid) is None

        return ShutdownOptions(predicate=predicate, task_timeout=task_timeout)

    @staticmethod
    def default(task_timeout: Optional[float] = 25):
        """
        Default shutdown options. If the agent is tied to a participant, it will shut down when that participant leaves.
        If the agent is not tied to a participant, it will shut down when the agent is the only remaining participant.

        Args:
            task_timeout (Optional[float], optional): _description_. Defaults to 25.
        """

        def predicate(ctx: "JobContext") -> Coroutine:
            part_so = ShutdownOptions.job_publisher_left(task_timeout=task_timeout)
            room_so = ShutdownOptions.room_empty(task_timeout=task_timeout)
            if ctx.participant is not None:
                return part_so.predicate(ctx)

            return room_so.predicate(ctx)

        return ShutdownOptions(predicate=predicate, task_timeout=task_timeout)

    @staticmethod
    def auto_shutdown_disabled():
        return ShutdownOptions(predicate=lambda _: False, task_timeout=None)


class JobRequest:
    """
    Represents a new job from the server, this worker can either accept or reject it.
    """

    def __init__(
        self,
        worker: "Worker",
        job_info: proto_agent.Job,
        simulated: bool = False,
    ) -> None:
        self._worker = worker
        self._info = job_info
        self._room = rtc.Room()
        self._answered = False
        self._simulated = simulated
        self._lock = asyncio.Lock()

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def room(self) -> protocol.models.Room:
        return self._info.room

    async def reject(self) -> None:
        """
        Tell the server that we cannot handle the job
        """
        async with self._lock:
            if self._answered:
                raise Exception("job already answered")

            self._answered = True
            if not self._simulated:
                await self._worker._send_availability(self.id, False)

        logging.info("rejected job %s", self.id)

    async def accept(
        self,
        agent: Callable[[JobContext], Coroutine],
        subscribe_options: SubscribeOptions = SubscribeOptions.subscribe_none(),
        shutdown_options: ShutdownOptions = ShutdownOptions.auto_shutdown_disabled(),
        grants: api.VideoGrants = None,
        name: str = "",
        identity: str = "",
        metadata: str = "",
    ) -> None:
        """
        Signal to the LiveKit Server that we can handle the job, and connect to the room.

        Args:
            agent (Callable[[JobContext], Coroutine]):
                Your agent entrypoint.

            subscribe_options (Callable[[rtc.TrackPublication, rtc.RemoteParticipant], bool]):

            shutdown_options (ShutdownOptions):

            grants (api.VideoGrants, optional):
                Additional grants to give to the agent participant in its token.
                Defaults to None.

            name (str, optional):
                Name of the agent participant. Defaults to "".

            identity (str, optional):
                Identity of the agent participant. Defaults to "".

            metadata (str, optional):
                Metadata of the agent participant. Defaults to "".
        """
        async with self._lock:
            if self._answered:
                raise Exception("job already answered")

            self._answered = True

            identity = identity or "agent-" + self.id
            grants = grants or api.VideoGrants()
            grants.room_join = True
            grants.agent = True
            grants.room = self.room.name
            grants.can_update_own_metadata = True

            jwt = (
                api.AccessToken(self._worker._api_key, self._worker._api_secret)
                .with_identity(identity)
                .with_grants(grants)
                .with_metadata(metadata)
                .with_name(name)
                .to_jwt()
            )

            # raise AssignmentTimeoutError if assignment times out
            if not self._simulated:
                _ = await self._worker._send_availability(self.id, True)

            try:
                options = rtc.RoomOptions(
                    auto_subscribe=subscribe_options._livekit_auto_subscribe
                )
                await self._room.connect(self._worker._rtc_url, jwt, options)
            except rtc.ConnectError as e:
                logging.error(
                    "failed to connect to the room, cancelling job %s: %s", self.id, e
                )
                await self._worker._send_job_status(
                    self.id, proto_agent.JobStatus.JS_FAILED, str(e)
                )
                raise e

            participant: Optional[rtc.Participant] = None
            if self._info.participant:
                participant = self._room.participants.get(self._info.participant.sid)

            job_ctx = JobContext(
                self.id,
                self._worker,
                self._room,
                participant=participant,
                agent_identity=identity,
            )

            def done_callback(t: asyncio.Task):
                try:
                    if t.cancelled():
                        logging.info(
                            "Task was cancelled. Worker: %s Job: %s",
                            self._worker.id,
                            self.id,
                        )
                    else:
                        logging.info(
                            "Task completed successfully. Worker: %s Job: %s",
                            self._worker.id,
                            self.id,
                        )
                except asyncio.CancelledError:
                    logging.info(
                        "Task was cancelled. Worker: %s Job: %s",
                        self._worker.id,
                        self.id,
                    )
                except Exception as e:
                    logging.error(
                        "Task raised an uncaught exception. Worker: %s Job: %s Exception: %s",
                        self._worker.id,
                        self.id,
                        e,
                    )

            task = self._worker._loop.create_task(agent(job_ctx))
            task.add_done_callback(done_callback)

            def shutdown_if_needed(*_):
                if shutdown_options.predicate(job_ctx):
                    asyncio.ensure_future(
                        job_ctx.shutdown(task_timeout=shutdown_options.task_timeout),
                        loop=self._worker._loop,
                    )

            @self._room.on("track_published")
            def on_track_published(
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                shutdown_if_needed()

                if not subscribe_options.predicate(publication, participant):
                    return

                publication.set_subscribed(True)

            self._room.on("participant_connected", shutdown_if_needed)
            self._room.on("participant_disconnected", shutdown_if_needed)
            self._room.on("track_unpublished", shutdown_if_needed)

            for participant in self._room.participants.values():
                for publication in participant.tracks.values():
                    if not subscribe_options.predicate(publication, participant):
                        continue

                    publication.set_subscribed(True)

            # Call shutdown_if_needed() once to check if the conditions
            # for auto disconnect are already met
            shutdown_if_needed()

        logging.info("accepted job %s", self.id)
