"""Python SDK and application callback adaptation for the native session owner."""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any, cast

from livekit import rtc

from .. import _bindings, _core, llm
from .._core_view import realtime_model
from .._generated.voice_session import VoiceSessionArgs
from ..job import get_job_context
from ..llm import ChatContext
from ..types import NotGivenOr
from ..utils import is_given
from .agent import Agent
from .events import CloseReason
from .remote_session import ConsoleCall, RemoteSessionAccess
from .room_io import Room as _RtcRoom, RoomOptions

if TYPE_CHECKING:
    from .agent_session import VoiceSession


class SessionHost:
    def __init__(
        self,
        session: VoiceSession[Any],
        agent: Agent,
        room: NotGivenOr[rtc.Room | None],
        room_options: NotGivenOr[RoomOptions],
        chat_ctx: ChatContext | None,
    ) -> None:
        self.session = session
        self.agent = agent
        self.room = room
        self.room_options = room_options
        self.chat_ctx = chat_ctx
        thread = session._thread
        self.storage = _core._core_SessionStorage(
            thread._storage.threads if thread is not None else None,
            thread.id if thread is not None else None,
            thread._agent_name if thread is not None else agent.name,
            agent.name,
            chat_ctx.items if chat_ctx is not None else None,
        )

    async def prepare(self) -> _core._core_SessionBindings:
        from ._session_bindings import _NODES, _detection_llm, _record_keyterms
        from ._session_callbacks import _AgentHooks
        from ._session_types import _HOOK_BOUND, RecordingOptions

        session, agent = self.session, self.agent
        room, room_options, chat_ctx = self.room, self.room_options, self.chat_ctx
        console = ConsoleCall.claim()
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            session._event_retention.claim(job_ctx.job.id)
        if console is not None:
            room = None
        elif not is_given(room):
            room = job_ctx.room if job_ctx is not None else None
        if chat_ctx is not None:
            session._chat_ctx = chat_ctx.copy()
        options = room_options if is_given(room_options) else RoomOptions()
        session._audio_input_options = options.get_audio_input_options()
        text_input = options.get_text_input_options()
        session._text_input_cb = text_input.text_input_cb if text_input is not None else None
        # the job's session (on its room, or the console's) records and reports for the job and
        # closes with it; the primary one, and the console's, serve the remote session protocol
        in_job = job_ctx is not None and (console is not None or room is job_ctx.room)
        hosting = console is not None
        if in_job:
            assert job_ctx is not None
            plan = _core.plan_session_start(
                job_default=job_ctx.job.enable_recording,
                another_session_is_primary=not job_ctx._claim_session(session),
            )
            session._recording_options = cast(RecordingOptions, dataclasses.asdict(plan.recording))
            hosting = hosting or plan.hosting
            job_ctx.init_recording(session._recording_options)

        from ..storage.backends import _MemoryStorage
        from ..storage.context_store import ContextStore
        from ..storage.resolve import Thread

        storage = session._owner.storage
        assert storage is not None
        session._store = ContextStore(activation=storage.activation, scope_id=storage.scope_id)
        if session._thread is None:
            session._thread = Thread(
                _MemoryStorage(native=storage.namespace), storage.thread_id, agent_name=agent.name
            )
        if (saved := storage.history) is not None:
            session._chat_ctx = ChatContext(list(saved))

        hooks = _AgentHooks(session)
        frame = session._frame(agent)
        model = frame.get("llm", session._llm)
        # an OpenAI-protocol model or a fallback chain runs in the core; any other one is a host
        # realtime model
        realtime = (
            realtime_model(model, lambda: session._adapter._pool(session.current_agent))
            if isinstance(model, llm.RealtimeModel)
            else None
        )
        sampler = session._video_sampler
        keyterm_llm = _detection_llm(session._stt_context)
        session._room = room
        args = VoiceSessionArgs(
            instructions=frame["instructions"],
            dynamic_instructions=frame["dynamic_instructions"],
            agent=frame["agent"],
            agent_host=frame["agent_host"],
            chat_ctx=session._chat_ctx.items,
            stt=frame["stt"],
            vad=_bindings.core_leg(session._vad),
            llm=None if isinstance(model, llm.RealtimeModel) else model,
            realtime=realtime,
            tts=frame["tts"],
            host_audio_input=session._input.audio,
            host_video_input=session._input.video,
            host_audio_output=session._output.audio,
            host_transcription_output=session._output.transcription,
            on_event=session,
            hooks=hooks,
            session_toolsets=hooks._core_toolset_resources(session._tools),
            tools_host=frame["tools_host"],
            voice_agent=frame["voice_agent"],
            room_io_config=options._to_core(),
            video_sampler=(sampler.speaking_fps, sampler.silent_fps) if sampler else None,
            using_default_vad=session._using_default_vad,
            turn_detector=session._turn_detector,
            hook_bound=_HOOK_BOUND,
            max_unrecoverable_errors=session._conn_options.max_unrecoverable_errors,
            interruption_detector=session._adapter._interruption_detector(agent, model),
            stt_context=session._stt_context,
            keyterm_llm=_bindings.core_leg(keyterm_llm) if keyterm_llm is not None else None,
            keyterm_tools=[_record_keyterms] if keyterm_llm is not None else None,
            **{node: frame[node] for node in _NODES if node in frame},
            remote_session=RemoteSessionAccess(session) if hosting else None,
            console=console,
            # the core connects the job's room beside its opening
            job=job_ctx if in_job and console is None else None,
        )
        session._started_at = time.time()
        if in_job:
            assert job_ctx is not None
            if session._recording_options["audio"] or (console is not None and ConsoleCall.record):
                session._recording_path = job_ctx.session_directory / "audio.ogg"
                session._recorder = _core.SessionRecorder(str(session._recording_path))
            job_ctx.add_shutdown_callback(lambda: session._aclose(CloseReason.JOB_SHUTDOWN))
        rtc_room = _RtcRoom.of(room) if room is not None else None
        return _core._core_SessionBindings(
            room=_core._core_create_room_session(rtc_room), args=args, recorder=session._recorder
        )

    async def started(self) -> None:
        if self.session.options["ivr_detection"]:
            await self.session._start_ivr_detection()

    async def closed(self) -> None:
        session = self.session
        if session._store is not None:
            session._store._close()
        audio_in = session._audio_input_options
        if audio_in is not None and isinstance(audio_in.noise_cancellation, rtc.FrameProcessor):
            audio_in.noise_cancellation._close()
