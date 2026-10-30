//! The session decision core and the input vocabulary its runtime queue carries.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::VoiceSession;
use crate::agent_session::commands::{
    Answer, Close, Command, HostRecorder, IvrCommand, Origin, RequestId, Sent, SpeechCall,
};
use crate::agent_session::effects::{Out, Timer};
use crate::agent_session::time::Now;
use crate::audio::ChannelEnd;
use crate::events::SessionEvent;
use crate::generation::GenEvent;
use crate::room_io::StreamOpened;
use crate::speech_handle::{SpeechState, SpeechUpdate};
use crate::trace::{CloseAct, CloseStep, CoreStep, PlayoutStep, RoomStep, SpeechStep, Step};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::voice::rtc::RoomEvent;
use lk_agents_abi::{FrameId, SpeechEvent, TimerSet, Ts, VadEvent};
use std::sync::Arc;

/// Only what a host could read, in loop order; never machine state.
#[derive(Debug, Clone)]
pub enum SystemTap {
    Session(SessionEvent),
    Vad(lk_agents_abi::VadEvent),
    Stt(lk_agents_abi::SpeechEvent),
    Ivr(IvrCommand),
    CloseRequested,
}

/// What each source sends, as the loop receives it.
macro_rules! queue_events {
    ($($ty:ty => |$v:pat_param| $event:expr;)*) => {$(
        impl From<$ty> for Event {
            fn from($v: $ty) -> Event {
                $event
            }
        }
    )*};
}
queue_events! {
    Input => |input| Event::Input(input);
    crate::data_plane::ForwarderCommand => |command| Event::Forwarder(command);
    Command => |command| Event::Input(Input::Host(Sent { origin: Origin::Host, command }));
    crate::audio::Tagged<SpeechEvent> => |ev| Event::Input(Input::Stt(ev, None));
    VadEvent => |ev| Event::Input(Input::Vad(ev));
    RoomEvent => |ev| Event::Input(Input::Room(ev));
    crate::room_io::RoomIoEvent => |ev| Event::Input(Input::RoomIo(ev));
    GenEvent => |ev| Event::Input(Input::Gen(ev));
    crate::agent_activity::realtime::RealtimeWake => |wake| Event::Input(Input::Realtime(wake));
    lk_agents_abi::turn::detector::LegEvent<crate::audio_recognition::OverlapVerdict> =>
        |ev| Event::Input(Input::Overlap(ev));
    ChannelEnd => |end| Event::Input(Input::ChannelEnded(end));
    StreamOpened => |opened| Event::LegOpened(opened);
    lk_agents_abi::voice::rtc::VideoFrameRef => |frame| Event::Input(Input::Video(frame));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueClosed;

/// Timers are the loop's own.
#[derive(lk_agents_abi::Vocabulary)]
pub enum Input {
    Host(Sent),
    Stt(crate::audio::Tagged<SpeechEvent>, Option<lk_agents_abi::ProviderId>),
    Vad(VadEvent),
    Room(RoomEvent),
    /// Forwarded from the loop so they share one arrival order.
    RoomIo(crate::room_io::RoomIoEvent),
    Gen(GenEvent),
    /// The lane storage gave back for a speech, and the request items it was read with.
    StorageReloaded {
        read: Vec<lk_agents_abi::ChatItem>,
        history: Vec<lk_agents_abi::ChatItem>,
    },
    StorageSettled(crate::agent_activity::scheduling::SpeechId),
    StorageAbandoned(crate::agent_activity::scheduling::SpeechId),
    StorageLost(crate::agent_activity::scheduling::SpeechId),
    Realtime(crate::agent_activity::realtime::RealtimeWake),
    Eot(
        lk_agents_abi::turn::detector::ReqId,
        Result<lk_agents_abi::EotPrediction, lk_agents_abi::errors::ProviderError>,
        Option<lk_agents_abi::tokenize::language::ThresholdOptions>,
    ),
    Overlap(lk_agents_abi::turn::detector::LegEvent<crate::audio_recognition::OverlapVerdict>),
    ChannelEnded(ChannelEnd),
    /// The STT stream failed an attempt it retries.
    SttRetried(lk_agents_abi::errors::ProviderError),
    StreamOpened {
        open: u64,
        leg: crate::agent_session::effects::StreamKind,
        outcome: Result<lk_agents_abi::resources::ResourceId, ProviderError>,
    },
    Video(lk_agents_abi::voice::rtc::VideoFrameRef),
    /// The data plane's reports. The source's first frame forwarded (`input_started_at`).
    InputStarted {
        at: Ts,
    },
    /// Silence substituted per channel since the previous gate change, at a gate change.
    GateSilenced(Vec<(crate::audio::AudioChannel, u64)>),
    InputCensus(crate::data_plane::InputCensus),
}

/// What the runtime queue owner handles: an event in arrival order, or time reaching
/// [`Driver::wake_at`].
pub enum Event {
    /// For the decision core.
    Input(Input),
    /// For the decision core, from a sender awaiting the answer to the call it names: the
    /// runtime keeps the call's reply until the core answers it.
    Awaited(Input, RequestId),
    /// A provider completion, adopted into runtime resources before entering the core.
    LegOpened(StreamOpened),
    /// For the data plane, from the decision core.
    Forwarder(crate::data_plane::ForwarderCommand),
    Wake,
    /// The host asked something of its speeches: the requests wait in their own list, which the
    /// runtime steps before anything else it takes.
    HostRequests,
}

/// The session's decision core: handed every [`Input`] in arrival order by the
/// runtime queue owner, it merges them with its timers and orchestrates the
/// components in sequence: the [`AgentSession`], its running
/// [`AgentActivity`](crate::agent_activity::AgentActivity), and the [`AgentInput`] legs. Every
/// decision is journalled, dated on the session clock.
pub struct Driver {
    pub(crate) session: VoiceSession,
    pub(crate) activity: crate::agent_activity::AgentActivity,
    pub(crate) input: crate::agent_activity::io::AgentInput,
    /// Machines never see it; [`Out`] carries it for them.
    pub(crate) current: FrameId,
    pub(crate) timers: TimerSet<Timer>,
    pub(crate) journalled: bool,
    /// Decides ownership (a tool's hold is its run's).
    pub(crate) command_origin: Origin,
    /// The time of the step being handled, or of the last one.
    pub(crate) step: crate::agent_session::time::StepClock,
    pub(crate) stopped: bool,
    pub(super) opened: bool,
    /// The activity changed this cycle: published once, as the cycle's effects are done.
    pub(crate) activity_changed: std::cell::Cell<bool>,
    /// What the core decided outside a step's [`Out`] (opening, the loop's end, published views),
    /// returned with the step's own effects.
    pub(crate) outbox: Vec<crate::agent_session::effects::Effect>,
    /// The [`SessionView`](crate::agent_session::SessionView) last published, so a step that changed
    /// none of it publishes nothing.
    pub(crate) view: Arc<crate::agent_session::SessionView>,
}

impl Driver {
    /// Only while performing an effect, when no cycle exists.
    pub(crate) fn journal(&mut self, step: Step) {
        if !self.journalled {
            return;
        }
        self.outbox.push(crate::agent_session::effects::Effect::Journal(Box::new(step)));
    }

    /// An event sent to the host where no cycle exists, journalled first.
    pub(crate) fn emit_now(&mut self, ev: SessionEvent) {
        if self.journalled {
            self.journal(Step::Emitted(ev.journal_form()));
        }
        self.outbox.push(crate::agent_session::effects::Effect::Emit(ev));
    }

    pub(crate) fn out(&self) -> Out {
        Out::new(self.current, self.journalled)
    }

    /// Performed with its events on the host stream, so `start()` returns after them. Idempotent.
    pub fn open(&mut self, now: Now) -> Vec<crate::agent_session::effects::Effect> {
        if std::mem::replace(&mut self.opened, true) {
            return Vec::new();
        }
        self.step.set(now);
        let now = now.at;
        // The AEC warmup is *not* armed here but at the first `speaking` edge; armed now, it would
        // refuse every interruption for the session.
        self.publish_gate();
        // The away deadline runs from start: both are silent until somebody speaks.
        {
            let mut out = self.out();
            let mut spans = Vec::new();
            let wall = self.step.wall_secs_at(now);
            let agent = self.activity.agent.clone();
            self.session.trace.open(wall, &agent, &mut spans);
            spans.into_iter().for_each(|ev| out.machine().emit(ev));
            // `start()`'s `_update_agent_state("listening")`: the host's first state edge.
            self.set_agent_state(now, AgentState::Listening, &mut out);
            let closing = self.session.closing_past_edge();
            out.machine().append(self.session.states.reconcile_away_timer(now, closing));
            if let Some(recorder) = self.session.start_recorder.take() {
                self.attach_recorder(now, recorder, &mut out);
            }
            self.issue_on_enter(now, &mut out);
            let mut spans = Vec::new();
            self.session.trace.opened(self.step.wall_secs_at(now), &mut spans);
            spans.into_iter().for_each(|ev| out.machine().emit(ev));
            self.settle_step(out)
        }
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// What the core decided outside a step (the opening, the loop's end), for the queue owner to
    /// perform.
    pub fn take_outbox(&mut self) -> Vec<crate::agent_session::effects::Effect> {
        std::mem::take(&mut self.outbox)
    }

    /// The frame the core is stepping in: the queue owner performs its effects inside it.
    pub fn frame(&self) -> FrameId {
        self.current
    }

    /// `ivr_detection=True`: the IVR system is the host's to start; the core starts no system.
    pub fn ivr_detection(&self) -> bool {
        self.session.config.effective.turn.ivr_detection()
    }

    /// The session's STT, whose provider identity a recognition input is stamped with.
    pub fn stt(&self) -> Option<lk_agents_abi::resources::ResourceId> {
        self.session.stt.as_ref().map(|stt| stt.id)
    }

    /// Answer an awaited command; `None` is a command nobody awaits.
    pub(crate) fn answer(&mut self, call: Option<RequestId>, answer: Answer) {
        if let Some(call) = call {
            self.outbox.push(crate::agent_session::effects::Effect::Answer(call, answer));
        }
    }

    /// The speech the loop creates now: under the id the host's `say()` / `generate_reply()`
    /// answered with, else a new one.
    pub(crate) fn new_speech(
        &mut self,
        host: Option<SpeechId>,
        allow_interruptions: bool,
    ) -> (SpeechId, SpeechState) {
        let id = host.unwrap_or_else(|| self.session.ids.next_speech_id());
        (id, SpeechState::new(allow_interruptions))
    }

    /// The host's call is answered once its speech exists, in the cycle that announces it.
    pub(crate) fn answer_speech(&mut self, call: Option<SpeechCall>) {
        if let Some(call) = call {
            self.answer(Some(call.id), Answer::Speech);
        }
    }

    /// The loop stopped. What was queued and never delivered is refused (a census is still
    /// journalled), then the session's last event goes out, then every waiter and task the loop
    /// leaves is released: nothing settles after `close`.
    pub fn finish(
        &mut self,
        undelivered: Vec<Input>,
    ) -> Vec<crate::agent_session::effects::Effect> {
        let mut out = self.out();
        for input in undelivered {
            match input {
                // Only a host command carries a promise somebody awaits. What it hands over is
                // released with what the core held.
                Input::Host(Sent { origin: _, command }) => {
                    self.session.providers.extend(command.resources());
                    self.refuse_command(command, &mut out)
                }
                Input::InputCensus(census) => self.on_input_census(census, &mut out),
                Input::StreamOpened { outcome: Ok(sink), .. } => {
                    out.machine().push(crate::agent_session::effects::Effect::DiscardLeg(sink));
                }
                _ => {}
            }
        }
        // Only the refusals are sent: nothing else takes effect once the loop stopped.
        for planned in out.take() {
            use crate::agent_session::effects::{Effect, Planned};
            if let Planned::Effect(
                effect @ (Effect::Answer(..) | Effect::DiscardLeg(_) | Effect::Journal(_)),
            ) = planned
            {
                self.outbox.push(effect);
            }
        }
        self.run_close_at_end();
        let mut out = self.out();
        if let Some((reason, error)) =
            self.session.closing.as_ref().map(|c| (c.reason, c.error.clone()))
        {
            out.machine().emit(SessionEvent::Close { reason, error });
        }
        // The final local accounting boundary (usage.md "Voice sessions").
        self.session.usage.settle();
        let mut effects = self.settle_step(out);
        self.settle_waiters_at_end();
        self.release_all_providers();
        effects.append(&mut self.outbox);
        effects
    }

    /// When the core next wants [`Event::Wake`]: its earliest deadline, or the last step's instant
    /// when a watched task is already due.
    pub fn wake_at(&self) -> Option<Ts> {
        self.timers.next_deadline()
    }

    /// Whether `timer` is armed, this cycle's own arms and cancels counted.
    pub(crate) fn armed(&self, timer: Timer, out: &mut Out) -> bool {
        out.arms(timer).unwrap_or_else(|| self.timers.is_armed(timer))
    }

    /// Cancels `timer` if it is armed: `true` when it was.
    pub(crate) fn disarm(&self, timer: Timer, out: &mut Out) -> bool {
        let armed = self.armed(timer, out);
        if armed {
            out.machine().cancel(timer);
        }
        armed
    }

    /// One step: `input` at `now` (`None`: time reached [`Self::wake_at`]), then everything it
    /// made due in the same instant.
    pub fn handle(
        &mut self,
        input: Option<Input>,
        now: Now,
    ) -> Vec<crate::agent_session::effects::Effect> {
        self.step.set(now);
        let now = now.at;
        let mut out = self.out();
        match input {
            None => {
                if self.timers.next_deadline().is_some_and(|at| at <= now) {
                    self.on_timers(now, &mut out);
                }
            }
            Some(Input::Stt(ev, provider)) => {
                self.input.stt_provider = provider;
                self.on_stt(now, ev, &mut out);
            }
            Some(Input::Vad(ev)) => self.on_vad(now, ev, &mut out),
            Some(Input::Room(ev)) => self.on_rtc(now, ev, &mut out),
            Some(Input::Host(Sent { origin, command })) => {
                out.trace(CoreStep::CommandReceived {
                    origin: origin.clone(),
                    kind: command.kind(),
                });
                self.command_origin = origin;
                self.session.providers.extend(command.resources());
                // Past the close's hard edge no command takes effect: its promise is answered.
                // A `Close` joins; a hook's answer is still read.
                let late = self.session.closing_past_edge()
                    && !matches!(command, Command::Close(_) | Command::HookDone { .. });
                if late {
                    self.refuse_command(command, &mut out);
                    self.pump_schedule(now, &mut out);
                    let mut effects = self.settle_step(out);
                    self.publish_gate();
                    effects.append(&mut self.outbox);
                    return effects;
                }
                self.on_api(now, command, &mut out);
            }
            Some(Input::RoomIo(ev)) => self.on_room_io(now, ev, &mut out),
            Some(Input::Gen(ev)) => self.on_generation(now, ev, &mut out),
            Some(Input::StorageReloaded { read, history }) => {
                self.storage_reloaded(read, history, &mut out)
            }
            Some(Input::StorageSettled(speech)) => self.storage_settled(now, speech, &mut out),
            Some(Input::StorageAbandoned(speech)) => self.storage_abandoned(now, speech, &mut out),
            Some(Input::StorageLost(speech)) => self.storage_lost(speech, &mut out),
            Some(Input::Realtime(ev)) => self.on_realtime(now, ev, &mut out),
            Some(Input::Eot(req, p, thresholds)) => {
                self.activity.audio_recognition.set_thresholds(thresholds);
                self.on_eot(now, req, p, &mut out);
            }
            Some(Input::Overlap(ev)) => self.on_overlap_verdict(now, ev, &mut out),
            Some(Input::ChannelEnded(end)) => self.on_leg_ended(now, end, &mut out),
            Some(Input::SttRetried(error)) => {
                self.report_error(
                    now,
                    crate::events::ErrorSource::Stt,
                    true,
                    error,
                    None,
                    &mut out,
                );
            }
            Some(Input::StreamOpened { open, leg, outcome }) => {
                self.on_leg_opened(now, open, leg, outcome, &mut out);
            }
            Some(Input::Video(frame)) => self.on_video_frame(now, frame, &mut out),
            Some(Input::InputStarted { at }) => self.input.first_input_at = Some(at),
            Some(Input::GateSilenced(silenced)) => {
                for (leg, frames) in silenced {
                    out.trace(PlayoutStep::GateSilenced { leg, frames });
                }
            }
            Some(Input::InputCensus(census)) => self.on_input_census(census, &mut out),
        }
        // Scheduled in the same cycle as the wake that changed it.
        self.pump_schedule(now, &mut out);
        let mut effects = self.settle_step(out);
        // The run's settle check, after the wake's own effects.
        let mut out = self.out();
        self.run_check(&mut out);
        self.settle_waiters(now, &mut out);
        // What the waiters started (a background tool's deferred reply, a filler's say) is scheduled
        // in this cycle too, or it waits, the floor free, for some unrelated wake.
        self.pump_schedule(now, &mut out);
        self.release_providers(&mut out);
        effects.extend(self.settle_step(out));
        self.activity.live_speech_at_final = false;
        self.publish_gate();
        effects.append(&mut self.outbox);
        effects
    }

    /// **Already done**, not dropped: an unattached handle would answer `done() == false` forever.
    pub(crate) fn settle_undelivered(&mut self, what: &str, call: Option<SpeechCall>) {
        let allow_interruptions = self.session.config.effective.turn.interruption.enabled();
        let (id, mut state) = self.new_speech(call.as_ref().map(|c| c.speech), allow_interruptions);
        state.mark_done(Some(lk_agents_abi::errors::ProviderError::Local(
            lk_agents_abi::errors::ApiError::new(format!(
                "the session loop ended before {what}() ran"
            )),
        )));
        self.session.trace.speech_ended(id, state.view());
        self.outbox.extend(unqueued_speech(id, state));
        self.activity.speech_scheduler.retire_later(id);
        // The host stream still exists, so the speech is announced and ended on it like every other.
        let kind = if what == "say" {
            crate::agent_activity::scheduling::SpeechKind::Say
        } else {
            crate::agent_activity::scheduling::SpeechKind::GenerateReply
        };
        let announce = SessionEvent::SpeechCreated { speech: id, kind, user_initiated: true };
        self.emit_now(announce);
        // Born past the close's edge: the scheduling task's refusal and the close's release.
        self.journal(Step::Speech(SpeechStep::ReplyCreated { speech: id }));
        self.journal(Step::Speech(SpeechStep::ScheduleRejected {
            speech: id,
            reason: crate::agent_activity::scheduling::Rejected::Stopped,
        }));
        self.journal(Step::Close(CloseStep::Performed(CloseAct::Released { speech: id })));
        let done =
            SessionEvent::SpeechDone { speech: id, outcome: crate::events::SpeechOutcome::Failed };
        self.emit_now(done);
        self.answer_speech(call);
    }

    pub(super) fn on_rtc(&mut self, now: Ts, ev: RoomEvent, out: &mut Out) {
        out.trace(RoomStep::Event(crate::trace::RoomEventKind::of(&ev)));
        match ev {
            // The room going away is the job ending (`CloseReason.JOB_SHUTDOWN`, no drain).
            RoomEvent::Close { .. } => {
                let reason = lk_agents_abi::voice::events::CloseReason::JobShutdown;
                self.begin_closing(now, Close { reason, drain: false, error: None }, out)
            }
            // The departure arrives as the link's own edge (`ParticipantUnlinked`); the rest is
            // `RoomIo`'s.
            RoomEvent::TrackSubscribed { .. }
            | RoomEvent::TrackUnsubscribed { .. }
            | RoomEvent::TrackPublished { .. }
            | RoomEvent::TrackUnpublished { .. }
            | RoomEvent::LocalTrackPublished { .. }
            | RoomEvent::ParticipantConnected { .. }
            | RoomEvent::ParticipantActive { .. }
            | RoomEvent::ParticipantAttributesChanged { .. }
            | RoomEvent::ParticipantDisconnected { .. }
            | RoomEvent::ConnectionStateChanged { .. }
            | RoomEvent::SipDtmf { .. } => {}
        }
    }

    pub(super) fn on_api(&mut self, now: Ts, cmd: Command, out: &mut Out) {
        match cmd {
            Command::Speech(cmd) => self.on_speech(now, cmd, out),
            Command::Activity(cmd) => self.on_frame(now, cmd, out),
            Command::Turn(cmd) => self.on_turn(now, cmd, out),
            Command::Video(image) => {
                self.push_video(image, out);
            }
            Command::Recorder(crate::agent_session::RecorderCommand::Attach(HostRecorder(
                recorder,
            ))) => self.attach_recorder(now, recorder, out),
            Command::Recorder(crate::agent_session::RecorderCommand::Detach) => {
                self.detach_recorder(now, out)
            }
            Command::Close(close) => self.begin_closing(now, close, out),
            Command::HookDone { id, outcome } => self.on_hook_done(now, id, outcome, out),
            Command::Run(spec) => self.on_run(now, *spec, out),
            Command::VoiceTask(cmd) => self.on_voice_task(now, cmd, out),
            Command::AmdSpan(verdict) => {
                let mut spans = Vec::new();
                let wall = self.step.wall_secs_at(now);
                self.session.trace.amd(wall, verdict.as_ref(), &mut spans);
                spans.into_iter().for_each(|ev| out.machine().emit(ev));
            }
            Command::Dtmf(cmd) => self.dtmf_step(cmd.into(), out),
            Command::HostMetrics(record) => {
                out.machine().emit(crate::events::SessionEvent::MetricsCollected {
                    metrics: record,
                    speech: None,
                });
            }
            Command::SetStaticKeyterms(terms) => {
                self.keyterms_step(
                    now,
                    crate::keyterm_detection::KeytermIn::StaticTerms(terms),
                    out,
                );
            }
            // Addressed to the IVR navigator: journalled and relayed on the feed, acted on by
            // nobody: except `AddTools`, since the core owns what a request is offered.
            Command::Ivr(IvrCommand::AddTools(tools)) => {
                let added = self.session.extend_tools(tools);
                out.trace(crate::trace::ToolStep::SessionToolsExtended { added });
            }
            Command::Ivr(command) => {
                out.machine()
                    .push(crate::agent_session::effects::Effect::Tap(SystemTap::Ivr(command)));
            }
        }
    }

    /// The next timer due by `now` that `pick` accepts, fired. One an earlier fire of this step
    /// cancelled or moved is not due: its cancel effect runs after the step, and asyncio's
    /// `_run_once` likewise skips a handle cancelled earlier in the same tick.
    pub(crate) fn pop_due(
        &mut self,
        now: Ts,
        out: &mut Out,
        pick: impl Fn(Timer) -> bool,
    ) -> Option<Timer> {
        for touched in out.timers_touched() {
            self.timers.cancel(touched);
        }
        let timer = self.timers.pop_due_where(now, pick)?;
        out.trace(CoreStep::TimerFired(timer.into(), now));
        Some(timer)
    }

    /// Every due timer but the scheduler's, which the scheduling pass fires wherever it runs.
    pub(super) fn on_timers(&mut self, now: Ts, out: &mut Out) {
        let pick = |t| !matches!(t, Timer::InterruptionTimeout(_) | Timer::ConsecutiveSpeechDelay);
        while let Some(timer) = self.pop_due(now, out, pick) {
            match timer {
                // A close that moved on ends the pass: the edge cancels the turn's timers, and
                // whatever else is due fires on the next wake, after the phase's effects.
                Timer::Close => {
                    if self.close_tick(now, out) {
                        return;
                    }
                }
                Timer::Commit => self.perform_pending_commit(now, out),
                Timer::Hook(id) => self.hook_timed_out(now, id, out),
                // `PausedSpeech` owns this one.
                Timer::FalseInterruption => self.on_false_interruption_timeout(now, out),
                // `_on_aec_warmup_expired`: the one place `aec_warmup_armed` goes back to `false`
                // before close.
                Timer::AecWarmup => {
                    let closing = self.session.closing_past_edge();
                    self.session.aec.expire(closing);
                }
                Timer::UserAway => {
                    out.machine().append(self.session.states.on_user_away(now));
                }
                Timer::RecorderFlush => self.flush_recorder(now, true, out),
                Timer::KeytermTimeout => self.on_keyterm_timeout(now, out),
                Timer::DtmfDebounce => {
                    self.dtmf_step(crate::workflows::dtmf::DtmfTaskEvent::Debounced, out)
                }
                Timer::EndCallReply => self.end_call_reply_timed_out(now, out),
                // A realtime request the server never answered: the machine settles it.
                Timer::RealtimeRequest(key) => self.realtime_request_timed_out(key, out),
                Timer::Filler(id) => self.on_filler_timer(now, id, out),
                Timer::LaneRelease(speech) => self.lane_release_due(speech, out),
                Timer::EotPredictionTimeout
                | Timer::EndpointingDelay
                | Timer::TranscriptionTimeout
                | Timer::BackchannelBoundary
                | Timer::UserTurnDuration => {
                    let eff = self.activity.audio_recognition.on_timer(now, timer);
                    out.machine().append(eff);
                }
                // Fired by the scheduling pass.
                Timer::InterruptionTimeout(_) | Timer::ConsecutiveSpeechDelay => {}
            }
        }
    }

    /// Sent to the data plane and journalled only when the verdict changed. Through
    /// [`Driver::journal`]: no cycle exists here.
    pub(super) fn publish_gate(&mut self) {
        let session = &self.session;
        let inputs = crate::audio::GateInputs {
            agent_state: session.states.agent_state(),
            aec_warmup_remaining: session.aec.remaining(),
            aec_warmup_armed: session.aec.armed(),
            uninterruptible_speech_active: self
                .current_speech()
                .is_some_and(|(_, h)| !h.done() && !h.interrupted() && !h.allow_interruptions()),
            discard_audio_if_uninterruptible: session
                .config
                .effective
                .turn
                .interruption
                .discard_audio_if_uninterruptible(),
        };
        let discard = crate::audio::audio_gate(&inputs);
        if self.input.gate_discard.replace(discard) != Some(discard) {
            self.outbox.push(crate::agent_session::effects::Effect::Forwarder(
                crate::data_plane::ForwarderCommand::SetGateDiscard(discard),
            ));
            self.journal(Step::Playout(PlayoutStep::GatePublished {
                stt_real: !discard,
                realtime_real: !discard,
                vad_real: true,
                amd_real: true,
            }));
        }
    }

    /// Interruptible speeches only (an uninterruptible greeting plays over the user by design); a
    /// speech whose first frame is out is never held again.
    pub(super) fn launch_hold(&self) -> std::collections::BTreeSet<SpeechId> {
        // A block or supersession can move the floor away from a speech still at launch.
        if !self.activity.audio_recognition.user_speaking() {
            return std::collections::BTreeSet::new();
        }
        self.activity
            .launching
            .iter()
            .copied()
            .filter(|speech| {
                self.activity.speech_scheduler.speech(*speech).is_some_and(|h| h.interruptible())
            })
            .collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    /// **Moved, never stringified.**
    #[error("provider: {0}")]
    Provider(#[from] ProviderError),

    #[error("output admission refused: {0}")]
    OutputBindingRefused(#[from] super::output::OutputBindingError),

    #[error("resample: {0}")]
    Resample(String),

    #[error("playback report timed out for attachment {} segment {}", .0.attachment, .0.segment)]
    PlaybackReportTimeout(lk_agents_abi::voice::io::PlaybackSegment),

    #[error("text output: {0}")]
    TextOutput(lk_agents_abi::voice::io::IoError),

    #[error("the session has no {0}")]
    Missing(&'static str),
}

/// A speech the loop creates and ends in the same step, never queued: what it published, then its
/// release.
pub(crate) fn unqueued_speech(
    id: SpeechId,
    mut state: SpeechState,
) -> impl Iterator<Item = crate::agent_session::effects::Effect> {
    let updates = state.take_updates().into_iter().chain([SpeechUpdate::Released]);
    updates.map(move |update| crate::agent_session::effects::Effect::Speech(id, update))
}
