//! The session decision core and the input vocabulary its runtime queue carries. Initialization
//! and ordered recognition dispatch are implementations of this same driver, colocated here.

use crate::agent_activity::false_interruption::PausedSpeech;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_activity::{AgentActivity, DriverConfig};
use crate::agent_session::VoiceSessionState;
use crate::agent_session::commands::{
    Answer, Close, Command, IvrCommand, RecorderHandle, RequestId, Sent, SpeechCall,
    VoiceCommandOrigin,
};
use crate::agent_session::effects::Timer;
use crate::agent_session::effects::VoiceEffect;
use crate::agent_session::machine::coordination::{PendingInputOperation, VoiceCallback};
use crate::agent_session::time::Now;
use crate::agent_session::{NextId, SessionParts, SessionView};
use crate::audio::AudioChannel;
use crate::audio::ChannelEnd;
use crate::audio_recognition::TurnDetectionMode;
use crate::audio_recognition::effects::AudioRecognitionEffect;
use crate::audio_recognition::observation::RecognitionObservation;
use crate::events::SessionEvent;
use crate::generation::GenEvent;
use crate::generation::{ExpressiveLayers, Instructions};
use crate::speech_handle::{SpeechState, SpeechUpdate};
use crate::trace::RealtimeStep;
use crate::trace::RoomEventKind;
use crate::trace::{CloseAct, CloseStep, CoreStep, PlayoutStep, RoomStep, SpeechStep, Step};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::resources::{RealtimeRef, ResourceId};
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::{SpeechEvent, TimerSet, Ts, VadEvent};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

/// Only what a host could read, in loop order; never machine state.
#[derive(Debug, Clone)]
pub enum SystemTap {
    Session(SessionEvent),
    Vad(lk_agents_abi::VadEvent),
    Stt(lk_agents_abi::SpeechEvent),
    Ivr(IvrCommand),
    CloseRequested,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueClosed;

/// Timers are the loop's own.
#[derive(lk_agents_abi::Vocabulary)]
pub enum VoiceEvent {
    Host(Sent),
    Stt(crate::audio::Tagged<SpeechEvent>, Option<lk_agents_abi::ProviderId>),
    Vad(VadEvent),
    Room(RoomEventKind),
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
    /// The process's loop monitor: the host's event loop stalled this long. Recorded on the
    /// session's span only; it decides nothing.
    EventLoopBlocked(std::time::Duration),
}

/// The session's decision core: handed every [`VoiceEvent`] in arrival order by the
/// runtime queue owner, it merges them with its timers and orchestrates the
/// components in sequence: the `AgentSession`, its running
/// [`AgentActivity`](crate::agent_activity::AgentActivity), and the [`AgentInput`](crate::agent_activity::io::AgentInput) legs. Every
/// decision is journalled, dated on the session clock.
pub struct VoiceDriver {
    pub(crate) session: VoiceSessionState,
    pub(crate) activity: crate::agent_activity::AgentActivity,
    pub(crate) input: crate::agent_activity::io::AgentInput,
    pub(crate) timers: TimerSet<Timer>,
    pub(crate) journalled: bool,
    /// Decides ownership (a tool's hold is its run's).
    pub(crate) command_origin: VoiceCommandOrigin,
    /// The time of the step being handled, or of the last one.
    pub(crate) step: crate::agent_session::time::StepClock,
    pub(crate) stopped: bool,
    pub(in crate::agent_session) opened: bool,
    /// The activity changed this cycle: published once, as the cycle's effects are done.
    pub(crate) activity_changed: std::cell::Cell<bool>,
    /// What the core decided outside an input transition (opening, the loop's end, published views),
    /// returned with the step's own effects.
    pub(super) outbox: Vec<crate::agent_session::effects::VoiceEffect>,
    pub(super) callbacks: std::collections::VecDeque<VoiceCallback>,
    pub(super) pending_input: Vec<(usize, PendingInputOperation)>,
    /// Revised deadlines become fireable at the next atomic boundary, not recursively
    /// inside the transition that set them (including zero-duration deadlines).
    pub(super) revised_deadlines: std::collections::BTreeSet<Timer>,
    /// The [`SessionView`](crate::agent_session::SessionView) last published, so a step that changed
    /// none of it publishes nothing.
    pub(crate) view: Arc<crate::agent_session::SessionView>,
    /// The session's span state: what the entities owning spans open and end them through.
    pub(crate) telemetry: crate::telemetry::Trace,
    /// A runtime exports the records: only a unit test's core may be dropped before its close.
    pub(crate) exported: bool,
}

impl VoiceDriver {
    /// Only while performing an effect, when no cycle exists.
    pub(crate) fn journal(&mut self, step: Step) {
        if !self.journalled {
            return;
        }
        self.outbox.push(crate::agent_session::effects::VoiceEffect::Journal(Box::new(step)));
    }

    /// An event sent to the host where no cycle exists, journalled first.
    pub(crate) fn emit_now(&mut self, ev: SessionEvent) {
        if self.journalled {
            self.journal(Step::Emitted(ev.journal_form()));
        }
        self.outbox.push(crate::agent_session::effects::VoiceEffect::Emit(ev));
    }

    /// The spans ended since the last flush, for the exporter. An unclaimed agent turn handover
    /// lapses with them.
    pub(crate) fn flush_spans(&mut self) {
        self.activity.lapse_agent_turn_handover();
        let records = self.telemetry.drain();
        if !records.is_empty() {
            let ev = SessionEvent::Telemetry(Arc::new(records));
            self.outbox.push(crate::agent_session::effects::VoiceEffect::Emit(ev));
        }
    }

    /// A core no runtime runs (a unit test): dropping it before its close loses nothing.
    #[cfg(test)]
    pub(crate) fn unexported(mut self) -> Self {
        self.exported = false;
        self
    }

    /// Performed with its events on the host stream, so `start()` returns after them. Idempotent.
    pub fn open(&mut self, now: Now) -> Vec<crate::agent_session::effects::VoiceEffect> {
        if std::mem::replace(&mut self.opened, true) {
            return Vec::new();
        }
        self.step.set(now);
        self.telemetry.step(now.at);
        let now = now.at;
        // The AEC warmup is *not* armed here but at the first `speaking` edge; armed now, it would
        // refuse every interruption for the session.
        self.publish_gate();
        // The away deadline runs from start: both are silent until somebody speaks.
        {
            // `start()`'s `_update_agent_state("listening")`: the host's first state edge.
            self.set_agent_state(now, AgentState::Listening);
            let closing = self.session.closing_past_edge();
            {
                let updates = self.session.states.reconcile_away_timer(now, closing);
                self.apply_user_state_updates(updates);
            };
            if let Some(recorder) = self.session.recorder.take_initial() {
                self.attach_recorder(now, recorder);
            }
            self.issue_on_enter(now);
            self.session_opened();
            self.settle_step()
        }
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// What the core decided outside a step (the opening, the loop's end), for the queue owner to
    /// perform.
    pub fn take_outbox(&mut self) -> Vec<crate::agent_session::effects::VoiceEffect> {
        self.flush_spans();
        std::mem::take(&mut self.outbox)
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
            self.outbox.push(crate::agent_session::effects::VoiceEffect::Answer(call, answer));
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
        undelivered: Vec<VoiceEvent>,
    ) -> Vec<crate::agent_session::effects::VoiceEffect> {
        self.stopped = true;
        let previously_queued = self.outbox.len();
        for input in undelivered {
            match input {
                // Only a host command carries a promise somebody awaits. What it hands over is
                // released with what the core held.
                VoiceEvent::Host(Sent { origin: _, command }) => {
                    self.session.providers.extend(command.resources());
                    self.refuse_command(command)
                }
                VoiceEvent::InputCensus(census) => self.on_input_census(census),
                VoiceEvent::StreamOpened { outcome: Ok(sink), .. } => {
                    self.record_effect(
                        crate::agent_session::effects::VoiceEffect::DiscardOpenedInput(sink),
                    );
                }
                _ => {}
            }
        }
        // Terminal refusals cannot restart domain work. Preserve earlier undrained work;
        // only this terminal admission pass is restricted to its refusal/release output.
        let refusals: Vec<_> = self
            .outbox
            .drain(previously_queued..)
            .filter(|effect| {
                matches!(
                    effect,
                    VoiceEffect::Answer(..)
                        | VoiceEffect::DiscardOpenedInput(_)
                        | VoiceEffect::Journal(_)
                )
            })
            .collect();
        self.outbox.extend(refusals);
        self.callbacks.clear();
        self.pending_input.clear();
        self.run_close_at_end();
        if let Some((reason, error)) =
            self.session.closing.as_ref().map(|c| (c.reason, c.error.clone()))
        {
            self.publish_event(SessionEvent::Close { reason, error });
        }
        // The final local accounting boundary (usage.md "Voice sessions"): a recognition stream
        // still open ends with the session.
        for stream in self.session.stt_requests.keys().copied().collect::<Vec<_>>() {
            self.stt_stream_ended(stream);
        }
        self.session.usage.settle();
        self.settle_internal();
        self.settle_waiters_at_end();
        self.release_all_providers();
        self.take_outbox()
    }

    /// When the core next wants the runtime wake event: its earliest deadline, or the last step's instant
    /// when a watched task is already due.
    pub fn wake_at(&self) -> Option<Ts> {
        self.timers.next_deadline().into_iter().chain(self.reply_wake_at()).min()
    }

    /// Cancels `timer` if it is armed: `true` when it was.
    pub(crate) fn clear_deadline_if_set(&mut self, timer: Timer) -> bool {
        let armed = self.deadline(timer).is_some();
        if armed {
            self.clear_deadline(timer);
        }
        armed
    }

    /// One step: `input` at `now` (`None`: time reached [`Self::wake_at`]), then everything it
    /// made due in the same instant.
    pub fn handle(
        &mut self,
        input: Option<VoiceEvent>,
        now: Now,
    ) -> Vec<crate::agent_session::effects::VoiceEffect> {
        self.step.set(now);
        self.telemetry.step(now.at);
        let now = now.at;
        // A shared invocation may be settling cancelled host work even when the session
        // has no domain timer. Its cleanup bound is part of this driver's wake contract.
        self.wake_reply_runs(now);
        match input {
            None => {
                if self.timers.next_deadline().is_some_and(|at| at <= now) {
                    self.on_timers(now);
                }
            }
            Some(VoiceEvent::Stt(ev, provider)) => {
                self.input.stt_provider = provider;
                self.on_stt(now, ev);
            }
            Some(VoiceEvent::Vad(ev)) => self.on_vad(now, ev),
            Some(VoiceEvent::Room(ev)) => self.on_rtc(now, ev),
            Some(VoiceEvent::Host(Sent { origin, command })) => {
                self.record_trace(CoreStep::CommandReceived {
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
                    self.refuse_command(command);

                    self.settle_internal();
                    self.publish_gate();
                    return self.take_outbox();
                }
                self.on_api(now, command);
            }
            Some(VoiceEvent::RoomIo(ev)) => self.on_room_io(now, ev),
            Some(VoiceEvent::Gen(ev)) => self.on_generation(now, ev),
            Some(VoiceEvent::StorageReloaded { read, history }) => {
                self.storage_reloaded(read, history)
            }
            Some(VoiceEvent::StorageSettled(speech)) => self.storage_settled(now, speech),
            Some(VoiceEvent::StorageAbandoned(speech)) => self.storage_abandoned(now, speech),
            Some(VoiceEvent::StorageLost(speech)) => self.storage_lost(speech),
            Some(VoiceEvent::Realtime(ev)) => self.on_realtime(now, ev),
            Some(VoiceEvent::Eot(req, p, thresholds)) => {
                let (observations, effects) = self.activity.on_eot(now, req, p, thresholds);
                for observation in observations {
                    self.record_trace(observation);
                }
                self.apply_recognition_effects(effects);
            }
            Some(VoiceEvent::Overlap(ev)) => self.on_overlap_verdict(now, ev),
            Some(VoiceEvent::ChannelEnded(end)) => self.on_leg_ended(now, end),
            Some(VoiceEvent::SttRetried(error)) => {
                self.report_error(now, crate::events::ErrorSource::Stt, true, error, None);
            }
            Some(VoiceEvent::StreamOpened { open, leg, outcome }) => {
                self.on_leg_opened(now, open, leg, outcome);
            }
            Some(VoiceEvent::Video(frame)) => self.on_video_frame(now, frame),
            Some(VoiceEvent::InputStarted { at }) => self.input.first_input_at = Some(at),
            Some(VoiceEvent::GateSilenced(silenced)) => {
                for (leg, frames) in silenced {
                    self.record_trace(PlayoutStep::GateSilenced { leg, frames });
                }
            }
            Some(VoiceEvent::InputCensus(census)) => self.on_input_census(census),
            Some(VoiceEvent::EventLoopBlocked(duration)) => self.event_loop_blocked(duration),
        }
        // Scheduled in the same cycle as the wake that changed it.

        self.settle_internal();
        // The run's settle check, after the wake's own effects.
        self.run_check();
        self.settle_waiters(now);
        // What the waiters started (a background tool's deferred reply, a filler's say) is scheduled
        // in this cycle too, or it waits, the floor free, for some unrelated wake.

        self.release_providers();
        self.settle_internal();
        self.activity.finish_step();
        self.publish_gate();
        self.take_outbox()
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
        self.outbox.extend(unqueued_speech(id, state));
        self.activity.retire_speech_later(id);
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

    pub(super) fn on_rtc(&mut self, now: Ts, event: RoomEventKind) {
        self.record_trace(RoomStep::Event(event));
        if event == RoomEventKind::Close {
            let reason = lk_agents_abi::voice::events::CloseReason::JobShutdown;
            self.begin_closing(now, Close { reason, drain: false, error: None });
        }
    }

    pub(super) fn on_api(&mut self, now: Ts, cmd: Command) {
        match cmd {
            Command::Speech(cmd) => self.on_speech(now, cmd),
            Command::Activity(cmd) => self.on_frame(now, cmd),
            Command::Turn(cmd) => self.on_turn(now, cmd),
            Command::Video(image) => {
                self.push_video(image);
            }
            Command::Recorder(crate::agent_session::RecorderCommand::Attach(RecorderHandle(
                recorder,
            ))) => self.attach_recorder(now, recorder),
            Command::Recorder(crate::agent_session::RecorderCommand::Detach) => {
                self.detach_recorder(now)
            }
            Command::Close(close) => self.begin_closing(now, close),
            Command::HookDone { id, outcome } => self.on_hook_done(now, id, outcome),
            Command::Run(spec) => self.on_run(now, *spec),
            Command::VoiceTask(cmd) => self.on_voice_task(now, cmd),
            Command::AmdSpan(ev) => self.amd_span(&ev),
            Command::Dtmf(cmd) => self.dtmf_step(cmd.into()),
            Command::HostMetrics(record) => {
                self.publish_event(crate::events::SessionEvent::MetricsCollected {
                    metrics: record,
                    speech: None,
                    counted: false,
                });
            }
            Command::SetStaticKeyterms(terms) => {
                self.keyterms_step(now, crate::keyterm_detection::KeytermIn::StaticTerms(terms));
            }
            // Addressed to the IVR navigator: journalled and relayed on the feed, acted on by
            // nobody: except `AddTools`, since the core owns what a request is offered.
            Command::Ivr(IvrCommand::AddTools(tools)) => {
                let added = self.session.extend_tools(tools);
                self.record_trace(crate::trace::ToolStep::SessionToolsExtended { added });
            }
            Command::Ivr(command) => {
                self.record_effect(crate::agent_session::effects::VoiceEffect::Tap(
                    SystemTap::Ivr(command),
                ));
            }
        }
    }

    /// The next timer due by `now` that `pick` accepts, fired. One an earlier fire of this step
    /// cancelled or moved is not due: its cancel effect runs after the step, and asyncio's
    /// `_run_once` likewise skips a handle cancelled earlier in the same tick.
    pub(crate) fn pop_due(&mut self, now: Ts, pick: impl Fn(Timer) -> bool) -> Option<Timer> {
        let revised = &self.revised_deadlines;
        let timer =
            self.timers.pop_due_where(now, |timer| !revised.contains(&timer) && pick(timer))?;
        self.record_trace(CoreStep::TimerFired(timer.into(), now));
        Some(timer)
    }

    /// Every due timer but the scheduler's, which the scheduling pass fires wherever it runs.
    pub(super) fn on_timers(&mut self, now: Ts) {
        let pick = |t| !matches!(t, Timer::InterruptionTimeout(_) | Timer::ConsecutiveSpeechDelay);
        while let Some(timer) = self.pop_due(now, pick) {
            match timer {
                // A close that moved on ends the pass: the edge cancels the turn's timers, and
                // whatever else is due fires on the next wake, after the phase's effects.
                Timer::Close => {
                    if self.close_tick(now) {
                        return;
                    }
                }
                Timer::Commit => self.perform_pending_commit(now),
                Timer::Hook(id) => self.hook_timed_out(now, id),
                // `PausedSpeech` owns this one.
                Timer::FalseInterruption => self.on_false_interruption_timeout(now),
                // `_on_aec_warmup_expired`: the one place `aec_warmup_armed` goes back to `false`
                // before close.
                Timer::AecWarmup => {
                    let closing = self.session.closing_past_edge();
                    self.session.aec.expire(closing);
                }
                Timer::UserAway => {
                    {
                        let updates = self.session.states.on_user_away(now);
                        self.apply_user_state_updates(updates);
                    };
                }
                Timer::RecorderFlush => self.flush_recorder(now, true),
                Timer::KeytermTimeout => self.on_keyterm_timeout(now),
                Timer::DtmfDebounce => {
                    self.dtmf_step(crate::workflows::dtmf::DtmfTaskEvent::Debounced)
                }
                Timer::EndCallReply => self.end_call_reply_timed_out(now),
                // A realtime request the server never answered: the machine settles it.
                Timer::RealtimeRequest(key) => self.realtime_request_timed_out(key),
                Timer::Filler(id) => self.on_filler_timer(now, id),
                Timer::LaneRelease(speech) => self.lane_release_due(speech),
                Timer::EotPredictionTimeout
                | Timer::EndpointingDelay
                | Timer::TranscriptionTimeout
                | Timer::BackchannelBoundary
                | Timer::UserTurnDuration => {
                    let effects = self.activity.recognition_timer(now, timer);
                    self.apply_recognition_effects(effects);
                }
                // Fired by the scheduling pass.
                Timer::InterruptionTimeout(_) | Timer::ConsecutiveSpeechDelay => {}
            }
        }
    }

    /// Sent to the data plane and journalled only when the verdict changed. Through
    /// [`VoiceDriver::journal`]: no cycle exists here.
    pub(super) fn publish_gate(&mut self) {
        let session = &self.session;
        let inputs = crate::audio::GateInputs {
            agent_state: session.states.agent_state(),
            aec_warmup_remaining: session.aec.remaining(),
            aec_warmup_armed: session.aec.armed(),
            uninterruptible_speech_active: self
                .activity
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
            self.outbox.push(crate::agent_session::effects::VoiceEffect::UpdateAudioInputRouting(
                crate::data_plane::AudioInputRouterCommand::SetGateDiscard(discard),
            ));
            self.journal(Step::Playout(PlayoutStep::GatePublished {
                stt_real: !discard,
                realtime_real: !discard,
                vad_real: true,
                amd_real: true,
            }));
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    #[error("the session closed before startup was admitted")]
    StartupClosed,

    /// **Moved, never stringified.**
    #[error("provider: {0}")]
    Provider(#[from] ProviderError),

    #[error("output admission refused: {0}")]
    OutputBindingRefused(#[from] crate::agent_session::output::OutputBindingError),

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
) -> impl Iterator<Item = crate::agent_session::effects::VoiceEffect> {
    let updates = state.take_updates().into_iter().chain([SpeechUpdate::Released]);
    updates.map(move |update| crate::agent_session::effects::VoiceEffect::Speech(id, update))
}

impl VoiceDriver {
    pub(crate) fn attach_recorder(
        &mut self,
        now: Ts,
        recorder: lk_agents_abi::resources::ResourceId,
    ) {
        let output = self.session.outputs.audio.as_ref().map(|output| output.id);
        self.session.recorder.attach(recorder);
        self.record_effect(VoiceEffect::AttachRecorder {
            recorder: crate::agent_session::commands::RecorderHandle(recorder),
            output,
        });
        self.set_deadline(
            Timer::RecorderFlush,
            now.saturating_add(crate::agent_session::RECORDER_FLUSH),
        );
    }
    pub(crate) fn detach_recorder(&mut self, now: Ts) {
        let output = self.session.outputs.audio.as_ref().map(|output| output.id);
        if self.session.recorder.detach() {
            self.record_effect(VoiceEffect::FlushRecorder(self.step.wall_secs_at(now)));
            self.record_effect(VoiceEffect::DetachRecorder(output));
            self.clear_deadline(Timer::RecorderFlush);
            self.record_trace(PlayoutStep::RecorderDetached);
        }
    }
    pub(crate) fn flush_recorder(&mut self, now: Ts, rearm: bool) {
        let rearm = rearm && !self.session.closing_past_edge();
        if self.session.recorder.is_attached() {
            self.record_effect(VoiceEffect::FlushRecorder(self.step.wall_secs_at(now)));
            if rearm {
                self.set_deadline(
                    Timer::RecorderFlush,
                    now.saturating_add(crate::agent_session::RECORDER_FLUSH),
                );
            }
        }
    }
}

#[cfg(test)]
impl VoiceDriver {
    /// Observe deferred ordering without exposing mutable coordinator storage to components.
    pub(crate) fn pending_callbacks(
        &self,
    ) -> &std::collections::VecDeque<super::coordination::VoiceCallback> {
        &self.callbacks
    }
}

/// What the core is built from: its options, the providers by id, the opening agent and the
/// sinks the microphone feeds.
pub struct InitialSession {
    pub config: DriverConfig,
    pub parts: SessionParts,
    pub instructions: Instructions,
    pub dynamic_instructions: Option<lk_agents_abi::resources::ResourceId>,
    pub output_nodes: crate::agent_session::OutputNodes,
    pub toolset: crate::tool_executor::Toolset,
    pub hook_bound: Option<Duration>,
    pub expressive: ExpressiveLayers,
    pub sinks: BTreeMap<AudioChannel, ResourceId>,
    pub video_instance: Option<lk_agents_abi::resources::VideoInputRef>,
    pub video_sampler: Option<crate::agent_activity::video_sampler::VideoSampler>,
    pub eot_thresholds: Option<lk_agents_abi::tokenize::language::ThresholdOptions>,
    pub eot_provider: Option<lk_agents_abi::ProviderId>,
    pub rt_session: Option<RealtimeRef>,
    pub rt_caps: lk_agents_abi::llm::realtime::RealtimeCapabilities,
    pub rt_disabled: bool,
    pub rt_resolved: TurnDetectionMode,
    pub view: Arc<SessionView>,
    pub next_id: NextId,
    pub now: Now,
    pub journalled: bool,
    pub forward_chat_context: bool,
    pub keyterms: Option<(crate::keyterm_detection::KeytermSetup, Option<ResourceId>)>,
}

impl VoiceDriver {
    pub fn new(mut initial: InitialSession) -> Self {
        let (sink, audio_enabled) = initial.parts.outputs.pause_facts();
        let paused_speech = PausedSpeech::new(&initial.config.effective, sink, audio_enabled);
        let realtime_disabled = initial.rt_session.as_ref().map(|_| initial.rt_disabled);
        let activity = AgentActivity::from_initial(&mut initial, paused_speech);
        let mut telemetry = std::mem::take(&mut initial.parts.telemetry);
        let trace = crate::agent_session::spans::trace(
            &initial.parts.session_id,
            &mut telemetry,
            initial.now,
        );
        let mut session =
            VoiceSessionState::new(initial.config.clone(), initial.parts, initial.next_id);
        session.expressive = initial.expressive;
        let input = crate::agent_activity::io::AgentInput::new(
            initial.sinks,
            initial.video_instance,
            initial.video_sampler,
        );
        let mut outbox = Vec::new();
        if initial.journalled {
            outbox.push(crate::agent_session::effects::VoiceEffect::Journal(Box::new(Step::Core(
                CoreStep::ConfigApplied {
                    layer: crate::agent_activity::options::ConfigLayer::Session,
                    effective: Box::new(initial.config.effective),
                },
            ))));
            let since_zero = initial.now.at.saturating_duration_since(Ts::ZERO);
            outbox.push(crate::agent_session::effects::VoiceEffect::Journal(Box::new(Step::Core(
                CoreStep::WallEpoch {
                    at: initial.now.wall.checked_sub(since_zero).unwrap_or(std::time::UNIX_EPOCH),
                },
            ))));
            if let Some(disabled) = realtime_disabled {
                outbox.push(crate::agent_session::effects::VoiceEffect::Journal(Box::new(
                    Step::Realtime(RealtimeStep::Opened { turn_detection_disabled: disabled }),
                )));
            }
        }
        let mut driver = VoiceDriver {
            activity_changed: Default::default(),
            outbox,
            callbacks: Default::default(),
            pending_input: Vec::new(),
            revised_deadlines: Default::default(),
            view: initial.view,
            session,
            activity,
            input,
            timers: TimerSet::new(),
            journalled: initial.journalled,
            command_origin: VoiceCommandOrigin::Host,
            step: crate::agent_session::time::StepClock::new(),
            stopped: false,
            opened: false,
            telemetry: trace,
            exported: true,
        };
        if let Some(stream) = driver.input.live_stt {
            driver.stt_stream_opened(stream);
        }
        driver.session.providers = driver.providers_held();
        driver.step.set(initial.now);
        let agent = driver.activity.agent_name().to_owned();
        driver.open_session_spans(telemetry, &agent);
        driver.publish_activity();
        driver.flush_activity();
        driver.open_session(initial.forward_chat_context, initial.keyterms);
        driver
    }
}

impl VoiceDriver {
    /// The opening cycles: the first facts, in order, the gate, then the opening agent's records
    /// (the handoff from nobody, the starting config), and the opening agent on the realtime
    /// socket; then the static keyterms.
    fn open_session(
        &mut self,
        forward_chat_context: bool,
        keyterms: Option<(
            crate::keyterm_detection::KeytermSetup,
            Option<lk_agents_abi::resources::ResourceId>,
        )>,
    ) {
        self.publish_gate();
        {
            self.record_handoff(None);
            if self.activity.realtime().is_some() {
                let opening = self.session.chat_ctx.for_realtime();
                self.push_activity(Some(opening));
            }
            self.settle_internal();
        }
        self.session.forward_chat_context = forward_chat_context;
        // `KeytermDetector.start`: the static terms reach the recogniser now.
        if let Some((setup, llm)) = keyterms {
            self.keyterms_start(setup, llm);
            self.settle_internal();
        }
    }
}

impl VoiceDriver {
    pub(crate) fn apply_recognition_effects(&mut self, effects: Vec<AudioRecognitionEffect>) {
        for effect in effects {
            match effect {
                AudioRecognitionEffect::SetDeadline(timer, at) => self.set_deadline(timer, at),
                AudioRecognitionEffect::ClearDeadline(timer) => self.clear_deadline(timer),
                AudioRecognitionEffect::FlushInput(input) => self.flush_input(input),
                AudioRecognitionEffect::PublishSessionEvent(event) => self.publish_event(event),
                AudioRecognitionEffect::RequestEot(req, timeout) => {
                    self.record_effect(VoiceEffect::EotPredict { req, timeout });
                }
                AudioRecognitionEffect::CancelEot(req, timed_out) => {
                    self.record_effect(VoiceEffect::EotCancel { req, timed_out });
                }
                AudioRecognitionEffect::NotifyInterruption(signal) => {
                    self.notify_interruption(signal)
                }
                AudioRecognitionEffect::FlushTurnDetector(reason) => {
                    self.flush_turn_detector(reason)
                }
                AudioRecognitionEffect::EndOfTurn(info) => {
                    self.record_trace(Step::Decided(RecognitionObservation::EndOfTurn(
                        info.clone(),
                    )));
                    self.defer_callback(VoiceCallback::EndOfTurn(info));
                }
                AudioRecognitionEffect::TurnSettled(settled) => {
                    self.record_trace(Step::Decided(RecognitionObservation::TurnSettled(settled)));
                    self.defer_callback(VoiceCallback::TurnSettled(settled));
                }
                AudioRecognitionEffect::AudioInterruptGuard(guard) => {
                    self.record_trace(Step::Decided(RecognitionObservation::AudioInterruptGuard(
                        guard,
                    )));
                    self.defer_callback(VoiceCallback::AudioInterruptGuard(guard));
                }
                AudioRecognitionEffect::InterruptByAudioActivity(trigger) => {
                    self.record_trace(Step::Decided(
                        RecognitionObservation::InterruptByAudioActivity(trigger),
                    ));
                    self.defer_callback(VoiceCallback::InterruptByAudioActivity(trigger));
                }
                AudioRecognitionEffect::EndpointingDelayUpdated { min_delay } => {
                    self.record_trace(Step::Decided(
                        RecognitionObservation::EndpointingDelayUpdated { min_delay },
                    ));
                }
                AudioRecognitionEffect::EouDetected(detected) => {
                    self.defer_callback(VoiceCallback::EouDetected(detected));
                }
                AudioRecognitionEffect::TurnDeclared { owner } => {
                    self.record_trace(Step::Decided(RecognitionObservation::TurnDeclared {
                        owner,
                    }));
                }
            }
        }
    }
}

impl VoiceDriver {
    pub(crate) fn apply_user_state_updates(
        &mut self,
        updates: Vec<crate::agent_session::agent_user_state::AgentUserStateUpdate>,
    ) {
        use crate::agent_session::agent_user_state::AgentUserStateUpdate;
        use crate::agent_session::effects::Timer;
        for update in updates {
            match update {
                AgentUserStateUpdate::PublishSessionEvent(event) => {
                    if let SessionEvent::UserStateChanged { old, new, at } = &event {
                        self.user_state_span_event(*old, *new, *at);
                    }
                    self.publish_event(event)
                }
                AgentUserStateUpdate::SetAwayDeadline(at) => self.set_deadline(Timer::UserAway, at),
                AgentUserStateUpdate::ClearAwayDeadline => self.clear_deadline(Timer::UserAway),
            }
        }
    }
}
