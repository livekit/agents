//! Speech from request to release: `say`, `generate_reply` and the turn's reply through
//! `start_speech`, the generation's reports, the agent state, and `wait_for_idle`.

use crate::agent::{Source, Visibility};
use crate::agent_activity::false_interruption::{
    InterruptOutcome, InterruptRefusal, InterruptWaiter,
};
use crate::agent_activity::scheduling::{PreemptiveDiscard, SpeechId, SpeechKind, Status};
use crate::agent_session::effects::Out;
use crate::agent_session::effects::{GateCause, GateOp};
use crate::agent_session::host_events::HookCall;
use crate::agent_session::{
    Answer, ClosePhase, Driver, IdleRefused, IdleWaiter, Origin, SayText, SpeechCommand,
};
use crate::audio_recognition::{EndOfTurnInfo, TurnEdits};
use crate::events::SessionEvent;
use crate::generation::pipeline_reply::pipeline_reply_task;
use crate::generation::{
    EndOfSpeech, Instructions, TurnInstructions, TurnOutcome as GenTurnOutcome, end_of_speech,
};
use crate::generation::{GenEvent, Providers, ReplyContext, ReplySource};
use crate::speech_handle::{InputDetails, SpeechHandle};
use crate::tool_executor::Handling;
use crate::tool_executor::Toolset;
use crate::trace::{
    CoreStep, GenerationStep, HandoffStep, InterruptCause, InterruptStep, PlayoutStep, SpeechStep,
    ToolStep, TurnStep,
};
use lk_agents_abi::ChatItem;
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::{ChatContext, ChatMessage, Modality, Role, SpawnerExt, Ts};
use std::sync::Arc;

/// A `say(stream)` / `say(audio=)` refused on a realtime model (the wire carries one text item).
pub(crate) const SAY_SHAPE_REFUSED_ON_REALTIME: &str = "say(stream) or say(audio=) on a realtime model: the wire takes one text item and owns its own synthesis";

/// A text `say()` with no TTS while audio output is attached and enabled, refused before a handle
/// exists (`AgentActivity.say`'s first `RuntimeError`).
pub(crate) const SAY_WITHOUT_TTS_REFUSED: &str = "trying to generate speech from text without a TTS model or a RealtimeSession that supports say(); add a TTS model to AgentSession to enable say()";

pub(crate) struct SpeechSpec {
    pub(crate) new_message: Option<ChatMessage>,
    pub(crate) turn_instructions: Option<TurnInstructions>,
    pub(crate) source: ReplySource,
    /// Already resolved against the session default: the loop does not re-resolve options.
    pub(crate) allow_interruptions: bool,
    pub(crate) add_to_chat_ctx: bool,
    /// `generate_reply(tool_choice=)`. `None` is `NOT_GIVEN`, which is **not** `"none"`.
    pub(crate) tool_choice: Option<lk_agents_abi::ToolChoice>,
    pub(crate) chat_ctx: Option<ChatContext>,
    /// `None` is `generate_reply(user_input=)` (`Source::UserInput`).
    pub(crate) turn: Option<crate::audio_recognition::TurnId>,
    /// A hook's `UpdateInstructions`: the prompt for this one request, already rendered.
    pub(crate) instructions_override: Option<String>,
    pub(crate) toolset: Option<Arc<Toolset>>,
    /// Its methods alone decide scheduling, priority and draining-queue entry.
    pub(crate) kind: SpeechKind,
    /// Issued from inside the agent's `on_enter`, so `IGNORE_ON_ENTER` tools are withheld. Decided
    /// by the host (an ambient context var the core may not read).
    pub(crate) on_enter: bool,
    /// The host asked by name (`say`/`generate_reply`) although the kind says otherwise: a
    /// realtime generation for a host request is `user_initiated` to the host.
    pub(crate) host_asked: bool,
}

impl SpeechSpec {
    /// Added to the context, planned against the record and the frame's prompt and tools.
    pub(crate) fn new(kind: SpeechKind, source: ReplySource, allow_interruptions: bool) -> Self {
        SpeechSpec {
            new_message: None,
            turn_instructions: None,
            source,
            allow_interruptions,
            add_to_chat_ctx: true,
            tool_choice: None,
            chat_ctx: None,
            turn: None,
            instructions_override: None,
            toolset: None,
            kind,
            on_enter: false,
            host_asked: false,
        }
    }
}

impl Driver {
    pub(crate) fn on_speech(&mut self, now: Ts, cmd: SpeechCommand, out: &mut Out) {
        match cmd {
            // A realtime session answers on its own socket. These arms must come *before* the
            // pipeline arms: a guarded arm after its unguarded twin is dead code.
            SpeechCommand::GenerateReply {
                user_input,
                instructions,
                allow_interruptions,
                tool_choice,
                chat_ctx,
                toolset,
                call,
                on_enter: _,
            } if self.activity.rt_session.is_some() => {
                // A realtime session's context and tools live in its session config, not per
                // response: refused by name.
                if chat_ctx.is_some() || toolset.is_some() {
                    out.trace(CoreStep::PortFailure(
                        "generate_reply(chat_ctx=) / generate_reply(tools=) on a realtime model: the \
                         session owns both",
                    ));
                    self.fail_waiter(
                        call,
                        "generate_reply(chat_ctx=, tools=) is not supported on a realtime model",
                    );
                    return;
                }
                self.realtime_reply(
                    now,
                    user_input,
                    instructions,
                    allow_interruptions,
                    tool_choice,
                    call,
                    out,
                );
            }
            SpeechCommand::Say { text, allow_interruptions, add_to_chat_ctx, call }
                if self.activity.rt_session.is_some() =>
            {
                // The wire carries a string; a streamed `say` is refused by name, as in the
                // reference.
                let SayText::Text(text) = text else {
                    out.trace(CoreStep::PortFailure(SAY_SHAPE_REFUSED_ON_REALTIME));
                    self.fail_waiter(
                        call,
                        "say(stream) / say(audio=) is not supported on a realtime model",
                    );
                    return;
                };
                let request =
                    crate::agent_activity::realtime::Request::Say { text, add_to_chat_ctx };
                self.realtime_request(now, request, call, allow_interruptions, out);
            }
            SpeechCommand::CommitUserTurn {
                transcript_timeout,
                stt_flush_duration,
                skip_reply,
                call,
            } => {
                self.commit_user_turn(
                    now,
                    transcript_timeout,
                    stt_flush_duration,
                    skip_reply,
                    call,
                    out,
                );
            }
            SpeechCommand::WaitForIdle { hold, call } => {
                let speech = self.tool_speech(&self.command_origin);
                self.session.idle.ask(IdleWaiter { hold, speech, grant: None, call });
            }
            SpeechCommand::WaitForIdleAndHold { call } => {
                // Minted at the ask, so the run a tool's hold belongs to can own it before idle.
                let id = self.session.idle.mint_hold();
                let origin = self.command_origin.clone();
                self.run_own_hold(&origin, id, out);
                let speech = self.tool_speech(&self.command_origin);
                self.session.idle.ask(IdleWaiter { hold: None, speech, grant: Some(id), call });
            }
            SpeechCommand::ReleaseIdleHold { hold } => {
                if self.session.idle.release(hold) {
                    out.trace(SpeechStep::IdleHoldReleased { hold });
                }
                self.run_release(crate::run_result::Owned::Hold(hold), out);
            }
            SpeechCommand::ClaimUserTurn => {
                let (depth, eff) = self.session.states.claim_user_turn(now);
                out.machine().append(eff);
                out.trace(TurnStep::UserTurnClaims { depth });
            }
            SpeechCommand::ReleaseUserTurn => {
                let speaking = self.activity.audio_recognition.user_speaking();
                let (depth, eff) = self.session.states.release_user_turn(now, speaking);
                out.machine().append(eff);
                out.trace(TurnStep::UserTurnClaims { depth });
            }
            SpeechCommand::Say { text, allow_interruptions, add_to_chat_ctx, call } => {
                // `say(text)` with no TTS and an enabled audio output: refused before a handle
                // exists. `say(audio=)` is exempt; a detached or disabled output speaks as text.
                if !matches!(text, SayText::Audio { .. })
                    && self.session.tts.is_none()
                    && self.shared.output.plays_audio()
                {
                    out.trace(SpeechStep::SayRefused { why: SAY_WITHOUT_TTS_REFUSED });
                    self.fail_waiter(call, SAY_WITHOUT_TTS_REFUSED);
                    return;
                }
                let h = self.start_speech(
                    now,
                    SpeechSpec {
                        add_to_chat_ctx,
                        ..SpeechSpec::new(
                            SpeechKind::Say,
                            match text {
                                SayText::Text(text) => ReplySource::Text(text),
                                SayText::Stream(rx) => {
                                    ReplySource::TextStream(std::sync::Mutex::new(Some(rx)))
                                }
                                SayText::Audio { text, rx } => {
                                    ReplySource::Audio { text, rx: std::sync::Mutex::new(Some(rx)) }
                                }
                            },
                            allow_interruptions
                                .unwrap_or(self.session.config.effective.turn.interruption.enabled),
                        )
                    },
                    out,
                );
                self.answer(call, Answer::Speech(h));
            }
            SpeechCommand::GenerateReply {
                user_input,
                instructions,
                allow_interruptions,
                tool_choice,
                chat_ctx,
                toolset,
                on_enter,
                call,
            } => {
                let h = self.generate_reply_speech(
                    now,
                    user_input,
                    instructions,
                    allow_interruptions,
                    tool_choice,
                    chat_ctx,
                    toolset,
                    on_enter,
                    out,
                );
                self.answer(call, Answer::Speech(h));
            }
            // `AgentTask.complete()` from a speech's tool: recorded for the run that ends on it.
            SpeechCommand::SetRunOutput { speech, output } => {
                let ok = matches!(output, crate::run_result::RunOutput::Value(_));
                out.trace(HandoffStep::RunOutput { speech, ok });
                self.session.runs.record_output(speech, output);
            }
            // The AMD system's authorization hold: the scheduling task keeps its head.
            SpeechCommand::PauseAuthorization => {
                out.trace(SpeechStep::AuthorizationPaused);
                self.activity.speech_scheduler.authorization_paused = true;
            }
            // Its release: the head is authorized in this same cycle.
            SpeechCommand::ResumeAuthorization => {
                out.trace(SpeechStep::AuthorizationResumed);
                self.activity.speech_scheduler.authorization_paused = false;
            }
            SpeechCommand::Interrupt { force, call } => {
                // The AMD's verdict is a detector's cut, not a host interrupt: no `HostInterrupt`
                // step.
                let cause = match self.command_origin {
                    Origin::Amd => InterruptCause::Detector,
                    Origin::Host
                    | Origin::Tool { .. }
                    | Origin::Hook
                    | Origin::Task
                    | Origin::Ivr => InterruptCause::Host,
                };
                if cause == InterruptCause::Host {
                    out.trace(InterruptStep::HostInterrupt { speech: None, force });
                }
                match self.interrupt_all(now, force, cause, out) {
                    Ok(speeches) => {
                        self.activity.interrupt_waiters.push(InterruptWaiter { speeches, call });
                    }
                    // The reference raises here, before the socket and the queue.
                    Err(refused) => self.answer(call, Answer::Interrupt(Err(refused))),
                }
            }
            SpeechCommand::InterruptSpeech { speech: id, force } => {
                out.trace(InterruptStep::HostInterrupt { speech: Some(id), force });
                let tell_socket = if !force
                    && self.activity.speech_scheduler.status(id) == Some(Status::Working)
                {
                    // Unforced, on a speech whose tools are running (the tool interrupting its own
                    // handle): scoped to that step, journalled, and the tool reply still follows.
                    // Forced is the host ending the speech, tools and all.
                    out.trace(InterruptStep::Interrupted {
                        speech: id,
                        cause: InterruptCause::Host,
                        force,
                        position: None,
                    });
                    false
                } else {
                    // Current or queued. The refusal is the same one by name.
                    match self.interrupt_speech(now, id, InterruptCause::Host, force, out) {
                        // The socket too, once, for a pipeline speech on a realtime session.
                        Ok(InterruptOutcome::Now { .. }) => !self.is_realtime_speech(id),
                        // Already interrupted or done: a no-op, so another speech's live
                        // generation is not cancelled under it.
                        Ok(InterruptOutcome::Already) => false,
                        // The reference raises before anything moves.
                        Err(_) => {
                            out.trace(InterruptStep::ByAudioActivityRefused(
                                InterruptRefusal::NotAllowed,
                            ));
                            false
                        }
                    }
                };
                if tell_socket {
                    self.realtime_interrupt(out);
                }
                // The user taking the floor back invalidates a preemptive generation about what
                // they were saying.
                self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, out);
            }
            // `disallow_interruptions()` and its release: read live by interruption, so no scheduling
            // step. A done handle refuses quietly; an unknown id is a no-op.
            SpeechCommand::SetAllowInterruptions { speech, allow } => {
                if let Some(h) = self.activity.speech_scheduler.handle(speech) {
                    let _ = h.set_allow_interruptions(allow);
                    out.trace(InterruptStep::InterruptionsToggled {
                        speech,
                        allow: h.allow_interruptions(),
                    });
                }
            }
            // A background tool's later update or return: the pair, the event, the coalesced reply.
            SpeechCommand::ToolUpdate { call_id, update } => {
                self.on_tool_update(now, call_id, update, out)
            }
            SpeechCommand::FillerStart {
                id,
                call_id,
                speech,
                delay,
                interval,
                max_steps,
                source,
            } => self
                .on_filler_start(now, id, call_id, speech, delay, interval, max_steps, source, out),
            SpeechCommand::FillerStop { id } => {
                self.end_filler(id, crate::trace::FillerEnd::Stopped, out)
            }
        }
    }

    /// Shared with the run's own replies.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_reply_speech(
        &mut self,
        now: Ts,
        user_input: Option<String>,
        instructions: Option<String>,
        allow_interruptions: Option<bool>,
        tool_choice: Option<lk_agents_abi::ToolChoice>,
        chat_ctx: Option<ChatContext>,
        toolset: Option<Arc<Toolset>>,
        on_enter: bool,
        out: &mut Out,
    ) -> Arc<SpeechHandle> {
        let wall = self.step.wall();
        let new_message = user_input
            .map(|text| ChatMessage::text(self.session.ids.mint("item"), Role::User, &text, wall));
        let turn_instructions = instructions.map(|text| TurnInstructions {
            instructions: Instructions::plain(text),
            message_id: self.session.ids.mint("item"),
        });
        self.start_speech(
            now,
            SpeechSpec {
                new_message,
                turn_instructions,
                on_enter,
                tool_choice,
                chat_ctx,
                toolset,
                ..SpeechSpec::new(
                    SpeechKind::GenerateReply,
                    ReplySource::Llm,
                    allow_interruptions
                        .unwrap_or(self.session.config.effective.turn.interruption.enabled),
                )
            },
            out,
        )
    }

    /// The reply to a decided user turn (`on_end_of_turn`'s `_generate_reply`). A preemptive
    /// generation for another turn is discarded; adopted, the user item enters the conversation
    /// stamped at the turn's onset.
    pub(crate) fn reply_to_turn(
        &mut self,
        now: Ts,
        info: &EndOfTurnInfo,
        edits: TurnEdits,
        out: &mut Out,
    ) {
        self.activity.last_turn = Some(info.turn_id);
        // A realtime session answers on its own socket: the words go as an item, the buffer is
        // committed when the server did not segment, the reply is asked for.
        if self.activity.rt_session.is_some() {
            self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, out);
            let user = self.user_turn_item(info, edits.user_text.as_deref());
            let has_words = user
                .content
                .iter()
                .any(|c| matches!(c, lk_agents_abi::ChatContent::Text(t) if !t.trim().is_empty()));
            if has_words {
                let item = Box::new(ChatItem::Message(user.clone()));
                let push = lk_agents_abi::PushItem { item, reply_required: true };
                self.rt_send(lk_agents_abi::RealtimeCommand::PushItem(push), out);
            }
            self.commit_item(
                ChatItem::Message(user),
                Source::UserTurn { turn: info.turn_id },
                Visibility::Both,
                out,
            );
            // The commit of the server's buffer and the ask, as one request.
            self.realtime_user_turn(now, out);
            // The realtime reply's speech is created when the server answers; no speech id here.
            self.emit_eou_metrics(info, None, out);
            return;
        }
        // A preemptive generation for another turn, against a record since changed, or whose
        // request the hook edited, answers a question this turn is not asking.
        let edited =
            edits.instructions.is_some() || edits.chat_ctx.is_some() || edits.user_text.is_some();
        let stale = self.activity.preemptive_record_version != self.session.chat_ctx.version();
        if let Some(preemptive_turn) = self.activity.preemptive_turn.take()
            && (preemptive_turn != info.turn_id || edited || stale)
            && self.activity.speech_scheduler.preemptive_generation().is_some()
        {
            self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, out);
        }
        let adopted = self.adopt_preemptive_generation(now, &info.new_transcript, out);
        let TurnEdits { instructions, chat_ctx, user_text } = edits;
        let user = self.user_turn_item(info, user_text.as_deref());
        if let Some(preemptive) = adopted {
            self.commit_item(
                ChatItem::Message(user),
                Source::UserTurn { turn: info.turn_id },
                Visibility::Both,
                out,
            );
            self.set_agent_state(self.step.now, AgentState::Thinking, out);
            // The adopted preemptive generation is the turn's reply: the record names it.
            self.emit_eou_metrics(info, Some(preemptive), out);
            return;
        }
        let handle = self.start_speech(
            now,
            SpeechSpec {
                new_message: Some(user),
                chat_ctx,
                turn: Some(info.turn_id),
                instructions_override: instructions,
                ..SpeechSpec::new(
                    SpeechKind::TurnReply,
                    ReplySource::Llm,
                    self.session.config.effective.turn.interruption.enabled,
                )
            },
            out,
        );
        self.emit_eou_metrics(info, Some(handle.id()), out);
    }

    /// Emitted once the reply speech exists so it carries its id; never for a turn with no reply.
    /// `metadata` names the detector (model, or mode with `model_name="unknown"`).
    pub(crate) fn emit_eou_metrics(
        &mut self,
        info: &EndOfTurnInfo,
        speech: Option<SpeechId>,
        out: &mut Out,
    ) {
        use crate::agent_activity::options::TurnDetection;
        let detection = self.session.config.effective.turn.detection;
        let metadata = match (self.activity.audio_recognition.eot(), detection) {
            (Some(eot), _) => Some(eot.id().metadata()),
            (None, TurnDetection::Vad) => Some(mode_metadata("vad")),
            (None, TurnDetection::Stt) => Some(mode_metadata("stt")),
            (None, TurnDetection::Manual) => Some(mode_metadata("manual")),
            (None, TurnDetection::RealtimeLlm) => Some(mode_metadata("realtime_llm")),
            (None, TurnDetection::Auto) => None,
        };
        let secs = |d: Option<std::time::Duration>| d.map_or(0.0, |d| d.as_secs_f64());
        let record = lk_agents_abi::metrics::base::EouMetrics {
            timestamp: self.step.wall_secs(),
            end_of_utterance_delay: secs(info.metrics.end_of_turn_delay),
            transcription_delay: secs(info.metrics.transcription_delay),
            on_user_turn_completed_delay: secs(info.metrics.on_user_turn_completed_delay),
            speech_id: speech.map(|s| crate::speech_handle::SpeechWireId(s).to_string()),
            metadata,
        };
        out.machine().emit(SessionEvent::MetricsCollected {
            metrics: Box::new(lk_agents_abi::metrics::base::AgentMetrics::Eou(record)),
            speech,
        });
    }

    /// Stamped at the turn's onset when known.
    pub(crate) fn user_turn_item(
        &mut self,
        info: &EndOfTurnInfo,
        text: Option<&str>,
    ) -> ChatMessage {
        let created_at = match info.metrics.started_speaking_at {
            Some(at) => self.step.wall_at(at),
            None => self.step.wall(),
        };
        let text = text.unwrap_or(&info.new_transcript);
        let mut user =
            ChatMessage::text(self.session.ids.mint("item"), Role::User, text, created_at);
        user.transcript_confidence = Some(info.transcript_confidence);
        user
    }

    /// Through the one write, in order with `thinking`.
    /// The single entry point for every speech. The task is created **eagerly**, before
    /// scheduling, which is what puts several replies in flight at once.
    pub(crate) fn start_speech(
        &mut self,
        now: Ts,
        spec: SpeechSpec,
        out: &mut Out,
    ) -> Arc<SpeechHandle> {
        let wall = self.step.wall();
        let user_metrics = spec.new_message.as_ref().map(|m| m.metrics.clone());
        let is_reply = matches!(spec.source, ReplySource::Llm);
        let kind = spec.kind;

        // `chat_ctx=` and `ReplaceItems` replace what this request is planned against; the record
        // is not written.
        let record =
            spec.chat_ctx.clone().unwrap_or_else(|| self.session.chat_ctx.agent_chat_ctx());
        // The frame's prompt for the modality, or a hook's edit for this turn.
        let instructions = spec
            .instructions_override
            .clone()
            .unwrap_or_else(|| self.activity.instructions.render(Some(Modality::Audio)));
        // Read once, so the request's specs and host objects cannot disagree.
        let advertised = self.activity.advertised_toolset(&self.session.tools);
        let turn_toolset = spec.toolset.clone().unwrap_or_else(|| Arc::clone(&advertised));
        let plan = crate::generation::PlanSpec {
            record: spec.chat_ctx.clone(),
            new_message: spec.new_message.clone(),
            instructions: spec.instructions_override.clone(),
            turn_instructions: spec.turn_instructions.clone(),
            // Snapshotted for this request: handoffs replace the list, replies overlap.
            tools: turn_toolset.specs.clone(),
            on_enter: spec.on_enter,
            running_tool_calls: self.activity.tools.running_tool_calls(),
            audio_output_enabled: self.shared.output.plays_audio(),
            text_output_enabled: self.shared.output.shows_transcription(),
        };
        let request = match &spec.source {
            ReplySource::Llm
            | ReplySource::Text(_)
            | ReplySource::TextStream(_)
            | ReplySource::Audio { .. } => plan.plan(&record, &instructions, wall),
            // No plan: the server decides what to say. The step needs the record and tools; there
            // is no TTS leg, but the words still go on the text stream.
            ReplySource::Realtime { .. } => crate::generation::Request {
                chat_ctx: record,
                tools: turn_toolset.specs.clone(),
                tts_enabled: false,
                text_output_enabled: self.shared.output.shows_transcription(),
                instructions_hash: crate::generation::stable_hash(&instructions),
            },
        };

        // A reply created while a speech plays or is being cut is requested once that speech's
        // item is on the record.
        let after = self
            .current_handle()
            .filter(|h| {
                !h.done()
                    && (h.interrupted()
                        || self.session.states.agent_state() == AgentState::Speaking)
            })
            .map(|h| h.id());
        let replan = match (after, &spec.source) {
            (Some(after), ReplySource::Llm) => {
                Some(crate::generation::Replan { after, spec: plan })
            }
            (
                Some(_),
                ReplySource::Text(_)
                | ReplySource::TextStream(_)
                | ReplySource::Audio { .. }
                | ReplySource::Realtime { .. },
            )
            | (None, _) => None,
        };

        let id = self.session.ids.next_speech_id();
        // A preemptive generation plans against the message but does not commit it;
        // `generate_reply` commits on adoption.
        if let (Some(user), true) = (spec.new_message, kind.scheduled()) {
            let source = match spec.turn {
                Some(turn) => Source::UserTurn { turn },
                None => Source::UserInput { speech: id },
            };
            self.commit_item(ChatItem::Message(user), source, Visibility::Both, out);
        }

        let speech_id = crate::speech_handle::SpeechWireId(id).to_string();
        let handle = Arc::new(SpeechHandle::new(
            id,
            speech_id,
            spec.allow_interruptions,
            InputDetails::default(),
        ));
        self.activity.note_speech_agent(id);

        let message_id = self.session.ids.mint("item");
        let ctx = ReplyContext {
            preemptive_tts: self.session.config.effective.turn.preemptive.enabled
                && self.session.config.effective.turn.preemptive.preemptive_tts,
            request,
            replan,
            source: spec.source,
            tool_choice: spec.tool_choice,
            frame: self.current,
            toolset: spec.toolset,
            providers: Providers {
                llm: self.session.llm.as_ref().cloned(),
                tts: self.session.tts.as_ref().cloned(),
                toolset: advertised,
                tools_host_extra: self.session.tool_hosts.clone(),
            },
            user_metrics,
            message_id,
            speech: self.session.config.effective.speech,
        };

        // Past the close's hard edge no generation is started.
        let past_edge = self.session.closing_past_edge();
        if !past_edge {
            // Eager: the task parks in `wait_for_authorization` until it is the current speech or
            // is interrupted.
            let shared = Arc::clone(&self.shared);
            let task = Arc::clone(&handle);
            let abort = self
                .shared
                .spawner
                .spawn_unboxed(async move { pipeline_reply_task(shared, task, ctx).await });
            // The only way a spawned generation is cancellable; a dropped `JoinHandle` detaches.
            handle.add_task(Arc::new(abort));
        }

        // Announced, recorded and scheduled in `request`, the one place every speech passes.
        let admitted =
            self.schedule_speech(now, &handle, kind, spec.add_to_chat_ctx, spec.host_asked, out);
        if admitted && kind.scheduled() {
            handle.mark_scheduled();
        }
        // `thinking` when the generation exists, only for an admitted reply (`say()` goes straight
        // to `speaking`; a refused or discarded preemptive generation never showed thinking).
        if admitted && is_reply && kind.scheduled() && !past_edge {
            self.set_agent_state(self.step.now, AgentState::Thinking, out);
        }
        // The open run owns a session-level speech; turn replies, preemptive generations and
        // server-created generations are not its.
        if kind.user_initiated() {
            self.run_own(crate::run_result::Owned::Speech(id), out);
        }
        handle
    }

    pub(crate) fn on_generation(&mut self, now: Ts, ev: GenEvent, out: &mut Out) {
        match ev {
            // A text-only turn's first frame; with audio output the edge is `PlaybackStarted`'s.
            GenEvent::FirstFrame { speech } => self.agent_started_speaking(now, speech, out),

            GenEvent::Done { speech, outcome, interrupted } => {
                self.activity.set_tools_in_flight(speech, false);
                self.finish_reply(now, speech, *outcome, interrupted, out);
            }

            // One step's words are done, before its tools run.
            GenEvent::StepSpoken {
                speech,
                step,
                interrupted,
                has_tool_output,
                tools_running,
                item,
            } => {
                if tools_running {
                    self.activity.set_tools_in_flight(speech, true);
                } else {
                    self.activity.set_tools_in_flight(speech, false);
                }
                self.step_spoken(
                    now,
                    speech,
                    step,
                    interrupted,
                    has_tool_output,
                    tools_running,
                    item,
                    out,
                );
            }
            GenEvent::Trace(step) => out.trace(step),
            GenEvent::ProviderFailed { speech, leg, recoverable, error } => {
                let source = match leg {
                    crate::trace::ProviderKind::Llm => crate::events::ErrorSource::Llm,
                    crate::trace::ProviderKind::Tts => crate::events::ErrorSource::Tts,
                };
                self.report_error(now, source, recoverable, *error, Some(speech), out);
            }
            GenEvent::KeytermAnswer { req, arguments, failed, metrics } => {
                self.on_keyterm_answer(now, req, arguments, failed, metrics, out);
            }

            // Reported by a task; said by the loop, which is the only place with the bus.
            GenEvent::Emit(ev) => out.machine().emit(*ev),

            GenEvent::ToolsFinished { speech, step, commit, executed, warnings, verdict } => {
                // With a host running hooks, `function_tools_executed` is its to emit, and the step
                // waits for what the handlers leave.
                if let (Some(event), Some(verdict)) = (&executed, verdict) {
                    let id = self.activity.hooks.mint();
                    let call = HookCall::FunctionToolsExecuted {
                        id,
                        speech,
                        event: Box::new((**event).clone()),
                    };
                    let wait = self.hook_wait();
                    if self.activity.hook_bound.is_some()
                        && self.issue_hook(now, call, Some(wait), out)
                    {
                        let event = (**event).clone();
                        let handling = Handling { speech, step, event, warnings, verdict };
                        self.activity.tools.handle_executed(id, handling);
                        return;
                    }
                    self.finish_tools(
                        speech,
                        step,
                        commit,
                        executed,
                        warnings,
                        Some(verdict),
                        false,
                        out,
                    );
                    return;
                }
                self.finish_tools(speech, step, commit, executed, warnings, None, false, out);
            }

            // The reply's tools are done and it has a next step: behind whatever was queued.
            GenEvent::ToolStep { speech, step } => {
                out.trace(ToolStep::StepStarted { speech, step });
                self.requeue_speech(now, speech, out);
            }

            GenEvent::Handoff { speech, agent } => {
                // The swap is the driver's once the host has described the `Agent` handle, on the
                // one host stream so it precedes this speech's `SpeechDone`.
                self.request_describe(now, speech, agent, out);
            }

            GenEvent::Failed { speech, error } => {
                self.activity.set_tools_in_flight(speech, false);
                // Released by the close's edge already: the edge was its end (`CloseAct::Released`).
                if self.activity.finished.contains(&speech)
                    && !self.activity.speech_scheduler.contains(speech)
                {
                    return;
                }
                out.trace(SpeechStep::ReplyFailed {
                    speech,
                    class: crate::trace::FailureClass::of(&error),
                });
                if let Some(handle) = self.activity.speech_scheduler.handle(speech) {
                    // Moved, not formatted: the binding releases a `ProviderError::Host`'s entry.
                    handle.mark_done(Some(*error));
                }
                self.release_speech(now, speech, out);
            }

            GenEvent::PlaybackStarted { speech } => {
                // The launch hold is over for this speech: its first frame is out (I-06, I-07).
                self.shared.launch.note_started(speech);
                out.trace(PlayoutStep::PlaybackStarted { speech });
                // `_on_first_frame` listens on the sink's `playback_started`: the edge is the sink's
                // report, not the words half's (which waits for its segment).
                self.agent_started_speaking(now, speech, out);
            }
            GenEvent::LaunchReleased { speech } => {
                // Released because the user stopped, or cancelled because the speech became
                // uninterruptible while parked.
                let cause = match self
                    .activity
                    .speech_scheduler
                    .handle(speech)
                    .map(|h| h.allow_interruptions())
                {
                    Some(true) | None => GateCause::Launch,
                    Some(false) => GateCause::Cancelled,
                };
                out.trace(InterruptStep::PlayoutGate { speech, op: GateOp::Resume, cause })
            }
            GenEvent::ChatFinished { speech, step, error } => {
                let failed = error.is_some();
                out.trace(GenerationStep::ChatFinished { speech, step, error });
                // A failed preemptive generation answers nothing; the commit starts a fresh reply.
                if failed && self.activity.speech_scheduler.preemptive_generation() == Some(speech)
                {
                    self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, out);
                }
            }
            GenEvent::ToolCallStarted { speech, step, call, hosted } => {
                self.on_tool_started(now, speech, step, *call, hosted, out);
            }
            GenEvent::ToolCallEnded { speech, call_id, end, message } => {
                self.on_tool_ended(now, speech, call_id, end, message, out);
            }
        }
    }

    /// `_on_first_frame`: `speaking` and `_on_start_of_agent_speech` (which enables interruption) at
    /// the same instant, once per step.
    fn agent_started_speaking(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        if !self.activity.spoke.insert(speech) {
            return;
        }
        // The labels arm the AEC warmup at the session's first `speaking` edge with audio output.
        self.set_agent_state(now, AgentState::Speaking, out);
        let eff = self.activity.audio_recognition.on_start_of_agent_speech(now, now);
        out.machine().append(eff);
        out.trace(SpeechStep::AgentStartedSpeaking { speech });
    }

    /// Releases the current speech and moves the state (`thinking` while tools run, else
    /// `listening`). `add_to_chat_ctx=False` keeps the item out of the conversation only.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_spoken(
        &mut self,
        now: Ts,
        speech: SpeechId,
        step: u32,
        interrupted: bool,
        has_tool_output: bool,
        tools_running: bool,
        item: Option<Box<ChatMessage>>,
        out: &mut Out,
    ) {
        out.trace(SpeechStep::StepSpoken { speech, step, interrupted });
        // The step's edge is spent: the next step's first frame is a new one.
        self.activity.spoke.remove(&speech);
        if let Some(msg) = item {
            let item = ChatItem::Message(*msg);
            let source = match self.activity.speech_scheduler.kind(speech) {
                Some(SpeechKind::Say) => Source::Say { speech },
                Some(
                    SpeechKind::TurnReply
                    | SpeechKind::GenerateReply
                    | SpeechKind::Preemptive
                    | SpeechKind::Realtime,
                )
                | None => Source::Reply { speech, step },
            };
            if self.activity.speech_scheduler.add_to_chat_ctx(speech) {
                self.commit_item(item, source, Visibility::Both, out);
            } else if let Some(h) = self.activity.speech_scheduler.handle(speech) {
                h.item_added(&item);
            }
        }
        self.activity.speech_scheduler.spoken(speech, now);
        // A blocked current speech's step ending is what a pause from inside its tool waits on.
        self.settle_scheduling_pause(out);
        // The state moves on the speech's interruption, not the step's; a filler ending inside
        // another speech's tool span goes back to `thinking` (listening would arm the away timer).
        let speech_interrupted =
            self.activity.speech_scheduler.handle(speech).is_some_and(|h| h.interrupted_speech());
        let eos = end_of_speech(
            speech_interrupted,
            has_tool_output,
            tools_running || self.tools_running_elsewhere(speech),
            self.session.states.agent_state() == AgentState::Speaking,
            self.session.config.effective.turn.interruption.detection_enabled,
        );
        self.apply_end_of_speech(now, eos, out);
    }

    fn apply_end_of_speech(&mut self, now: Ts, eos: EndOfSpeech, out: &mut Out) {
        if let Some(state) = eos.state {
            self.set_agent_state(now, state, out);
        }
        if eos.end_of_agent_speech {
            let eff = self.activity.audio_recognition.on_end_of_agent_speech(now, now);
            out.machine().append(eff);
        }
        if eos.restore_interruption_by_audio_activity {
            self.session.by_audio_activity = true;
        }
    }

    pub(crate) fn finish_reply(
        &mut self,
        now: Ts,
        speech: SpeechId,
        outcome: GenTurnOutcome,
        interrupted: bool,
        out: &mut Out,
    ) {
        // Already released by the close's hard edge: a second terminal would be a second end.
        if self.activity.finished.contains(&speech)
            && !self.activity.speech_scheduler.contains(speech)
        {
            return;
        }
        out.trace(SpeechStep::ReplyFinished {
            speech,
            text: outcome.forwarded_text.clone(),
            interrupted,
        });
        if let Some(h) = self.activity.speech_scheduler.handle(speech) {
            out.trace(SpeechStep::ItemsRecorded {
                speech,
                ids: h.chat_items().iter().map(|it| it.id().to_owned()).collect(),
            });
        }
        // **`speaking` belongs to the current speech**: releasing a refused `Say` or a discarded
        // preemptive generation must not drop the agent out of `speaking` while another speech
        // plays.
        let was_current = self.activity.speech_scheduler.current_speech() == Some(speech);
        let eos = end_of_speech(
            interrupted,
            false,
            false,
            was_current && self.session.states.agent_state() == AgentState::Speaking,
            self.session.config.effective.turn.interruption.detection_enabled,
        );
        self.apply_end_of_speech(now, eos, out);

        self.release_speech(now, speech, out);
    }

    pub(crate) fn release_speech(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        self.activity.spoke.remove(&speech);
        // `_scheduling_task`'s post-await release: a finished current speech that is the paused one
        // drops the pause and resumes the sink. The release at `Authorized` can miss a pause taken
        // in the same instant, which would leave the sink paused until the close.
        if self.activity.speech_scheduler.current_speech() == Some(speech) {
            self.release_paused_speech(speech, out);
        }
        // A close ending a speech with blocking tools still running: they are answered with the
        // closing error before the speech's own `SpeechDone`; nothing is journalled after `close`.
        if self.session.closing.is_some()
            && self.activity.speech_scheduler.contains(speech)
            && self.activity.tools.live_blocking(speech) > 0
            && self.session.close_phase() != Some(ClosePhase::Closed)
        {
            self.end_live_tools(now, speech, out);
        }
        if let Some(h) = self.activity.speech_scheduler.handle(speech) {
            // The host learns of the end only here, with the handle already `done()` (idempotent).
            if !h.done() {
                h.mark_done(None);
            }
            let outcome = if matches!(h.exception(), Ok(Some(_))) {
                crate::events::SpeechOutcome::Failed
            } else if h.interrupted() {
                crate::events::SpeechOutcome::Interrupted
            } else {
                crate::events::SpeechOutcome::Completed
            };
            out.machine().emit(SessionEvent::SpeechDone { speech, outcome });
            // The run that owned it lets it go, and its verdict moves to this speech.
            let h = Arc::clone(h);
            self.run_speech_ended(&h, out);
        }
        // The deferred tool reply's fate is read off the handle; the tools' fillers go with it.
        self.tool_reply_done(now, speech, out);
        self.fillers_speech_ended(speech, out);
        // The handle leaves with the queue's entry (`speech_finished`, below); `speech_frames`
        // keeps its entry: a call and output can land after the speech is done.
        self.activity.finished.insert(speech);
        self.publish_activity();
        self.speech_finished(now, speech, out);
    }

    /// `_update_agent_state(next)`: the one door for agent-state writes, reading the audio-output
    /// fact the warmup edge gates on at the instant of the edge.
    pub(crate) fn set_agent_state(
        &mut self,
        now: Ts,
        next: AgentState,
        out: &mut Out,
    ) -> Option<AgentState> {
        let audio_output_ready = self.shared.output.plays_audio();
        let (old, eff) = self.session.update_agent_state(now, next, audio_output_ready);
        out.machine().append(eff);
        old
    }

    /// `wait_for_idle`: nothing is speaking, queued, winding down or deciding a turn, and no user
    /// turn is claimed.
    pub(crate) fn is_idle(&self) -> bool {
        use crate::audio_recognition::Phase;
        let activity = &self.activity;
        let deciding = match activity.audio_recognition.phase() {
            Phase::Idle => false,
            Phase::Speaking { .. }
            | Phase::Ended { .. }
            | Phase::Awaiting { .. }
            | Phase::Delaying { .. }
            | Phase::ManualCommitted { .. } => true,
            Phase::Committed { speaking, .. } => *speaking,
        };
        // Busy: every held handle is in flight, except one whose tools run (the current speech is
        // released for `foreground()` from inside a tool) and a live preemptive generation. A
        // speech parked for an inline task still plays its step, so it counts.
        let scheduling = &activity.speech_scheduler;
        let winding_down = scheduling.speech_ids().any(|id| match scheduling.status(id) {
            None => true,
            Some(Status::Working) => !activity.tools_in_flight.contains(&id),
            Some(Status::Preemptive) => false,
            Some(Status::Parked | Status::Current | Status::Queued | Status::Discarded) => true,
        });
        scheduling.no_current_speech()
            && scheduling.speech_q().is_empty()
            && !winding_down
            && !deciding
            && activity.pending_turn.is_none()
            && activity.handoff.held_user_turn.is_none()
            && self.session.states.user_turn_claims() == 0
    }

    pub(crate) fn settle_waiters(&mut self, now: Ts, out: &mut Out) {
        self.settle_interrupt_waiters(out);

        // A pending deferred tool reply takes the idle instant first.
        self.try_deliver_tool_reply(now, out);
        self.pump_fillers(now, out);
        if !self.session.idle.has_waiters() || !self.is_idle() {
            return;
        }
        for waiter in self.session.idle.admit() {
            out.trace(SpeechStep::IdleAnswered { hold: waiter.grant, closing: false });
            self.answer(waiter.call, waiter.answer(Ok(())));
        }
    }

    pub(crate) fn withdraw_idle_asks(&mut self, speech: SpeechId, out: &mut Out) {
        for w in self.session.idle.withdraw(speech) {
            out.trace(SpeechStep::IdleAskWithdrawn { hold: w.grant, speech });
            if let Some(id) = w.grant {
                self.run_release(crate::run_result::Owned::Hold(id), out);
            }
            self.answer(w.call, w.answer(Err(IdleRefused::Closing)));
        }
    }

    pub(crate) fn refuse_idle_waiters(&mut self, out: &mut Out) {
        for waiter in self.session.idle.take_waiters() {
            // A refused hold was never granted, so none is named.
            out.trace(SpeechStep::IdleAnswered { hold: None, closing: true });
            // A hold refused was never granted: whatever run owned the ask lets it go.
            if let Some(id) = waiter.grant {
                self.run_release(crate::run_result::Owned::Hold(id), out);
            }
            self.answer(waiter.call, waiter.answer(Err(IdleRefused::Closing)));
        }
    }
}

/// `Metadata(model_name="unknown", model_provider=<the turn_detection string>)`.
fn mode_metadata(mode: &str) -> lk_agents_abi::metrics::base::Metadata {
    lk_agents_abi::metrics::base::Metadata {
        model_name: Some("unknown".to_owned()),
        model_provider: Some(mode.to_owned()),
    }
}
