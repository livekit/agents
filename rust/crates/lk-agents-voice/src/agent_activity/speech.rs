//! Speech from request to release: `say`, `generate_reply` and the turn's reply through
//! `start_speech`, the generation's reports, the agent state, and `wait_for_idle`.

use crate::agent::{Source, Visibility};
use crate::agent_activity::false_interruption::{InterruptOutcome, InterruptWaiter};
use crate::agent_activity::scheduling::{PreemptiveDiscard, SpeechId, SpeechKind};
use crate::agent_session::effects::Out;
use crate::agent_session::effects::{GateCause, GateOp};
use crate::agent_session::{
    Answer, ClosePhase, Driver, IdleWaiter, Origin, SayText, SpeechCommand,
};
use crate::audio_recognition::{EndOfTurnInfo, TurnEdits};
use crate::events::SessionEvent;
use crate::generation::{EndOfSpeech, TurnOutcome as GenTurnOutcome, end_of_speech};
use crate::generation::{GenEvent, Providers, ReplyContext, ReplySource};
use crate::speech_handle::SpeechHandle;
use crate::tool_executor::Toolset;
use crate::trace::{
    CoreStep, GenerationStep, InterruptCause, InterruptStep, PlayoutStep, SpeechStep, ToolStep,
    TurnStep,
};
use crate::tts::markup::ChatMessageExt;
use lk_agents_abi::ChatItem;
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::{ChatContext, ChatMessage, Modality, Role, Ts};
use std::sync::Arc;

/// A `say(stream)` the realtime model would speak itself (no TTS): the wire carries one text item.
pub(crate) const SAY_SHAPE_REFUSED_ON_REALTIME: &str =
    "say(stream) on a realtime model with no TTS: the wire takes one text item";

/// A text `say()` with no TTS while audio output is attached and enabled, refused before a handle
/// exists (`AgentActivity.say`'s first `RuntimeError`).
pub(crate) const SAY_WITHOUT_TTS_REFUSED: &str = "trying to generate speech from text without a TTS model or a RealtimeSession that supports say(); add a TTS model to AgentSession to enable say()";

pub(crate) struct SpeechSpec {
    pub(crate) new_message: Option<ChatMessage>,
    pub(crate) turn_instructions: Option<String>,
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
    /// The decided turn this speech replies to: its `EOUMetrics` go out between the speech's
    /// announcement and the user item, as the reference's `on_end_of_turn` emits them.
    pub(crate) end_of_turn: Option<EndOfTurnInfo>,
    /// The handle a host's `say()` / `generate_reply()` answered: the speech is created over it.
    pub(crate) host_handle: Option<Arc<SpeechHandle>>,
    /// The turn's reply answers an `InputRequired(tool=)`.
    pub(crate) answering: Option<crate::tool_executor::Answering>,
}

impl SpeechSpec {
    /// Added to the context, planned against the record and the agent's prompt and tools.
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
            end_of_turn: None,
            host_handle: None,
            answering: None,
        }
    }
}

impl Driver {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pub(crate) fn on_speech(&mut self, now: Ts, cmd: SpeechCommand, out: &mut Out) {
        // The host's speech during the drain is the arriving agent's (`_next_activity`); the
        // leaving agent's own tools and hooks are refused by the paused scheduler, as before.
        if matches!(cmd, SpeechCommand::Say { .. } | SpeechCommand::GenerateReply { .. })
            && self.command_origin == Origin::Host
            && self.session.closing.is_none()
            && self.activity.handoff.draining()
        {
            return self.activity.handoff.held_speech.push(cmd);
        }
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
                        SpeechKind::GenerateReply,
                        "generate_reply(chat_ctx=, tools=) is not supported on a realtime model",
                        out,
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
                    false,
                    out,
                );
            }
            // The model speaks a `say` only with no TTS to (`AgentActivity.say`); with one, a
            // realtime session's `say` is synthesised like a pipeline's, audio and streams too.
            SpeechCommand::Say { text, allow_interruptions, add_to_chat_ctx, call }
                if !matches!(text, SayText::Audio { .. }) && self.realtime_says() =>
            {
                // The wire carries a string; a streamed `say` is refused by name.
                let SayText::Text(text) = text else {
                    out.trace(SpeechStep::SayRefused { why: SAY_SHAPE_REFUSED_ON_REALTIME });
                    self.fail_waiter(
                        call,
                        SpeechKind::Say,
                        "say(stream) is not supported on a realtime model without a TTS",
                        out,
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
            SpeechCommand::EndCall { speech } => self.end_call(speech),
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
                    && self.session.outputs.plays_audio()
                {
                    out.trace(SpeechStep::SayRefused { why: SAY_WITHOUT_TTS_REFUSED });
                    self.fail_waiter(call, SpeechKind::Say, SAY_WITHOUT_TTS_REFUSED, out);
                    return;
                }
                let interruptible = self.session.config.effective.turn.interruption.enabled();
                self.start_speech(
                    now,
                    SpeechSpec {
                        add_to_chat_ctx,
                        host_handle: call.as_ref().map(|c| Arc::clone(&c.speech)),
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
                            allow_interruptions.unwrap_or(interruptible),
                        )
                    },
                    out,
                );
                self.answer_speech(call);
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
                self.generate_reply_speech(
                    now,
                    user_input,
                    instructions,
                    allow_interruptions,
                    tool_choice,
                    chat_ctx,
                    toolset,
                    on_enter,
                    call.as_ref().map(|c| Arc::clone(&c.speech)),
                    out,
                );
                self.answer_speech(call);
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
                    Origin::Host | Origin::Tool { .. } | Origin::Hook | Origin::Ivr => {
                        InterruptCause::Host
                    }
                };
                if cause == InterruptCause::Host {
                    out.trace(InterruptStep::HostInterrupt { speech: None, force });
                }
                match self.interrupt_all(now, force, cause, out) {
                    Ok(speeches) => {
                        self.activity.interrupt_waiters.push(InterruptWaiter { speeches, call });
                    }
                    // The reference raises here, before the socket and the queue.
                    Err(refused) => out.answer(call, Answer::Interrupt(Err(refused))),
                }
            }
            SpeechCommand::InterruptSpeech { speech: id, force } => {
                out.trace(InterruptStep::HostInterrupt { speech: Some(id), force });
                // The whole speech, wherever it is: queued, playing, or in its tools (which finish
                // and commit, with no follow-up). The refusal is the same one by name.
                let interrupted = self.interrupt_speech(now, id, InterruptCause::Host, force, out);
                let tell_socket = match interrupted {
                    // The socket too, once, for a pipeline speech on a realtime session.
                    Ok(InterruptOutcome::Now) => !self.is_realtime_speech(id),
                    // Already interrupted or done: a no-op, so another speech's live generation is
                    // not cancelled under it.
                    Ok(InterruptOutcome::Already) => false,
                    // Disallowed by the time it landed: the handle drops the request.
                    Err(_) => {
                        out.trace(InterruptStep::HostInterruptRefused { speech: id });
                        if let Some(h) = self.activity.speech_scheduler.handle(id) {
                            h.interrupt_refused();
                        }
                        return;
                    }
                };
                if tell_socket {
                    self.realtime_interrupt(InterruptCause::Host, out);
                }
                // The user taking the floor back invalidates a preemptive generation about what
                // they were saying.
                self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, out);
            }
            // `disallow_interruptions()` and its release: read live by interruption, so no scheduling
            // step. A done handle refuses quietly; an unknown id is a no-op.
            SpeechCommand::HoldInterruptions { speech, hold } => {
                if let Some(handle) = self.activity.speech_scheduler.handle(speech) {
                    if hold {
                        let _ = handle.hold_interruptions();
                    } else {
                        handle.release_interruptions();
                    }
                    out.trace(InterruptStep::InterruptionsToggled {
                        speech,
                        allow: handle.allow_interruptions(),
                    });
                }
            }
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
        host_handle: Option<Arc<SpeechHandle>>,
        out: &mut Out,
    ) -> Arc<SpeechHandle> {
        let wall = self.step.wall();
        let new_message = user_input
            .map(|text| ChatMessage::text(self.session.ids.mint("item"), Role::User, &text, wall));
        let interruptible = self.session.config.effective.turn.interruption.enabled();
        self.start_speech(
            now,
            SpeechSpec {
                new_message,
                turn_instructions: instructions,
                on_enter,
                tool_choice,
                chat_ctx,
                toolset,
                host_handle,
                ..SpeechSpec::new(
                    SpeechKind::GenerateReply,
                    ReplySource::Llm,
                    allow_interruptions.unwrap_or(interruptible),
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
        // A realtime session answers on its own socket, from the audio it heard: the buffer is
        // committed when the server did not segment and the reply is asked for. The STT's words
        // are neither sent nor kept ("ignore stt transcription for realtime model"); the server's
        // own transcript fills the user item.
        if self.activity.rt_session.is_some() {
            self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, out);
            // The commit of the server's buffer and the ask, as one request; its speech is
            // announced now, so the metrics name it.
            let speech = self.realtime_user_turn(now, out);
            self.emit_eou_metrics(info, speech, out);
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
        // A tool claimed this turn (`InputRequired(tool=)`): the reply calls it, a different
        // request from any guessed one.
        let claimed = self.consume_claim();
        if claimed.is_some() && self.activity.speech_scheduler.preemptive_generation().is_some() {
            self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, out);
        }
        let adopted = self.adopt_preemptive_generation(now, &info.new_transcript, out);
        let TurnEdits { instructions, chat_ctx, user_text } = edits;
        let user = self.user_turn_item(info, user_text.as_deref());
        let answering = claimed.map(|tool| crate::tool_executor::Answering {
            tool,
            said: user.text_content().unwrap_or_default(),
        });
        if let Some(preemptive) = adopted {
            if let Some(handle) = self.activity.speech_scheduler.handle(preemptive) {
                handle.set_user_metrics(user.metrics.clone());
            }
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
        self.start_speech(
            now,
            SpeechSpec {
                new_message: Some(user),
                chat_ctx,
                turn: Some(info.turn_id),
                instructions_override: instructions,
                end_of_turn: Some(info.clone()),
                tool_choice: answering
                    .as_ref()
                    .map(|a| lk_agents_abi::ToolChoice::Named(a.tool.clone())),
                answering,
                ..SpeechSpec::new(
                    SpeechKind::TurnReply,
                    ReplySource::Llm,
                    self.session.config.effective.turn.interruption.enabled(),
                )
            },
            out,
        );
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
        let detection = self.session.config.effective.turn.detection();
        let metadata = match (self.activity.audio_recognition.eot_provider(), detection) {
            (Some(eot), _) => Some(eot.metadata()),
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
            speech_id: speech.map(|s| s.to_string()),
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
        // `_init_metrics_from_end_of_turn`: the reply's `e2e_latency` is measured from here.
        let m = &info.metrics;
        let secs = |d: Option<std::time::Duration>| d.map(|d| d.as_secs_f64());
        user.metrics = lk_agents_abi::MetricsReport {
            started_speaking_at: m.started_speaking_at.map(|at| self.step.wall_secs_at(at)),
            stopped_speaking_at: m.stopped_speaking_at.map(|at| self.step.wall_secs_at(at)),
            transcription_delay: secs(m.transcription_delay),
            end_of_turn_delay: secs(m.end_of_turn_delay),
            on_user_turn_completed_delay: secs(m.on_user_turn_completed_delay),
            stt_metadata: self.stt_identity().map(|id| id.metrics_metadata()),
            ..Default::default()
        };
        user
    }

    /// Through the one write, in order with `thinking`.
    /// The single admission point for every speech. The runtime launches accepted work after
    /// this step settles.
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
        // The agent's prompt for the modality, or a hook's edit for this turn.
        let instructions = spec
            .instructions_override
            .clone()
            .unwrap_or_else(|| self.activity.instructions.render(Some(Modality::Audio)));
        // Read once, so the request's specs and host objects cannot disagree.
        let advertised = self.activity.advertised_toolset(&self.session.tools);
        let turn_toolset = spec.toolset.clone().unwrap_or_else(|| Arc::clone(&advertised));
        let expressive = match &spec.source {
            ReplySource::Realtime => None,
            source => self.session.expressive_guide(matches!(source, ReplySource::Llm)),
        };
        let mut plan = crate::generation::PlanSpec {
            record: spec.chat_ctx.clone(),
            new_message: spec.new_message.clone(),
            instructions: spec.instructions_override.clone(),
            turn_instructions: spec.turn_instructions.clone(),
            // Snapshotted for this request: handoffs replace the list, replies overlap.
            tools: turn_toolset.specs.clone(),
            on_enter: spec.on_enter,
            running_tool_calls: self.activity.tools.running_tool_calls(),
            audio_output_enabled: self.session.outputs.plays_audio(),
            text_output_enabled: self.session.outputs.shows_transcription(),
            expressive,
        };

        // A reply created while a speech is being cut is requested once that speech's item is on
        // the record (the cut settles within the interruption watchdog). One created while a
        // speech plays is requested now, on the record as it stands, as the reference's reply
        // task copies the context at creation: the playing speech's item may be seconds away.
        let after = self.current_handle().filter(|h| !h.done() && h.interrupted()).map(|h| h.id());

        let (id, handle) = self.new_speech(spec.host_handle.as_ref(), spec.allow_interruptions);

        self.activity.note_speech_agent(id, kind);

        // A model reply runs as one `AgentRun`, which names its first step's message and adds
        // the tools its output takes.
        let realtime = matches!(spec.source, ReplySource::Realtime);
        let opened = if is_reply || realtime {
            let budgets = self.activity.retries();
            let invocation_id = handle.speech_id().to_owned();
            let kind = (realtime, self.activity.voice_tasks.running());
            self.open_reply_run(id, invocation_id, &turn_toolset, budgets, &plan, kind, out)
        } else {
            None
        };
        let message_id = match opened {
            Some((message_id, tools)) => {
                plan.tools = tools;
                message_id
            }
            None => self.session.ids.mint("item"),
        };
        let request = match &spec.source {
            ReplySource::Llm
            | ReplySource::Text(_)
            | ReplySource::TextStream(_)
            | ReplySource::Audio { .. } => plan.plan(&record, &instructions, wall),
            // No plan: the server decides what to say. The step needs the record and tools; there
            // is no TTS leg, but the words still go on the text stream.
            ReplySource::Realtime => crate::generation::Request {
                chat_ctx: record,
                tools: plan.tools.clone(),
                tts_enabled: false,
                text_output_enabled: self.session.outputs.shows_transcription(),
                instructions_hash: crate::generation::stable_hash(&instructions),
                expressive: None,
            },
        };

        let replan = match (after, &spec.source) {
            (Some(after), ReplySource::Llm) => {
                Some(crate::generation::Replan { after, spec: plan })
            }
            (
                Some(_),
                ReplySource::Text(_)
                | ReplySource::TextStream(_)
                | ReplySource::Audio { .. }
                | ReplySource::Realtime,
            )
            | (None, _) => None,
        };

        let ctx = ReplyContext {
            preemptive_tts: self.session.config.effective.turn.preemptive.enabled()
                && self.session.config.effective.turn.preemptive.preemptive_tts(),
            request,
            replan,
            source: spec.source,
            tool_choice: spec.tool_choice,
            frame: self.current,
            toolset: spec.toolset,
            providers: Providers {
                dynamic_instructions: self.activity.dynamic_instructions.clone(),
                llm: None,
                tts: None,
                toolset: advertised,
                tools_host_extra: self.session.tool_hosts.clone(),
                output_nodes: self.activity.output_nodes.clone(),
            },
            user_metrics,
            message_id,
            speech: self.session.config.effective.speech,
            answering: spec.answering,
            task: self.activity.voice_tasks.running(),
        };

        // Past the close's hard edge no generation is started.
        let past_edge = self.session.closing_past_edge();
        // Announced, recorded and scheduled in `request`, the one place every speech passes.
        let admitted =
            self.schedule_speech(now, &handle, kind, spec.add_to_chat_ctx, spec.host_asked, out);
        if admitted && !past_edge && !self.session.storage.lost {
            out.machine().push(crate::agent_session::effects::Effect::SpeechStart {
                handle: Arc::clone(&handle),
                context: Box::new(ctx),
                llm: self.session.llm.as_ref().map(|l| l.id),
                tts: self.session.tts.as_ref().map(|t| t.id),
            });
        }
        if (!admitted || self.session.storage.lost) && self.activity.speech_scheduler.contains(id) {
            self.release_speech(now, id, out);
        }
        if admitted && kind.scheduled() {
            handle.mark_scheduled();
        }
        if let Some(info) = &spec.end_of_turn {
            self.emit_eou_metrics(info, Some(id), out);
        }
        // The reference's reply task adds the user item once its speech is scheduled: after the
        // announcement, never for a refused speech. A preemptive generation plans against the
        // message but does not commit it; `generate_reply` commits on adoption.
        if let (Some(user), true) = (spec.new_message, admitted && kind.scheduled()) {
            let source = match spec.turn {
                Some(turn) => Source::UserTurn { turn },
                None => Source::UserInput { speech: id },
            };
            self.commit_item(ChatItem::Message(user), source, Visibility::Both, out);
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
            GenEvent::BindOutput { call, speech } => self.bind_output(call, speech, out),
            GenEvent::ActivateOutput { call, binding, speech } => {
                self.activate_output(call, binding, speech, out)
            }
            GenEvent::ReleaseOutput { binding, speech } => {
                self.release_output(binding, speech, out)
            }
            GenEvent::OutputUnsettled { binding, speech } => {
                self.output_unsettled(binding, speech, out)
            }
            GenEvent::Plan { call, plan } => self.activity.reply_plans.push_back((call, *plan)),
            // A text-only turn's first frame; with audio output the edge is `PlaybackStarted`'s.
            GenEvent::FirstFrame { speech } => self.agent_started_speaking(now, speech, out),

            GenEvent::Done { speech, outcome, interrupted } => {
                if self.session.storage.enabled {
                    self.storage_settle(speech, *outcome, interrupted, out);
                } else {
                    self.activity.set_tools_in_flight(speech, false);
                    self.finish_reply(now, speech, *outcome, interrupted, out);
                }
            }

            // One step's words are done, before its tools run; the reply's run reads them once
            // they are on the record.
            GenEvent::StepSpoken {
                speech,
                step,
                interrupted,
                has_tool_output,
                tools_running,
                item,
            } => {
                self.activity.set_tools_in_flight(speech, tools_running);
                let text = item.as_ref().and_then(|m| m.raw_text_content());
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
                self.reply_spoken(speech, text, out);
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

            GenEvent::ToolsFinished { speech, step, end, answer } => {
                self.on_tools_finished(now, (speech, step), *end, answer, out);
            }

            // The reply's tools are done and it has a next step: behind whatever was queued.
            GenEvent::ToolStep { speech, step } => {
                out.trace(ToolStep::StepStarted { speech, step });
                self.requeue_speech(now, speech, out);
            }

            GenEvent::Failed { speech, error } => {
                if self.session.storage.enabled {
                    self.storage_abandon(speech, error, out);
                } else {
                    self.generation_failed(now, speech, error, out);
                }
            }

            GenEvent::PlaybackStarted { speech } => {
                // The launch hold is over for this speech: its first frame is out (I-06, I-07).
                self.activity.launching.remove(&speech);
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
            GenEvent::ToolCallMade { speech, step, made, go } => {
                self.on_tool_made(now, (speech, step), *made, go, out);
            }
            GenEvent::ToolPrepared { speech, call, identity, go } => {
                self.on_tool_prepared(speech, call, identity, go, out);
            }
            GenEvent::ReadBackRendered { speech, call, answer, go } => {
                self.on_read_back(speech, call, answer, go, out);
            }
            GenEvent::ToolCallEnded { speech, call, call_id, end, message, outcome } => {
                self.on_tool_ended(now, (speech, call), call_id, end, message, outcome, out);
            }
            GenEvent::OutputValidated { speech, validation, verdict, value } => {
                self.on_output_validated(speech, validation, verdict, value, out);
            }
        }
    }

    /// Answers each waiting [`ReplyPlan`](crate::generation::ReplyPlan) whose record is ready, or
    /// with `None` once its speech is gone or the close passed its edge. `true` when one was.
    pub(crate) fn settle_reply_plans(&mut self, out: &mut Out) -> bool {
        let mut ready = false;
        let mut waiting = std::collections::VecDeque::new();
        while let Some((call, plan)) = self.activity.reply_plans.pop_front() {
            let cancelled = self.session.closing_past_edge()
                || self
                    .activity
                    .speech_scheduler
                    .handle(plan.speech)
                    .is_none_or(|handle| handle.interrupted() || handle.done());
            let committed = self.activity.finished.contains(&plan.after)
                || (self.activity.speech_scheduler.handle(plan.after))
                    .is_none_or(|handle| handle.done())
                || self.activity.tools_committed.contains(&plan.after);
            if !cancelled && !committed {
                waiting.push_back((call, plan));
                continue;
            }
            ready = true;
            let reply = if cancelled {
                None
            } else {
                let instructions = self.activity.instructions.render(Some(Modality::Audio));
                let record = self.session.chat_ctx.agent_chat_ctx();
                let request = plan.spec.plan(&record, &instructions, self.step.wall());
                Some(Box::new(crate::generation::PlannedReply {
                    request,
                    tool_choice: plan.tool_choice,
                }))
            };
            out.answer(Some(call), Answer::ReplyPlan(reply));
        }
        self.activity.reply_plans = waiting;
        ready
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

    pub(crate) fn generation_failed(
        &mut self,
        now: Ts,
        speech: SpeechId,
        error: Box<lk_agents_abi::errors::ProviderError>,
        out: &mut Out,
    ) {
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
        if let Some(handle) = self.activity.speech_scheduler.handle(speech) {
            let _ = handle.mark_generation_done();
        }
        out.trace(SpeechStep::StepSpoken { speech, step, interrupted });
        // The step's edge is spent: the next step's first frame is a new one.
        self.activity.spoke.remove(&speech);
        // Read off the record kept past the release: a `say` cut by the close's edge reports its
        // step after the edge released it, and is still a `say`'s item.
        let kind =
            self.activity.speech_kind(speech).or(self.activity.speech_scheduler.kind(speech));
        let mut item = item.map(|msg| {
            let source = match kind {
                Some(SpeechKind::Say) => Source::Say { speech },
                Some(
                    SpeechKind::TurnReply
                    | SpeechKind::GenerateReply
                    | SpeechKind::Preemptive
                    | SpeechKind::Realtime,
                )
                | None => Source::Reply { speech, step },
            };
            (ChatItem::Message(*msg), source)
        });
        // The reference's realtime generation moves the agent's state before it adds the
        // messages; the pipeline adds its message first.
        let realtime = kind == Some(SpeechKind::Realtime);
        if !realtime {
            self.commit_spoken(speech, item.take(), out);
        }
        self.activity.speech_scheduler.spoken(speech, now);
        // A blocked current speech's step ending is what a pause from inside its tool waits on.
        self.settle_scheduling_pause(out);
        // The state moves on the speech's interruption, not the step's; a filler ending inside
        // another speech's tool span goes back to `thinking` (listening would arm the away timer).
        let speech_interrupted =
            self.activity.speech_scheduler.handle(speech).is_some_and(|h| h.interrupted());
        let eos = end_of_speech(
            speech_interrupted,
            has_tool_output,
            tools_running || self.tools_running_elsewhere(speech),
            self.session.states.agent_state() == AgentState::Speaking,
            self.session.config.effective.turn.interruption.detection_enabled(),
        );
        self.apply_end_of_speech(now, eos, out);
        self.commit_spoken(speech, item, out);
    }

    /// A step's words: into the record, or only told to the handle under `add_to_chat_ctx=False`.
    fn commit_spoken(&mut self, speech: SpeechId, item: Option<(ChatItem, Source)>, out: &mut Out) {
        let Some((item, source)) = item else { return };
        if self.activity.speech_scheduler.add_to_chat_ctx(speech) {
            self.commit_item(item, source, Visibility::Both, out);
        } else if let Some(h) = self.activity.speech_scheduler.handle(speech) {
            h.item_added(&item);
        }
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
            self.session.config.effective.turn.interruption.detection_enabled(),
        );
        self.apply_end_of_speech(now, eos, out);

        self.release_speech(now, speech, out);
    }

    pub(crate) fn release_speech(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        self.activity.spoke.remove(&speech);
        self.activity.launching.remove(&speech);
        // `_scheduling_task`'s post-await release: a finished speech that is the paused one drops
        // the pause and resumes the sink, current or not (the floor left it at its last step).
        // The release at `Authorized` can miss a pause taken in the same instant, which would
        // leave the sink paused until the close.
        self.release_paused_speech(speech, out);
        self.reply_released(speech, out);
        self.activity.tools.dropped.retain(|(s, _), _| *s != speech);
        // A close ending a speech with blocking tools still running: they are answered with the
        // closing error before the speech's own `SpeechDone`; nothing is journalled after `close`.
        // Their bodies are cancelled, as the reference's close cancels its tool tasks
        // (`_ToolExecutor.aclose`): answered already, nothing would ever hear their end.
        if self.session.closing.is_some()
            && self.activity.speech_scheduler.contains(speech)
            && self.activity.tools.live(speech).next().is_some()
            && self.session.close_phase() != Some(ClosePhase::Closed)
        {
            self.end_live_tools(now, speech, out);
            if let Some(h) = self.activity.speech_scheduler.handle(speech) {
                out.machine()
                    .push(crate::agent_session::effects::Effect::CancelSpeech(Arc::clone(h)));
            }
        }
        if let Some(h) = self.activity.speech_scheduler.handle(speech) {
            // The host learns of the end only here, with the handle already `done()` (idempotent).
            if !h.done() {
                h.mark_done(None);
            }
            // A host `interrupt()` still on its way when the speech ended landed on the handle.
            if h.take_interrupted_at_done() {
                let cause = InterruptCause::Host;
                out.trace(InterruptStep::Interrupted { speech, cause, force: false });
            }
            out.machine().emit(SessionEvent::SpeechDone { speech, outcome: h.outcome() });
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
        self.voice_task_speech_released(now, speech, out);
        // A realtime session swapped meanwhile gets its frame once what it cut is released.
        self.realtime_replay(now, Some(speech), out);
    }

    /// `_update_agent_state(next)`: the one door for agent-state writes, reading the audio-output
    /// fact the warmup edge gates on at the instant of the edge.
    pub(crate) fn set_agent_state(
        &mut self,
        now: Ts,
        next: AgentState,
        out: &mut Out,
    ) -> Option<AgentState> {
        let audio_output_ready = self
            .activity
            .speech_scheduler
            .current_speech()
            .map(|speech| self.speech_outputs(speech))
            .unwrap_or(&self.session.outputs)
            .plays_audio();
        let (old, eff) = self.session.update_agent_state(now, next, audio_output_ready);
        out.machine().append(eff);
        old
    }
}

/// `Metadata(model_name="unknown", model_provider=<the turn_detection string>)`.
fn mode_metadata(mode: &str) -> lk_agents_abi::metrics::base::Metadata {
    lk_agents_abi::metrics::base::Metadata {
        model_name: Some("unknown".to_owned()),
        model_provider: Some(mode.to_owned()),
    }
}
