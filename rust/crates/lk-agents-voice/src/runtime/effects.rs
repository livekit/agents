//! Executes the decision core's effects after synchronous settlement.

use super::Runtime;
use crate::agent_activity::realtime::RealtimeWake;
use crate::agent_session::effects::{Effect, StreamKind};
use crate::agent_session::{HostEvent, Input, SystemTap};
use crate::audio_recognition::InterruptionSignal;
use crate::events::SessionEvent;
use crate::generation::{GenEvent, ReplySource};
use crate::room_io::{Opened, StreamOpened};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{HOST_TEARDOWN_TIMEOUT, SpawnerExt};
use std::sync::Arc;

impl Runtime {
    pub(super) fn prepare_leg(&mut self, opened: StreamOpened) -> Input {
        let StreamOpened { open, leg, outcome } = opened;
        let outcome = outcome.map(|stream| {
            let (sink, stream) = match stream {
                Opened::Stt(sink, source) => (Arc::from(sink), PendingLeg::Stt(source)),
                Opened::Vad(sink, source) => (Arc::from(sink), PendingLeg::Vad(source)),
            };
            let sink = self.shared.resources.register_audio_sink(sink);
            self.pending_legs.insert(sink, stream);
            sink
        });
        Input::StreamOpened { open, leg, outcome }
    }

    pub(crate) fn perform_voice(&mut self, effects: Vec<Effect>) {
        let frames = Arc::clone(&self.shared.frames);
        let _scope = frames.enter(self.driver.current);
        // Views first: a receipt or event performed in the same batch must read what the core
        // accepted, not the state before it.
        let mut operations = Vec::with_capacity(effects.len());
        for effect in effects {
            match effect {
                Effect::SessionView(_)
                | Effect::ChatContext(_)
                | Effect::Outputs(_)
                | Effect::LaunchHold(_) => self.perform_voice_effect(effect),
                effect => operations.push(effect),
            }
        }
        for effect in operations {
            self.perform_voice_effect(effect);
        }
    }

    fn deliver_session_event(&mut self, event: SessionEvent) {
        self.watch.publish(SystemTap::Session(event.clone()));
        let _ = self.events.send(HostEvent::Session(event));
    }

    #[allow(clippy::disallowed_methods, reason = "S0: runtime migration")]
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    fn perform_voice_effect(&mut self, effect: Effect) {
        use crate::agent_session::effects::{
            AudioOp, DetectorCall as D, Effect as E, GateOp, ProviderCall as P,
        };
        // Journalled already: `Out::absorb` recorded every effect when the machine produced it.
        match effect {
            E::RealtimePlayer { speech, event } => {
                if self.realtime_players.get(&speech).is_some_and(|tx| tx.send(*event).is_err()) {
                    self.realtime_players.remove(&speech);
                    let _ = self.reports.send(RealtimeWake::PlayerEnded(speech));
                }
            }
            E::CloseRealtimePlayer(speech) => {
                self.realtime_players.remove(&speech);
            }
            E::CloseRealtimePlayers => self.realtime_players.clear(),
            E::CancelSpeech(handle) => handle.cancel_tasks(),
            E::AttachRecorder { recorder, output } => {
                self.unobserve_recorder(output);
                let tap =
                    crate::recorder_io::RecorderTap::new(recorder.0, self.shared.time.clock());
                *self.shared.recorder.write().unwrap_or_else(|p| p.into_inner()) = Some(tap);
                self.observe_recorder(None, output);
            }
            E::DetachRecorder(output) => {
                self.unobserve_recorder(output);
                *self.shared.recorder.write().unwrap_or_else(|p| p.into_inner()) = None;
            }
            E::FlushRecorder(at) => {
                if let Some(tap) = self.shared.recorder_tap() {
                    let ok = tap.recorder.flush(at).is_ok();
                    self.shared.journal.record(crate::trace::PlayoutStep::RecorderFlushed { ok });
                }
            }
            E::PushKeyterms { stt, terms, warn_unsupported } => {
                if let Some(stt) = stt.and_then(|id| self.shared.resources.stt(id)) {
                    if warn_unsupported {
                        log::warn!(target: "livekit.agents", stt = stt.label().as_str();
                            "keyterms are not supported by this STT, ignoring keyterms update");
                    }
                    stt.update_keyterms(&terms);
                }
                self.shared.keyterms.send_replace(Some(terms));
            }
            E::RequestKeyterms { llm, req, request } => {
                let shared = Arc::clone(&self.shared);
                let Some(llm) = shared.resources.llm(llm) else {
                    shared.report(GenEvent::KeytermAnswer {
                        req,
                        arguments: None,
                        failed: true,
                        metrics: None,
                    });
                    return;
                };
                let task = crate::keyterm_detection::detection_pass(
                    Arc::clone(&shared),
                    llm,
                    *request,
                    req,
                );
                let handle = shared.spawner.spawn_unboxed(task);
                self.keyterms = Some((req, handle));
            }
            E::CancelKeyterms(req) => {
                if self.keyterms.as_ref().is_some_and(|(pending, _)| *pending == req) {
                    if let Some((_, task)) = self.keyterms.take() {
                        task.abort();
                    }
                }
            }
            E::ActivateLeg { channel, sink } => {
                let Some(stream) = self.pending_legs.remove(&sink) else { return };
                let shared = &self.shared;
                match stream {
                    PendingLeg::Stt(mut source) => {
                        source.report_retries(crate::audio::report_to(self.reports.clone()));
                        crate::audio::spawn_drain(
                            &shared.spawner,
                            shared.time.clone(),
                            shared.closing.subscribe(),
                            crate::audio::tagged(sink, source),
                            self.reports.clone(),
                            (channel, sink),
                        );
                    }
                    PendingLeg::Vad(source) => {
                        crate::audio::spawn_drain(
                            &shared.spawner,
                            shared.time.clone(),
                            shared.closing.subscribe(),
                            source,
                            self.reports.clone(),
                            (channel, sink),
                        );
                    }
                }
            }
            E::DiscardLeg(sink) => {
                self.shared.resources.release(sink);
                let Some(stream) = self.pending_legs.remove(&sink) else { return };
                let deadline = self.shared.time.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
                self.shared.spawner.spawn_unboxed(async move {
                    match stream {
                        PendingLeg::Stt(source) => {
                            let _ = source.close(deadline).await;
                        }
                        PendingLeg::Vad(source) => {
                            let _ = source.close(deadline).await;
                        }
                    }
                });
            }
            E::Journal(step) => self.shared.journal.record(*step),
            E::Answer(call, answer) => {
                self.shared.calls.settle(call, answer);
            }
            E::Host(event) => {
                if let Err(error) = self.events.send(event) {
                    if let HostEvent::Hook(call) = error.0 {
                        let _ = self.reports.send(crate::agent_session::Command::HookDone {
                            id: call.id(),
                            outcome: crate::agent_session::HookOutcome::Failed,
                        });
                    }
                }
            }
            E::ChatContext(snapshot) => {
                self.shared.chat_ctx.send_replace(snapshot);
            }
            E::Outputs(outputs) => {
                self.shared.output.send_replace(outputs);
            }
            E::Tap(tap) => self.watch.publish(tap),
            E::LaunchHold(hold) => self.shared.launch.publish(hold),
            E::SessionView(view) => {
                self.shared.view.send_replace(view);
            }
            E::Draining => {
                self.shared.draining.send_replace(true);
            }
            E::Closing => {
                self.shared.closing.send_replace(true);
            }
            E::SttConversationItem { stt, role, text } => {
                if let Some(stt) = self.shared.resources.stt(stt) {
                    if stt.capabilities().chat_context {
                        stt.push_conversation_item(&role, &text);
                    }
                }
            }
            E::SpeechStart { handle, mut context, llm, tts } => {
                // A realtime speech plays what the socket sends it, through its own player.
                let player = matches!(context.source, ReplySource::Realtime).then(|| {
                    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                    self.realtime_players.insert(handle.id(), tx);
                    tokio::sync::Mutex::new(rx)
                });
                context.providers.llm = llm.and_then(|id| self.shared.resources.llm(id));
                context.providers.tts = tts.and_then(|id| self.shared.resources.tts(id));
                let shared = Arc::clone(&self.shared);
                let task = Arc::clone(&handle);
                let abort = self.shared.spawner.spawn_unboxed(async move {
                    crate::generation::pipeline_reply::pipeline_reply_task(
                        shared, task, *context, player,
                    )
                    .await;
                });
                handle.add_task(abort);
            }
            // Not traced here: `Out::absorb` journalled it where it was decided.
            E::Emit(ev) => {
                self.deliver_session_event(ev);
            }
            E::FlushStream(sink) => {
                if let Some(sink) = self.shared.resources.audio_sink(sink) {
                    let _ = sink.flush();
                }
            }
            // A session without a turn detector asks for no flush.
            E::Detector(D::Flush(reason)) => {
                if let Some(d) = self.eot.as_ref() {
                    d.flush(reason.as_str());
                }
            }
            E::EotPredict { req, timeout } => {
                if let Some(leg) = self.eot.as_ref() {
                    self.shared.journal.record(crate::trace::InputStep::LegRequested {
                        leg: crate::trace::DetectorName::Eot,
                        req: req.0,
                    });
                    lk_agents_abi::turn::detector::DetectorLeg::signal(
                        &**leg,
                        Some(req),
                        lk_agents_abi::EotRequest { timeout },
                    );
                }
            }
            // `_generate_reply` is the one `timed_out=True` site. A session without a detector
            // asks for none.
            E::Detector(D::CancelEot { req: _, timed_out }) => {
                if let Some(d) = self.eot.as_ref() {
                    d.cancel(timed_out);
                }
            }
            // Not awaited, like the reference's `send_nowait` on `_interruption_ch`. A session
            // with no detector raises no signals.
            E::InterruptionSignal { overlap, signal } => {
                if let Some(d) = self.interruption.clone() {
                    match signal {
                        InterruptionSignal::OverlapStarted { .. } => {
                            let overlap =
                                overlap.unwrap_or(lk_agents_abi::turn::detector::ReqId(0));
                            d.signal_overlap(overlap, signal);
                        }
                        InterruptionSignal::AgentSpeechStarted
                        | InterruptionSignal::AgentSpeechEnded
                        | InterruptionSignal::OverlapEnded { .. } => d.signal(signal),
                    }
                }
            }
            // Journal-only: the learn is the machine's. The open runs off the loop (the provider
            // awaits a connection); the stream comes back as `Input::StreamOpened`.
            E::Provider(P::OpenLeg { leg, open, resource }) => {
                enum Pending {
                    Stt(Arc<dyn lk_agents_abi::Stt>),
                    Vad(Arc<dyn lk_agents_abi::Vad>),
                }
                let resources = &self.shared.resources;
                let pending = match leg {
                    StreamKind::Stt => resources.stt(resource).map(Pending::Stt),
                    StreamKind::Vad => resources.vad(resource).map(Pending::Vad),
                };
                let Some(pending) = pending else {
                    let error = ProviderError::Local(
                        ApiError::new("provider resource is no longer available")
                            .with_retryable(false),
                    );
                    let _ =
                        self.reports.send_opened(StreamOpened { open, leg, outcome: Err(error) });
                    return;
                };
                let tx = self.reports.clone();
                let time = self.shared.time.clone();
                self.shared.spawner.spawn_unboxed(async move {
                    let outcome = match pending {
                        Pending::Stt(stt) => match stt.stream(None, None).await {
                            Some(Ok((sink, source))) => Ok(Opened::Stt(sink, source)),
                            Some(Err(e)) => Err(e),
                            None => Err(ProviderError::Local(ApiError::new(
                                "the new agent's STT does not stream",
                            ))),
                        },
                        Pending::Vad(vad) => {
                            vad.stream().await.map(|(sink, source)| Opened::Vad(sink, source))
                        }
                    };
                    if let Err(StreamOpened { outcome: Ok(stream), .. }) =
                        tx.send_opened(StreamOpened { open, leg, outcome })
                    {
                        let deadline = time.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
                        stream.close(deadline).await;
                    }
                });
            }
            E::Provider(P::Prewarm(id)) => self.shared.resources.prewarm(id),
            E::Provider(P::Release(id)) => self.shared.resources.release(id),
            E::Provider(P::CloseRealtime(id)) => {
                let Some(session) = self.shared.resources.realtime(id) else { return };
                let tx = self.reports.clone();
                self.shared.spawner.spawn_unboxed(async move {
                    session.aclose().await;
                    let _ = tx.send(RealtimeWake::Closed);
                });
            }
            E::Detector(D::Close) => {
                if let Some(leg) = self.eot.as_ref() {
                    leg.close();
                }
            }
            E::Audio(AudioOp::Gate { speech: _, op, cause: _, output }) => {
                let Some(sink) = self.shared.resources.audio_output(output) else { return };
                match op {
                    GateOp::Pause => sink.pause(),
                    GateOp::Resume => sink.resume(),
                    // A bare clear: the speech's own playout, cut by the same interruption, closes
                    // the segment it pushed (`clear_sink`). A flush here would end a synchronised
                    // segment none of whose audio reached the sink as if it had been said.
                    GateOp::ClearBuffer => sink.clear_buffer(),
                }
            }
            // A synchroniser left behind would otherwise keep pacing into the old leaf. The same
            // sink set again (under a new id) is neither detached nor re-attached.
            E::Audio(AudioOp::SetAudioOutput { previous, next, enabled }) => {
                let resources = &self.shared.resources;
                let old = previous.and_then(|id| resources.audio_output(id));
                let new = next.and_then(|id| resources.audio_output(id));
                if !same_sink(&old, &new) {
                    if let Some(old) = &old {
                        old.on_detached();
                    }
                    if let Some(new) = &new {
                        if enabled { new.on_attached() } else { new.on_detached() }
                    }
                }
                self.observe_recorder(old.as_deref(), next);
            }
            E::Audio(AudioOp::SetTranscriptionOutput { previous, next, enabled }) => {
                let resources = &self.shared.resources;
                let old = previous.and_then(|id| resources.text_output(id));
                let new = next.and_then(|id| resources.text_output(id));
                if !same_sink(&old, &new) {
                    if let Some(old) = &old {
                        old.on_detached();
                    }
                    if let Some(new) = &new {
                        if enabled { new.on_attached() } else { new.on_detached() }
                    }
                }
            }

            E::Audio(AudioOp::SetOutputEnabled { output, sink, enabled }) => {
                let resources = &self.shared.resources;
                match output {
                    crate::trace::OutputKind::Audio => {
                        if let Some(sink) = sink.and_then(|id| resources.audio_output(id)) {
                            if enabled { sink.on_attached() } else { sink.on_detached() }
                        }
                    }
                    crate::trace::OutputKind::Transcription => {
                        if let Some(sink) = sink.and_then(|id| resources.text_output(id)) {
                            if enabled { sink.on_attached() } else { sink.on_detached() }
                        }
                    }
                }
            }
            E::Forwarder(command) => {
                let _ = self.reports.send(command);
            }
            E::Audio(AudioOp::SetAudioInput(input)) => {
                let source = input.map(|input| self.shared.resources.register_audio_input(input.0));
                let _ = self.reports.send(crate::data_plane::ForwarderCommand::SetSource(source));
            }
            E::Audio(AudioOp::SetVideoInput(input)) => {
                if let Some(old) = self.video_forwarder.take() {
                    old.abort();
                }
                if let Some(old) = self.video_in.take() {
                    old.on_detached();
                }
                self.video_in = input.map(|input| input.0);
                if let Some(input) = &self.video_in {
                    input.on_attached();
                    self.video_forwarder = Some(
                        self.shared
                            .spawner
                            .spawn_unboxed(self.video_parts.clone().pump(Arc::clone(input))),
                    );
                }
            }
            E::Realtime { session, command } => {
                if let Some(session) = self.shared.resources.realtime(session) {
                    session.send(command);
                }
            }
            E::CancelTool { call_id, runner } => runner.cancel(&call_id),
            // The core's own timers, and the requests its step completes with the sink or overlap
            // they name (`Driver::settle_step`), never leave the step.
            E::Arm(..)
            | E::Cancel(..)
            | E::Provider(P::FlushStt | P::FlushVad)
            | E::Detector(D::RequestEot { .. } | D::Signal(_)) => {
                unreachable!("settled inside the core's step")
            }
        }
    }
}

fn same_sink<T: ?Sized>(a: &Option<Arc<T>>, b: &Option<Arc<T>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b)),
        (None, None) => true,
        _ => false,
    }
}

/// An opened stream's source half, held while the core decides: its sink is already registered,
/// so the core names it by id. Read once activated, closed once discarded.
pub(super) enum PendingLeg {
    Stt(lk_agents_abi::SttSource),
    Vad(lk_agents_abi::VadSource),
}
