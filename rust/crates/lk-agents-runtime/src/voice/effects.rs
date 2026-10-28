//! Executes the decision core's effects after synchronous settlement.

use super::Runtime;
use super::legs::PendingLeg;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{HOST_TEARDOWN_TIMEOUT, SpawnerExt};
use lk_agents_voice::agent_activity::realtime::RealtimeWake;
use lk_agents_voice::agent_session::effects::{Effect, StreamKind};
use lk_agents_voice::agent_session::{HostEvent, Input, SystemTap};
use lk_agents_voice::audio_recognition::InterruptionSignal;
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::generation::{GenEvent, ReplySource};
use lk_agents_voice::room_io::{Opened, StreamOpened};
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
        let _scope = frames.enter(self.driver.frame());
        // Views first: a receipt or event performed in the same batch must read what the core
        // accepted, not the state before it.
        let mut operations = Vec::with_capacity(effects.len());
        for effect in effects {
            match effect {
                Effect::SessionView(_) | Effect::ChatContext(_) | Effect::Outputs(_) => {
                    self.perform_voice_effect(effect)
                }
                effect => operations.push(effect),
            }
        }
        for effect in operations {
            self.perform_voice_effect(effect);
        }
    }

    fn deliver_session_event(&mut self, event: SessionEvent) {
        self.peer.publish(SystemTap::Session(event.clone()));
        let _ = self.peer.deliver(HostEvent::Session(event));
    }

    fn perform_voice_effect(&mut self, effect: Effect) {
        use lk_agents_voice::agent_session::effects::{AudioOp, Effect as E, GateOp};
        // Journalled already: `Out::absorb` recorded every effect when the machine produced it.
        match effect {
            E::RealtimePlayer { speech, event } => {
                let Some(player) = self.realtime_players.get(&speech) else {
                    return;
                };
                if let Err(error) = player.try_send(*event) {
                    if matches!(error, tokio::sync::mpsc::error::TrySendError::Full(_)) {
                        log::error!(target: "livekit.agents", speech = speech.0,
                            capacity = lk_agents_voice::generation::realtime_generation::PLAYER_CAPACITY;
                            "a realtime speech fell too far behind its reply; it ends on what it holds");
                    }
                    self.realtime_players.remove(&speech);
                    let _ = self.reports.send(RealtimeWake::PlayerEnded(speech));
                }
            }
            E::CloseRealtimePlayer(speech) => {
                self.realtime_players.remove(&speech);
            }
            E::CloseRealtimePlayers => self.realtime_players.clear(),
            E::CancelSpeech(handle) => handle.cancel_tasks(),
            E::AttachRecorder { recorder, output: _ } => {
                self.recorder_observers.detach(&self.shared, None);
                let tap = lk_agents_voice::recorder_io::RecorderTap::new(
                    recorder.0,
                    self.shared.time.clock(),
                );
                *self.shared.recorder.write().unwrap_or_else(|p| p.into_inner()) = Some(tap);
                for output in self.driver.active_audio_outputs() {
                    self.recorder_observers.attach(&self.shared, output);
                }
            }
            E::DetachRecorder(_) => {
                self.recorder_observers.detach(&self.shared, None);
                *self.shared.recorder.write().unwrap_or_else(|p| p.into_inner()) = None;
            }
            E::FlushRecorder(at) => {
                if let Some(tap) = self.shared.recorder_tap() {
                    let ok = tap.recorder.flush(at).is_ok();
                    self.shared
                        .journal
                        .record(lk_agents_voice::trace::PlayoutStep::RecorderFlushed { ok });
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
                let task = lk_agents_voice::keyterm_detection::detection_pass(
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
                if let Some(stream) = self.pending_legs.remove(&sink) {
                    self.drain(channel, sink, stream);
                }
            }
            E::DiscardLeg(sink) => {
                self.shared.resources.release(sink);
                let Some(stream) = self.pending_legs.remove(&sink) else {
                    return;
                };
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
            // A late or unknown call's answer is dropped.
            E::Answer(call, answer) => {
                if let Some(reply_to) = self.calls.remove(&call) {
                    reply_to.settle(answer);
                }
            }
            E::Host(event) => {
                if let Err(event) = self.peer.deliver(event) {
                    if let HostEvent::Hook(call) = event {
                        let _ =
                            self.reports.send(lk_agents_voice::agent_session::Command::HookDone {
                                id: call.id(),
                                outcome: lk_agents_voice::agent_session::HookOutcome::Failed,
                            });
                    }
                }
            }
            E::StorageAbort(speech) => {
                self.storage_command(lk_agents_abi::storage::voice::VoiceStorageCommand::Abort {
                    speech: speech.0,
                })
            }
            E::StorageCommit { speech, history } => {
                self.storage_command(lk_agents_abi::storage::voice::VoiceStorageCommand::Settle {
                    speech: speech.0,
                    history,
                })
            }
            E::ChatContext(snapshot) => {
                self.shared.chat_ctx.send_replace(snapshot);
            }
            E::Outputs(outputs) => {
                self.shared.output.send_replace(outputs);
            }
            E::Tap(tap) => self.peer.publish(tap),
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
                    let capacity =
                        lk_agents_voice::generation::realtime_generation::PLAYER_CAPACITY;
                    let (tx, rx) = tokio::sync::mpsc::channel(capacity);
                    self.realtime_players.insert(handle.id(), tx);
                    rx
                });
                context.providers.llm = llm.and_then(|id| self.shared.resources.llm(id));
                context.providers.tts = tts.and_then(|id| self.shared.resources.tts(id));
                let shared = Arc::clone(&self.shared);
                let task = Arc::clone(&handle);
                let storage = self.storage.clone();
                let queue = self.reports.clone();
                let abort = self.shared.spawner.spawn_unboxed(async move {
                    if let Some(storage) = storage {
                        match storage.acquire(task.id().0, context.request.chat_ctx.items.clone()).await {
                            Ok(items) => {
                                context.request.chat_ctx.items = items.clone();
                                let _ = queue.send(Input::StorageReloaded(items));
                                let _ = storage.clone().dispatch(
                                    lk_agents_abi::storage::voice::VoiceStorageCommand::ObserveLoss { speech: task.id().0 },
                                    Arc::new(super::storage::StorageReporter(queue.clone())),
                                );
                            }
                            Err(error) => {
                                log::warn!(target: "livekit.agents", "the voice lane could not be acquired: {error}");
                                if storage.lease_lost() {
                                    let _ = queue.send(Input::StorageLost(task.id()));
                                    return;
                                }
                            }
                        }
                    }
                    lk_agents_voice::generation::pipeline_reply::pipeline_reply_task(
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
            E::EotFlush(reason) => {
                if let Some(d) = self.eot.as_ref() {
                    d.flush(reason.as_str());
                }
            }
            E::EotPredict { req, timeout } => {
                if self.eot.is_some() {
                    self.shared.journal.record(lk_agents_voice::trace::InputStep::LegRequested {
                        leg: lk_agents_voice::trace::DetectorName::Eot,
                        req: req.0,
                    });
                    self.predict(req, timeout);
                }
            }
            // `_generate_reply` is the one `timed_out=True` site. A session without a detector
            // asks for none.
            E::EotCancel { req: _, timed_out } => {
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
            // The open runs off the loop (the provider awaits a connection); the stream comes back
            // as `Input::StreamOpened`.
            E::SttStreamOpen { open, stt } => {
                let stt = self.shared.resources.stt(stt);
                self.open_stream(open, StreamKind::Stt, stt.map(PendingOpen::Stt));
            }
            E::VadStreamOpen { open, vad } => {
                let vad = self.shared.resources.vad(vad);
                self.open_stream(open, StreamKind::Vad, vad.map(PendingOpen::Vad));
            }
            E::ProviderPrewarm(id) => self.shared.resources.prewarm(id),
            E::ProviderRelease(id) => self.shared.resources.release(id),
            E::RealtimeClose(id) => {
                let Some(session) = self.shared.resources.realtime(id) else {
                    return;
                };
                let tx = self.reports.clone();
                self.shared.spawner.spawn_unboxed(async move {
                    session.aclose().await;
                    let _ = tx.send(RealtimeWake::Closed);
                });
            }
            E::EotClose => {
                for task in self.predictions.drain(..) {
                    task.abort();
                }
            }
            E::Audio(AudioOp::Gate { speech: _, op, cause: _, output }) => {
                let Some(sink) = self.shared.resources.audio_output(output) else {
                    return;
                };
                match op {
                    GateOp::Pause => sink.pause(),
                    GateOp::Resume => sink.resume(),
                    // A bare clear: the speech's own playout, cut by the same interruption, closes
                    // the segment it pushed (`clear_sink`). A flush here would end a synchronised
                    // segment none of whose audio reached the sink as if it had been said.
                    GateOp::ClearBuffer => sink.clear_buffer(),
                }
            }
            E::Audio(AudioOp::SetAudioOutput { previous, next, enabled }) => {
                let resources = &self.shared.resources;
                if let Some(id) = previous {
                    self.recorder_observers.detach(&self.shared, Some(id));
                    if let Some(old) = resources.audio_output(id) {
                        old.on_detached();
                    }
                }
                if let Some(id) = next {
                    if let Some(new) = resources.audio_output(id) {
                        if enabled { new.on_attached() } else { new.on_detached() }
                    }
                    self.recorder_observers.attach(&self.shared, id);
                }
            }
            E::Audio(AudioOp::SetTranscriptionOutput { previous, next, enabled }) => {
                let resources = &self.shared.resources;
                if let Some(old) = previous.and_then(|id| resources.text_output(id)) {
                    old.on_detached();
                }
                if let Some(new) = next.and_then(|id| resources.text_output(id)) {
                    if enabled { new.on_attached() } else { new.on_detached() }
                }
            }

            E::Audio(AudioOp::SetOutputEnabled { output, sink, enabled }) => {
                let resources = &self.shared.resources;
                match output {
                    lk_agents_voice::trace::OutputKind::Audio => {
                        if let Some(sink) = sink.and_then(|id| resources.audio_output(id)) {
                            if enabled { sink.on_attached() } else { sink.on_detached() }
                        }
                    }
                    lk_agents_voice::trace::OutputKind::Transcription => {
                        if let Some(sink) = sink.and_then(|id| resources.text_output(id)) {
                            if enabled { sink.on_attached() } else { sink.on_detached() }
                        }
                    }
                }
            }
            E::Forwarder(command) => self.command(command),
            E::Audio(AudioOp::SetAudioInput(input)) => {
                let source = input.map(|input| self.shared.resources.register_audio_input(input.0));
                let _ = self
                    .reports
                    .send(lk_agents_voice::data_plane::ForwarderCommand::SetSource(source));
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
        }
    }

    /// Opens `provider`'s stream off the loop and reports it, or that the provider is gone, as
    /// `Input::StreamOpened`.
    fn open_stream(&mut self, open: u64, leg: StreamKind, provider: Option<PendingOpen>) {
        let Some(pending) = provider else {
            let error = ProviderError::Local(
                ApiError::new("provider resource is no longer available").with_retryable(false),
            );
            let _ = self.reports.send_opened(StreamOpened { open, leg, outcome: Err(error) });
            return;
        };
        let tx = self.reports.clone();
        let time = self.shared.time.clone();
        self.shared.spawner.spawn_unboxed(async move {
            let outcome = match pending {
                PendingOpen::Stt(stt) => match stt.stream(None, None).await {
                    Ok(Some((sink, source))) => Ok(Opened::Stt(sink, source)),
                    Err(e) => Err(e),
                    Ok(None) => Err(ProviderError::Local(ApiError::new(
                        "the new agent's STT does not stream",
                    ))),
                },
                PendingOpen::Vad(vad) => {
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
}

/// A provider whose stream is being opened.
enum PendingOpen {
    Stt(Arc<dyn lk_agents_abi::Stt>),
    Vad(Arc<dyn lk_agents_abi::Vad>),
}
