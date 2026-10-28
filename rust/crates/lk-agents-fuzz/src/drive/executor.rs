//! The run's own executor of the session's cores: the decision core ([`Driver`]) and the data
//! plane ([`AudioForwarder`]), stepped by the run's queue, and every effect they return performed
//! against the fakes the way the production runtime performs it against real providers.
//!
//! The simulation never runs the production runtime (owner decision 28): this is its fake. What
//! it must keep from the runtime is what the cores can observe: effects performed in the order
//! the step returned them with the views first, an opened stream's sink registered before the core
//! hears of it, the microphone's frames on a channel of `AUDIO_QUEUE_FRAMES`, and every
//! report back through the session's one event queue. The match over `Effect` has no wildcard, so
//! a new effect fails to compile here as it does in the runtime.

use std::collections::BTreeMap;
use std::sync::Arc;

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::{AudioInput, VideoInput};
use lk_agents_abi::{
    AUDIO_QUEUE_FRAMES, AbortHandle, AudioFrame, EotDetector, HOST_TEARDOWN_TIMEOUT, LegEvent,
    SpawnerExt, Stt, Vad,
};
use lk_agents_voice::agent_activity::false_interruption::InterruptionPort;
use lk_agents_voice::agent_activity::realtime::RealtimeWake;
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::effects::{AudioOp, Effect, GateOp, StreamKind};
use lk_agents_voice::agent_session::{
    Command, Driver, DriverError, Event, HookOutcome, HostEvent, Input, Now, OpeningLegs,
    PeerHandle, QueueOwner, ReplyTo, RequestId, SessionCores, SessionExecutor, SessionShared,
    SystemTap, Tx,
};
use lk_agents_voice::audio::{AUDIO_IN_LEG, AudioChannel, ChannelEnd, VideoForwarderParts};
use lk_agents_voice::audio_recognition::InterruptionSignal;
use lk_agents_voice::data_plane::{AudioForwarder, ForwarderCommand, ForwarderEffect};
use lk_agents_voice::generation::{GenEvent, ReplySource};
use lk_agents_voice::room_io::{Opened, StreamOpened};
use lk_agents_voice::trace::{ChannelEndCause, OutputKind};
use tokio::sync::{mpsc, oneshot, watch};

pub(crate) struct Executor {
    pub(crate) driver: Driver,
    shared: Arc<SessionShared>,
    /// Where host events and taps go.
    peer: PeerHandle,
    forwarder: AudioForwarder,
    /// Every source's events, in arrival order. Frames never come this way.
    pub(crate) queue: mpsc::UnboundedReceiver<Event>,
    /// The microphone's frames, bounded as the runtime bounds them: a full queue holds the reader
    /// back, so the loss falls to the input's own drop-oldest queue.
    pub(crate) frames: mpsc::Receiver<AudioFrame>,
    reports: Tx,
    frames_tx: mpsc::Sender<AudioFrame>,
    /// The microphone, its id and its reader.
    source: Option<(ResourceId, AbortHandle)>,
    /// Where each awaited call's answer goes, until the core answers it.
    calls: BTreeMap<RequestId, ReplyTo>,
    /// The keyterm detection pass in flight, by request.
    keyterms: Option<(u64, AbortHandle)>,
    recorder_observers: super::recorder::RecorderOutputs,
    /// Each realtime speech's player, fed the socket's events.
    realtime_players: BTreeMap<SpeechId, mpsc::Sender<lk_agents_abi::llm::realtime::Event>>,
    /// The camera read now, its forwarder, and what a new forwarder is built from.
    video_parts: VideoForwarderParts,
    video_forwarder: Option<AbortHandle>,
    video_in: Option<Arc<dyn VideoInput>>,
    /// Opened STT and VAD streams the core has neither activated nor discarded yet.
    pending_legs: BTreeMap<ResourceId, PendingLeg>,
    eot: Option<Arc<dyn EotDetector>>,
    /// The turn detector's predictions in flight, aborted at its close.
    predictions: Vec<AbortHandle>,
    interruption: Option<Arc<dyn InterruptionPort>>,
}

/// The session's executor: it opens the opening streams as the runtime does, then hands the
/// cores, opened by the session, to the run, which steps them from then on.
pub(crate) struct Handover {
    executor: Option<Executor>,
    run: Option<oneshot::Sender<Executor>>,
    /// The opening streams, read once the cores exist.
    opening: Vec<(AudioChannel, ResourceId, PendingLeg)>,
}

impl Handover {
    pub(crate) fn new(run: oneshot::Sender<Executor>) -> Self {
        Handover { executor: None, run: Some(run), opening: Vec::new() }
    }
}

impl SessionExecutor for Handover {
    fn stt_adapter(&self) -> lk_agents_voice::agent_session::SttAdapter {
        Arc::new(lk_agents_inference::stt::stream_adapter::StreamAdapter::adapted)
    }

    fn tts_adapter(&self) -> lk_agents_voice::agent_session::TtsAdapter {
        Arc::new(
            lk_agents_inference::tts::sentence_adapter::StreamAdapter::adapted_for::<
                lk_agents_voice::tts::processing::SpeechText,
            >,
        )
    }

    fn open_legs<'a>(
        &'a mut self,
        resources: &'a lk_agents_abi::resources::Resources,
        stt: Option<Arc<dyn Stt>>,
        vad: Option<Arc<dyn Vad>>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<OpeningLegs, DriverError>> {
        Box::pin(async move {
            let mut legs = OpeningLegs::default();
            if let Some(stt) = stt {
                let (sink, source) =
                    stt.stream(None, None).await?.ok_or(DriverError::Missing("a streaming STT"))?;
                let sink = resources.register_audio_sink(Arc::from(sink));
                self.opening.push((AudioChannel::Stt, sink, PendingLeg::Stt(source)));
                legs.stt = Some(sink);
            }
            if let Some(vad) = vad {
                let (sink, source) = vad.stream().await?;
                let sink = resources.register_audio_sink(Arc::from(sink));
                self.opening.push((AudioChannel::Vad, sink, PendingLeg::Vad(source)));
                legs.vad = Some(sink);
            }
            Ok(legs)
        })
    }

    fn own(mut self: Box<Self>, cores: SessionCores) -> Box<dyn QueueOwner> {
        let opening = std::mem::take(&mut self.opening);
        self.executor = Some(Executor::new(cores, opening));
        self
    }
}

impl QueueOwner for Handover {
    fn open(&mut self, now: Now) {
        if let Some(executor) = &mut self.executor {
            executor.open(now);
        }
    }

    fn run(mut self: Box<Self>) -> Option<lk_agents_abi::BoxFuture<'static, ()>> {
        if let (Some(executor), Some(run)) = (self.executor.take(), self.run.take()) {
            let _ = run.send(executor);
        }
        None
    }
}

impl Executor {
    fn new(cores: SessionCores, opening: Vec<(AudioChannel, ResourceId, PendingLeg)>) -> Self {
        let SessionCores {
            driver,
            forwarder,
            queue: (reports, queue),
            source,
            eot,
            interruption,
            room_events: (room, room_io),
            video: (video_parts, video_in),
            shared,
            peer,
            storage,
        } = cores;
        // The thread's storage is the storage survey's subject: a voice run has none.
        assert!(storage.is_none(), "the voice simulation runs without thread storage");
        let (frames_tx, frames) = mpsc::channel(AUDIO_QUEUE_FRAMES);
        let mut executor = Executor {
            driver,
            shared,
            peer,
            forwarder,
            queue,
            frames,
            reports,
            frames_tx,
            source: None,
            calls: BTreeMap::new(),
            keyterms: None,
            recorder_observers: Default::default(),
            realtime_players: BTreeMap::new(),
            video_parts,
            video_forwarder: None,
            video_in,
            pending_legs: BTreeMap::new(),
            eot,
            predictions: Vec::new(),
            interruption,
        };
        // The legs the session opens with, as the runtime starts them.
        if let Some(room_io) = room_io {
            let io_reports = executor.reports.clone();
            let rtc_reports = executor.reports.clone();
            room_io.forward_events(
                Arc::new(move |event| io_reports.send(event).is_ok()),
                Arc::new(move |event| rtc_reports.send(event).is_ok()),
            );
        }
        if let Some(rx) = room {
            executor.forward(rx);
        }
        for (channel, sink, stream) in opening {
            executor.drain(channel, sink, stream);
        }
        if let Some(detector) = executor.interruption.clone() {
            executor.pump_verdicts(detector);
        }
        executor.video_forwarder = executor.video_in.clone().map(|input| {
            executor.shared.spawner.spawn_unboxed(executor.video_parts.clone().pump(input))
        });
        // The session's first microphone is the host's attached one: nothing to tell it.
        if let Some(source) = source {
            executor.read(source);
        }
        let effects = executor.driver.take_outbox();
        executor.perform_voice(effects);
        executor
    }

    fn open(&mut self, now: Now) {
        let effects = self.driver.open(now);
        self.perform_voice(effects);
    }

    /// One event: an input or a wake to the decision core, a command to the data plane.
    pub(crate) fn handle(&mut self, event: Event, now: Now) {
        match event {
            Event::Input(mut input) => {
                if let Input::Stt(_, provider) = &mut input {
                    *provider = self
                        .driver
                        .stt()
                        .and_then(|stt| self.shared.resources.stt(stt))
                        .map(|stt| stt.id());
                }
                if let Input::Eot(_, _, thresholds) = &mut input {
                    *thresholds = self.eot.as_ref().and_then(|detector| detector.thresholds());
                }
                let effects = self.driver.handle(Some(input), now);
                self.perform_voice(effects);
            }
            Event::Awaited(input, call) => {
                self.calls.insert(call.id, call.reply_to);
                self.handle(Event::Input(input), now);
            }
            Event::Wake => {
                let effects = self.driver.handle(None, now);
                self.perform_voice(effects);
            }
            Event::LegOpened(opened) => {
                let input = self.prepare_leg(opened);
                let effects = self.driver.handle(Some(input), now);
                self.perform_voice(effects);
            }
            Event::Forwarder(command) => self.command(command),
        }
    }

    pub(crate) fn on_frame(&mut self, frame: AudioFrame, now: Now) {
        // The session recorder hears every frame the forwarder does, muted ones excepted.
        if self.forwarder.input_enabled()
            && let Some(tap) = self.shared.recorder_tap()
        {
            tap.recorder.on_input_frame(tap.now(), &frame);
        }
        let effects = self.forwarder.on_frame(frame, now.at);
        self.perform(effects);
    }

    /// The decision core stopped. The data plane's commands still queued are applied, so every
    /// replaced sink is ended; the decision core refuses the rest.
    pub(crate) fn finish(&mut self, undelivered: Vec<Event>) {
        let mut inputs = Vec::new();
        for event in undelivered {
            match event {
                Event::Input(input) => inputs.push(input),
                // Kept, so the core's refusal reaches it.
                Event::Awaited(input, call) => {
                    self.calls.insert(call.id, call.reply_to);
                    inputs.push(input);
                }
                Event::LegOpened(opened) => inputs.push(self.prepare_leg(opened)),
                // A microphone attached too late to be read is let go.
                Event::Forwarder(ForwarderCommand::SetSource(Some(source))) => {
                    self.shared.resources.release(source);
                }
                Event::Forwarder(command) => {
                    for effect in self.forwarder.handle(command) {
                        match effect {
                            ForwarderEffect::Report(input) => inputs.push(input),
                            effect => self.perform(vec![effect]),
                        }
                    }
                }
                Event::Wake => {}
            }
        }
        let effects = self.driver.finish(inputs);
        self.perform_voice(effects);
        self.release();
    }

    /// Every sink the data plane still feeds has its input ended, and the microphone is let go.
    fn release(&mut self) {
        let effects = self.forwarder.close();
        self.perform(effects);
        if let Some((source, _)) = self.source.take() {
            self.shared.resources.release(source);
        }
    }

    fn command(&mut self, command: ForwarderCommand) {
        match command {
            ForwarderCommand::SetSource(input) => self.set_source(input),
            ForwarderCommand::SetInputEnabled(enabled) => {
                let resources = &self.shared.resources;
                if let Some(input) =
                    self.source.as_ref().and_then(|(id, _)| resources.audio_input(*id))
                {
                    if enabled { input.on_attached() } else { input.on_detached() }
                }
            }
            ForwarderCommand::Install { .. }
            | ForwarderCommand::Remove(_)
            | ForwarderCommand::SetGateDiscard(_)
            | ForwarderCommand::FlushSttWithSilence { .. }
            | ForwarderCommand::ReportCensus => {}
        }
        let effects = self.forwarder.handle(command);
        self.perform(effects);
    }

    /// The old reader is aborted first; the old source is detached, the new one attached if input
    /// is enabled.
    fn set_source(&mut self, input: Option<ResourceId>) {
        let resources = self.shared.resources.clone();
        if let Some((old, reader)) = self.source.take() {
            reader.abort();
            if let Some(old) = resources.audio_input(old) {
                old.on_detached();
            }
            resources.release(old);
        }
        let Some(source) = input else { return };
        if self.forwarder.input_enabled()
            && let Some(input) = resources.audio_input(source)
        {
            input.on_attached();
        }
        self.read(source);
    }

    fn read(&mut self, source: ResourceId) {
        let Some(input) = self.shared.resources.audio_input(source) else {
            return;
        };
        let reader = read_audio(
            input,
            self.frames_tx.clone(),
            self.shared.closing.subscribe(),
            self.reports.clone(),
        );
        self.source = Some((source, self.shared.spawner.spawn_unboxed(reader)));
    }

    fn perform(&mut self, effects: Vec<ForwarderEffect>) {
        let resources = &self.shared.resources;
        for effect in effects {
            match effect {
                // A refused frame is the runtime's to log, and nothing the cores hear of.
                ForwarderEffect::Push { sink, frame } => {
                    if let Some(stream) = resources.audio_sink(sink) {
                        let _ = stream.push(frame);
                    }
                }
                ForwarderEffect::EndInput(sink) => {
                    if let Some(stream) = resources.audio_sink(sink) {
                        stream.end_input();
                    }
                    resources.release(sink);
                }
                ForwarderEffect::Report(input) => {
                    let _ = self.reports.send(input);
                }
            }
        }
    }

    fn prepare_leg(&mut self, opened: StreamOpened) -> Input {
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

    /// Views first: a receipt or event performed in the same batch reads what the core accepted.
    fn perform_voice(&mut self, effects: Vec<Effect>) {
        let frames = Arc::clone(&self.shared.frames);
        let _scope = frames.enter(self.driver.frame());
        let (views, operations): (Vec<_>, Vec<_>) = effects.into_iter().partition(|effect| {
            matches!(
                effect,
                Effect::SessionView(_)
                    | Effect::ChatContext(_)
                    | Effect::Outputs(_)
                    | Effect::LaunchHold(_)
            )
        });
        for effect in views.into_iter().chain(operations) {
            self.perform_voice_effect(effect);
        }
    }

    fn perform_voice_effect(&mut self, effect: Effect) {
        let shared = Arc::clone(&self.shared);
        let resources = &shared.resources;
        match effect {
            Effect::RealtimePlayer { speech, event } => {
                let Some(player) = self.realtime_players.get(&speech) else {
                    return;
                };
                if player.try_send(*event).is_err() {
                    self.realtime_players.remove(&speech);
                    let _ = self.reports.send(RealtimeWake::PlayerEnded(speech));
                }
            }
            Effect::CloseRealtimePlayer(speech) => {
                self.realtime_players.remove(&speech);
            }
            Effect::CloseRealtimePlayers => self.realtime_players.clear(),
            Effect::CancelSpeech(handle) => handle.cancel_tasks(),
            Effect::AttachRecorder { recorder, output: _ } => {
                self.recorder_observers.detach(&shared, None);
                let tap =
                    lk_agents_voice::recorder_io::RecorderTap::new(recorder.0, shared.time.clock());
                *shared.recorder.write().unwrap_or_else(|p| p.into_inner()) = Some(tap);
                for output in self.driver.active_audio_outputs() {
                    self.recorder_observers.attach(&shared, output);
                }
            }
            Effect::DetachRecorder(_) => {
                self.recorder_observers.detach(&shared, None);
                *shared.recorder.write().unwrap_or_else(|p| p.into_inner()) = None;
            }
            Effect::FlushRecorder(at) => {
                if let Some(tap) = shared.recorder_tap() {
                    let ok = tap.recorder.flush(at).is_ok();
                    shared
                        .journal
                        .record(lk_agents_voice::trace::PlayoutStep::RecorderFlushed { ok });
                }
            }
            Effect::PushKeyterms { stt, terms, warn_unsupported: _ } => {
                if let Some(stt) = stt.and_then(|id| resources.stt(id)) {
                    stt.update_keyterms(&terms);
                }
                shared.keyterms.send_replace(Some(terms));
            }
            Effect::RequestKeyterms { llm, req, request } => {
                let Some(llm) = resources.llm(llm) else {
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
                self.keyterms = Some((req, shared.spawner.spawn_unboxed(task)));
            }
            Effect::CancelKeyterms(req) => {
                if self.keyterms.as_ref().is_some_and(|(pending, _)| *pending == req)
                    && let Some((_, task)) = self.keyterms.take()
                {
                    task.abort();
                }
            }
            Effect::ActivateLeg { channel, sink } => {
                if let Some(stream) = self.pending_legs.remove(&sink) {
                    self.drain(channel, sink, stream);
                }
            }
            Effect::DiscardLeg(sink) => {
                resources.release(sink);
                let Some(stream) = self.pending_legs.remove(&sink) else {
                    return;
                };
                let deadline = shared.time.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
                shared.spawner.spawn_unboxed(async move {
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
            Effect::Journal(step) => shared.journal.record(*step),
            // A late or unknown call's answer is dropped.
            Effect::Answer(call, answer) => {
                if let Some(reply_to) = self.calls.remove(&call) {
                    reply_to.settle(answer);
                }
            }
            Effect::Host(event) => {
                if let Err(HostEvent::Hook(call)) = self.peer.deliver(event) {
                    let _ = self
                        .reports
                        .send(Command::HookDone { id: call.id(), outcome: HookOutcome::Failed });
                }
            }
            // No storage: a turn settles, or is abandoned, as soon as it is asked to.
            Effect::StorageAbort(speech) => {
                let _ = self.reports.send(Input::StorageAbandoned(speech));
            }
            Effect::StorageCommit { speech, history: _ } => {
                let _ = self.reports.send(Input::StorageSettled(speech));
            }
            Effect::ChatContext(snapshot) => {
                shared.chat_ctx.send_replace(snapshot);
            }
            Effect::Outputs(outputs) => {
                shared.output.send_replace(outputs);
            }
            Effect::Tap(tap) => self.peer.publish(tap),
            Effect::LaunchHold(hold) => shared.launch.publish(hold),
            Effect::SessionView(view) => {
                shared.view.send_replace(view);
            }
            Effect::Draining => {
                shared.draining.send_replace(true);
            }
            Effect::Closing => {
                shared.closing.send_replace(true);
            }
            Effect::SttConversationItem { stt, role, text } => {
                if let Some(stt) = resources.stt(stt)
                    && stt.capabilities().chat_context
                {
                    stt.push_conversation_item(&role, &text);
                }
            }
            Effect::SpeechStart { handle, mut context, llm, tts } => {
                // A realtime speech plays what the socket sends it, through its own player.
                let player = matches!(context.source, ReplySource::Realtime).then(|| {
                    let capacity =
                        lk_agents_voice::generation::realtime_generation::PLAYER_CAPACITY;
                    let (tx, rx) = mpsc::channel(capacity);
                    self.realtime_players.insert(handle.id(), tx);
                    rx
                });
                context.providers.llm = llm.and_then(|id| resources.llm(id));
                context.providers.tts = tts.and_then(|id| resources.tts(id));
                let task = lk_agents_voice::generation::pipeline_reply::pipeline_reply_task(
                    Arc::clone(&shared),
                    Arc::clone(&handle),
                    *context,
                    player,
                );
                handle.add_task(shared.spawner.spawn_unboxed(task));
            }
            Effect::Emit(event) => {
                self.peer.publish(SystemTap::Session(event.clone()));
                let _ = self.peer.deliver(HostEvent::Session(event));
            }
            Effect::FlushStream(sink) => {
                if let Some(sink) = resources.audio_sink(sink) {
                    let _ = sink.flush();
                }
            }
            Effect::EotFlush(reason) => {
                if let Some(eot) = &self.eot {
                    eot.flush(reason.as_str());
                }
            }
            Effect::EotPredict { req, timeout } => {
                if self.eot.is_some() {
                    shared.journal.record(lk_agents_voice::trace::InputStep::LegRequested {
                        leg: lk_agents_voice::trace::DetectorName::Eot,
                        req: req.0,
                    });
                    self.predict(req, timeout);
                }
            }
            Effect::EotCancel { req: _, timed_out } => {
                if let Some(eot) = &self.eot {
                    eot.cancel(timed_out);
                }
            }
            Effect::InterruptionSignal { overlap, signal } => {
                if let Some(detector) = &self.interruption {
                    match signal {
                        InterruptionSignal::OverlapStarted { .. } => {
                            let overlap =
                                overlap.unwrap_or(lk_agents_abi::turn::detector::ReqId(0));
                            detector.signal_overlap(overlap, signal);
                        }
                        InterruptionSignal::AgentSpeechStarted
                        | InterruptionSignal::AgentSpeechEnded
                        | InterruptionSignal::OverlapEnded { .. } => detector.signal(signal),
                    }
                }
            }
            Effect::SttStreamOpen { open, stt } => {
                let provider = resources.stt(stt).map(Provider::Stt);
                self.open_stream(open, StreamKind::Stt, provider);
            }
            Effect::VadStreamOpen { open, vad } => {
                let provider = resources.vad(vad).map(Provider::Vad);
                self.open_stream(open, StreamKind::Vad, provider);
            }
            Effect::ProviderPrewarm(id) => resources.prewarm(id),
            Effect::ProviderRelease(id) => resources.release(id),
            Effect::RealtimeClose(id) => {
                let Some(session) = resources.realtime(id) else {
                    return;
                };
                let reports = self.reports.clone();
                shared.spawner.spawn_unboxed(async move {
                    session.aclose().await;
                    let _ = reports.send(RealtimeWake::Closed);
                });
            }
            Effect::EotClose => {
                for task in self.predictions.drain(..) {
                    task.abort();
                }
            }
            Effect::Audio(AudioOp::Gate { speech: _, op, cause: _, output }) => {
                let Some(sink) = resources.audio_output(output) else {
                    return;
                };
                match op {
                    GateOp::Pause => sink.pause(),
                    GateOp::Resume => sink.resume(),
                    GateOp::ClearBuffer => sink.clear_buffer(),
                }
            }
            Effect::Audio(AudioOp::SetAudioOutput { previous, next, enabled }) => {
                if let Some(id) = previous {
                    self.recorder_observers.detach(&shared, Some(id));
                    if let Some(old) = resources.audio_output(id) {
                        old.on_detached();
                    }
                }
                if let Some(id) = next {
                    if let Some(new) = resources.audio_output(id) {
                        if enabled { new.on_attached() } else { new.on_detached() }
                    }
                    self.recorder_observers.attach(&shared, id);
                }
            }
            Effect::Audio(AudioOp::SetTranscriptionOutput { previous, next, enabled }) => {
                if let Some(old) = previous.and_then(|id| resources.text_output(id)) {
                    old.on_detached();
                }
                if let Some(new) = next.and_then(|id| resources.text_output(id)) {
                    if enabled { new.on_attached() } else { new.on_detached() }
                }
            }
            Effect::Audio(AudioOp::SetOutputEnabled { output, sink, enabled }) => match output {
                OutputKind::Audio => {
                    if let Some(sink) = sink.and_then(|id| resources.audio_output(id)) {
                        if enabled { sink.on_attached() } else { sink.on_detached() }
                    }
                }
                OutputKind::Transcription => {
                    if let Some(sink) = sink.and_then(|id| resources.text_output(id)) {
                        if enabled { sink.on_attached() } else { sink.on_detached() }
                    }
                }
            },
            Effect::Forwarder(command) => {
                let _ = self.reports.send(command);
            }
            Effect::Audio(AudioOp::SetAudioInput(input)) => {
                let source = input.map(|input| resources.register_audio_input(input.0));
                let _ = self.reports.send(ForwarderCommand::SetSource(source));
            }
            Effect::Audio(AudioOp::SetVideoInput(input)) => {
                if let Some(old) = self.video_forwarder.take() {
                    old.abort();
                }
                if let Some(old) = self.video_in.take() {
                    old.on_detached();
                }
                self.video_in = input.map(|input| input.0);
                if let Some(input) = &self.video_in {
                    input.on_attached();
                    let pump = self.video_parts.clone().pump(Arc::clone(input));
                    self.video_forwarder = Some(shared.spawner.spawn_unboxed(pump));
                }
            }
            Effect::Realtime { session, command } => {
                if let Some(session) = resources.realtime(session) {
                    session.send(command);
                }
            }
            Effect::CancelTool { call_id, runner } => runner.cancel(&call_id),
        }
    }

    /// Opens `provider`'s stream off the loop and reports it, or that the provider is gone, as
    /// `Input::StreamOpened`.
    fn open_stream(&mut self, open: u64, leg: StreamKind, provider: Option<Provider>) {
        let Some(provider) = provider else {
            let error = ProviderError::Local(
                ApiError::new("provider resource is no longer available").with_retryable(false),
            );
            let _ = self.reports.send_opened(StreamOpened { open, leg, outcome: Err(error) });
            return;
        };
        let reports = self.reports.clone();
        let time = self.shared.time.clone();
        self.shared.spawner.spawn_unboxed(async move {
            let outcome = match provider {
                Provider::Stt(stt) => match stt.stream(None, None).await {
                    Ok(Some((sink, source))) => Ok(Opened::Stt(sink, source)),
                    Err(e) => Err(e),
                    Ok(None) => Err(ProviderError::Local(ApiError::new(
                        "the new agent's STT does not stream",
                    ))),
                },
                Provider::Vad(vad) => {
                    vad.stream().await.map(|(sink, source)| Opened::Vad(sink, source))
                }
            };
            if let Err(StreamOpened { outcome: Ok(stream), .. }) =
                reports.send_opened(StreamOpened { open, leg, outcome })
            {
                stream.close(time.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
            }
        });
    }
}

/// Cores dropped before their finish (a panic, a livelock) still end every task that selects on
/// `closing`, and release everything both cores hold, as the runtime's drop does.
impl Drop for Executor {
    fn drop(&mut self) {
        self.shared.closing.send_replace(true);
        self.release();
        self.driver.release_all_providers();
        let effects = self.driver.take_outbox();
        self.perform_voice(effects);
    }
}

/// A provider whose stream is being opened.
enum Provider {
    Stt(Arc<dyn lk_agents_abi::Stt>),
    Vad(Arc<dyn lk_agents_abi::Vad>),
}

/// An opened stream's source half, held while the core decides: its sink is already registered.
enum PendingLeg {
    Stt(lk_agents_abi::SttSource),
    Vad(lk_agents_abi::VadSource),
}

impl Executor {
    /// Drains an activated stream into the queue.
    fn drain(&mut self, channel: AudioChannel, sink: ResourceId, stream: PendingLeg) {
        let shared = &self.shared;
        let (time, closing) = (shared.time.clone(), shared.closing.subscribe());
        match stream {
            PendingLeg::Stt(mut source) => {
                source.report_retries(lk_agents_voice::audio::report_to(self.reports.clone()));
                lk_agents_voice::audio::spawn_drain(
                    &shared.spawner,
                    time,
                    closing,
                    lk_agents_voice::audio::tagged(sink, source),
                    self.reports.clone(),
                    (channel, sink),
                );
            }
            PendingLeg::Vad(source) => lk_agents_voice::audio::spawn_drain(
                &shared.spawner,
                time,
                closing,
                source,
                self.reports.clone(),
                (channel, sink),
            ),
        }
    }

    /// The prediction off the loop; its answer joins the queue unless the session is closing.
    fn predict(&mut self, req: lk_agents_abi::ReqId, timeout: Option<std::time::Duration>) {
        let Some(detector) = self.eot.clone() else {
            return;
        };
        let queue = self.reports.clone();
        let closing = self.shared.closing.subscribe();
        let task = self.shared.spawner.spawn_unboxed(async move {
            let outcome = detector.predict(timeout).await;
            if !*closing.borrow() {
                let _ = queue.send(Input::Eot(req, outcome, None));
            }
        });
        self.predictions.retain(|task| !task.is_finished());
        self.predictions.push(task);
    }

    /// Until the detector's `Closed` (forwarded first), the loop's end, or `Closing`.
    fn pump_verdicts(&self, detector: Arc<dyn InterruptionPort>) {
        let tx = self.reports.clone();
        let mut closing = self.shared.closing.subscribe();
        self.shared.spawner.spawn_unboxed(async move {
            loop {
                let ev = tokio::select! {
                    biased;
                    _ = closing.wait_for(|c| *c) => break,
                    ev = detector.next_event() => ev,
                };
                let closed = matches!(ev, LegEvent::Closed);
                if tx.send(ev).is_err() || closed {
                    break;
                }
            }
        });
    }

    /// A source's events into the queue until `Closing`.
    fn forward<T: Into<Event> + Send + 'static>(
        &self,
        mut rx: tokio::sync::mpsc::UnboundedReceiver<T>,
    ) {
        let tx = self.reports.clone();
        let mut closing = self.shared.closing.subscribe();
        self.shared.spawner.spawn_unboxed(async move {
            loop {
                let ev = tokio::select! {
                    biased;
                    ev = rx.recv() => ev,
                    _ = closing.wait_for(|c| *c) => None,
                };
                if ev.is_none_or(|ev| tx.send(ev).is_err()) {
                    break;
                }
            }
        });
    }
}

/// The microphone's reader: every frame to the run's queue, until the track ends or the session
/// closes, as the runtime's reader does.
async fn read_audio(
    input: Arc<dyn AudioInput>,
    frames: mpsc::Sender<AudioFrame>,
    mut closing: watch::Receiver<bool>,
    queue: Tx,
) {
    let cause = loop {
        let frame = tokio::select! {
            biased;
            frame = input.next_frame() => match frame {
                Ok(Some(frame)) => frame,
                Ok(None) => break ChannelEndCause::SourceEnded,
                Err(_) => break ChannelEndCause::Error,
            },
            _ = closing.wait_for(|c| *c) => break ChannelEndCause::Closing,
        };
        tokio::select! {
            biased;
            sent = frames.send(frame) => if sent.is_err() { return },
            _ = closing.wait_for(|c| *c) => break ChannelEndCause::Closing,
        }
    };
    let end = ChannelEnd { leg: AUDIO_IN_LEG, cause, sink: None, closed: None, error: None };
    let _ = queue.send(end);
}
