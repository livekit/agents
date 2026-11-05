//! [`VoiceRuntime`](crate::voice::VoiceRuntime) executes a session's two cores through its
//! [`QueueOwner`](crate::voice::session::QueueOwner): the decision core
//! ([`VoiceDriver`](lk_agents_voice::agent_session::VoiceDriver)) and the data plane
//! ([`AudioInputRouter`](lk_agents_voice::data_plane::AudioInputRouter)). Its
//! [`run`](crate::voice::VoiceRuntime::run) loop steps them and performs their effects.
//! A simulation runs this same loop over fake ports, putting into its queue the events its
//! deterministic scheduler chooses, one at a time.
pub mod amd;
mod audio;
pub mod background_audio;
pub mod console;
pub mod default_nodes;
mod effects;
pub mod events;
pub use events::VoiceRuntimeEvent;
mod generation;
mod io;
mod ivr;
mod keyterm_detection;
mod legs;
pub mod nodes;
pub mod playback;
mod recorder;
pub mod remote_session;
pub mod resources;
pub mod session;
pub mod speech_handle;
pub mod transcription;
pub mod workflows;

pub use legs::Executor;

use crate::telemetry;
use crate::voice::session::Tx;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::{AudioInput, InputFrame};
use lk_agents_net::task::{TaskId, TaskSet};
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::{
    Now, Sent, SpeechCommand, VoiceCommandOrigin, VoiceDriver, VoiceEvent,
};
use lk_agents_voice::audio::{AUDIO_IN_LEG, ChannelEnd};
use lk_agents_voice::data_plane::{
    AudioInputRouter, AudioInputRouterCommand, AudioInputRouterEffect,
};
use lk_agents_voice::trace::ChannelEndCause;
use session::{Calls, PeerHandle, QueueOwner, SessionCores, SessionShared};
use speech_handle::Requests;
use std::sync::Arc;

pub use lk_agents_voice::agent_activity::options::InterruptionGateRules;
pub use lk_agents_voice::agent_session::lifecycle::SessionStartPlans;
pub use lk_agents_voice::amd::machine::classifier::AmdRules;
pub use lk_agents_voice::keyterm_detection::KeytermRules;
pub use lk_agents_voice::tool_executor::VoiceAgentFrame;
pub use lk_agents_voice::workflows::WorkflowRules;
pub use session::{SessionOptions, VoiceSession};

/// The microphone reader's handoff to the queue owner. A count, not a bound on audio: a full
/// handoff holds the reader back, so the loss falls to the input's own channel and its policy.
const READER_HANDOFF_FRAMES: usize = 50;
pub use speech_handle::SpeechHandle;

pub struct VoiceRuntime {
    publications: session::shared::Publications,
    driver: VoiceDriver,
    storage: Option<Arc<dyn lk_agents_abi::storage::voice::VoiceStorage>>,
    shared: Arc<SessionShared>,
    /// Where host events and taps go.
    peer: PeerHandle,
    forwarder: AudioInputRouter,
    /// One refused-audio warning per active stream, retired when that input ends.
    push_warned: std::collections::BTreeSet<ResourceId>,
    recorder_observers: recorder::RecorderOutputs,

    /// Every source's events, in arrival order. Frames never come this way. A simulation swaps in
    /// the queue it delivers its chosen events on.
    pub queue: tokio::sync::mpsc::UnboundedReceiver<VoiceRuntimeEvent>,
    /// The microphone's frames, straight from its reader. A full queue holds the reader back, as
    /// the reference's forwarding loop awaits each frame, so the loss falls to the input's own
    /// channel and the session never grows without limit.
    pub frames: tokio::sync::mpsc::Receiver<InputFrame>,
    /// Where the data plane's reports and a reader's end join the queue.
    reports: Tx,
    frames_tx: tokio::sync::mpsc::Sender<InputFrame>,
    /// The microphone, its id and its reader.
    source: Option<(ResourceId, TaskId)>,
    /// Where each awaited call's answer goes, until the core answers it.
    calls: Arc<Calls>,
    /// The keyterm detection pass in flight, by request.
    keyterms: Option<(u64, TaskId)>,
    /// Speech-scoped host callbacks, indexed for exact cancellation and final speech release.
    /// These are runtime task handles, not another source of run-completion decisions.
    host_callbacks: std::collections::BTreeMap<(SpeechId, HostCallbackId), TaskId>,
    /// Each realtime speech's player, fed the socket's events (`Effect::RealtimePlayer`).
    realtime_players: std::collections::BTreeMap<
        lk_agents_voice::agent_activity::scheduling::SpeechId,
        tokio::sync::mpsc::Sender<lk_agents_abi::llm::realtime::Event>,
    >,
    /// The camera read now, its forwarder, and what a new forwarder is built from.
    video_parts: audio::VideoForwarding,
    video_forwarder: Option<TaskId>,
    video_in: Option<Arc<dyn lk_agents_abi::voice::io::VideoInput>>,
    /// Opened STT and VAD streams the core has neither activated nor discarded yet.
    pending_legs: std::collections::BTreeMap<ResourceId, legs::OpeningLeg>,
    realtime: Option<legs::RealtimeConnection>,
    /// The turn and interruption detectors, addressed by the core's effects.
    eot: Option<Arc<dyn lk_agents_abi::EotDetector>>,
    /// The turn detector's predictions in flight, aborted at its close.
    predictions: Vec<TaskId>,
    interruption:
        Option<Arc<dyn lk_agents_voice::agent_activity::false_interruption::InterruptionPort>>,
    /// The executor's tasks: readers, drains, pumps, provider calls and closes. Most end at
    /// `Closing`; the run drains them within the cleanup bound and cancels the rest.
    tasks: Arc<TaskSet>,
    /// The tasks of the set each speech owns: its reply and the tool calls it made.
    speeches: SpeechTasks,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum HostCallbackId {
    OutputValidation(lk_agents_abi::agent::run::ValidationId),
    ToolRefusal(lk_agents_core::agent_run::CallId),
}

/// Which of the session's tasks each speech owns. Cancelling the speech (its interruption
/// watchdog) cancels them; a tool call outlives its reply until then.
#[derive(Clone)]
pub(crate) struct SpeechTasks {
    tasks: Arc<TaskSet>,
    spawner: Arc<dyn lk_agents_abi::Spawner>,
    owned: Arc<parking_lot::Mutex<std::collections::BTreeMap<SpeechId, Vec<TaskId>>>>,
}

impl SpeechTasks {
    pub(crate) fn spawn(
        &self,
        speech: SpeechId,
        name: &str,
        work: impl std::future::Future<Output = ()> + Send + 'static,
    ) -> TaskId {
        let task = self.tasks.spawn(self.spawner.as_ref(), name, work);
        let mut owned = self.owned.lock();
        owned.retain(|_, tasks| {
            tasks.retain(|task| self.tasks.contains(*task));
            !tasks.is_empty()
        });
        owned.entry(speech).or_default().push(task);
        task
    }

    fn cancel(&self, speech: SpeechId) {
        let owned = self.owned.lock().remove(&speech);
        owned.into_iter().flatten().for_each(|task| self.tasks.cancel(task));
    }
}

impl VoiceRuntime {
    /// Takes the cores over: the opening legs and the microphone are read and the start's
    /// effects performed.
    fn new(
        cores: SessionCores,
        opening: Vec<legs::OpeningLeg>,
        realtime: Option<legs::RealtimeConnection>,
    ) -> Self {
        let SessionCores {
            publications,
            driver,
            forwarder,
            queue: (reports, queue),
            source,
            eot,
            interruption,
            realtime_events,
            room_events: (room, room_io),
            video: video_in,
            shared,
            peer,
            storage,
        } = cores;
        let video_parts =
            audio::VideoForwarding { closing: shared.closing.clone(), queue: reports.clone() };
        let (frames_tx, frames) = tokio::sync::mpsc::channel(READER_HANDOFF_FRAMES);
        let tasks = Arc::<TaskSet>::default();
        let shared_spawner = Arc::clone(&shared.spawner);
        let calls = Arc::clone(&shared.calls);
        let mut runtime = VoiceRuntime {
            publications,
            driver,
            storage,
            shared,
            peer,
            forwarder,
            push_warned: Default::default(),
            recorder_observers: Default::default(),
            queue,
            frames,
            reports,
            frames_tx,
            source: None,
            pending_legs: Default::default(),
            realtime,
            calls,
            keyterms: None,
            host_callbacks: Default::default(),
            realtime_players: Default::default(),
            video_parts,
            video_forwarder: None,
            video_in,
            eot,
            predictions: Vec::new(),
            interruption,
            speeches: SpeechTasks {
                tasks: Arc::clone(&tasks),
                spawner: Arc::clone(&shared_spawner),
                owned: Arc::default(),
            },
            tasks,
        };
        if let Some(session) = realtime_events {
            runtime.realtime.get_or_insert_with(|| {
                legs::RealtimeConnection::new(
                    session.clone(),
                    runtime.shared.time.clone(),
                    runtime.shared.spawner.clone(),
                )
            });
            runtime.pump_realtime(session);
        }
        let ivr_detection = runtime.shared.view.borrow().config.effective.turn.ivr_detection();
        ivr::start(runtime.peer.clone(), ivr_detection, &runtime.tasks);
        if let Some(room_io) = room_io {
            let io_queue = runtime.reports.clone();
            let rtc_queue = runtime.reports.clone();
            room_io.forward_events(
                Arc::new(move |event| io_queue.send(event).is_ok()),
                Arc::new(move |event| rtc_queue.send(event).is_ok()),
            );
        }
        if let Some(rx) = room {
            runtime.forward(rx);
        }
        runtime.start_legs(opening);
        // The camera, one task per source, replaced whole by `SetVideoInput`.
        runtime.video_forwarder = runtime
            .video_in
            .clone()
            .map(|input| runtime.spawn("video forwarder", runtime.video_parts.clone().pump(input)));
        // The session's first microphone is the host's attached one: nothing to tell it.
        if let Some(source) = source {
            runtime.read(source);
        }
        let effects = runtime.driver.take_outbox();
        runtime.perform_voice(effects);
        runtime
    }

    /// The session's loop: [`serve`](Self::serve) until the decision core stops, then
    /// [`close`](Self::close).
    pub async fn run(&mut self) {
        self.serve().await;
        let late = self.close().await;
        if !late.is_empty() {
            log::error!(target: "livekit.agents", tasks = late.join(", ").as_str();
                "voice session tasks missed the cleanup deadline and were cancelled");
        }
    }

    /// Every event in arrival order, every frame as it comes, and a wake when
    /// [`VoiceDriver::wake_at`] comes, a due wake before any same-instant event; until the decision
    /// core stops.
    pub async fn serve(&mut self) {
        let time = self.shared.time.clone();
        self.open(time.read());
        while !self.driver.stopped() {
            let wake_at = self.driver.wake_at();
            if wake_at.is_some_and(|at| at <= time.now()) {
                self.handle(VoiceRuntimeEvent::Wake, time.read());
                continue;
            }
            tokio::select! {
                biased;
                () = time.wait_until_opt(wake_at) => self.handle(VoiceRuntimeEvent::Wake, time.read()),
                next = self.queue.recv() => match next {
                    Some(event) => self.handle(event, time.read()),
                    None => break,
                },
                Some(frame) = self.frames.recv() => self.on_frame(frame, time.read()),
            }
        }
    }

    /// The decision core stopped: the queue closes, what it still held is finished, and the
    /// session's tasks, which end at `Closing`, are drained. Returns the names of those still
    /// running at the cleanup bound, which are cancelled.
    pub async fn close(&mut self) -> Vec<String> {
        let mut undelivered = self.close_queue();
        // `try_recv` may miss a sender between reserving admission and publishing its value.
        // Once closed, recv reaches None only after those accepted sends have settled.
        while let Some(event) = self.queue.recv().await {
            self.retire_queued_event(event, &mut undelivered);
        }
        self.finish(undelivered);
        let _ = self.shared.report.set(self.driver.report(self.shared.time.wall()));
        let time = self.shared.time.clone();
        let deadline = time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let late = self.tasks.drain(time.clock().as_ref(), deadline).await;
        self.shared.resources.clear();
        late
    }

    fn open(&mut self, now: Now) {
        let effects = self.driver.open(now);
        self.perform_voice(effects);
    }

    /// One event: an input or a wake to the decision core, a command to the data plane.
    fn handle(&mut self, event: VoiceRuntimeEvent, now: Now) {
        match event {
            VoiceRuntimeEvent::Input(mut input) => {
                if let VoiceEvent::Gen(lk_agents_voice::generation::GenEvent::OutputValidated {
                    speech,
                    validation,
                    ..
                }) = &input
                {
                    self.host_callbacks
                        .remove(&(*speech, HostCallbackId::OutputValidation(*validation)));
                }
                if let VoiceEvent::Gen(
                    lk_agents_voice::generation::GenEvent::ToolHandlingTemplateRendered {
                        rendering:
                            lk_agents_voice::tool_executor::Rendering::Refusal { speech, call },
                        ..
                    },
                ) = &input
                {
                    self.host_callbacks.remove(&(*speech, HostCallbackId::ToolRefusal(*call)));
                }
                if let lk_agents_voice::agent_session::VoiceEvent::Stt(_, provider) = &mut input {
                    *provider = self
                        .driver
                        .stt()
                        .and_then(|stt| self.shared.resources.stt(stt))
                        .map(|stt| stt.id());
                }
                if let lk_agents_voice::agent_session::VoiceEvent::Eot(_, _, thresholds) =
                    &mut input
                {
                    *thresholds = self.eot.as_ref().and_then(|detector| detector.thresholds());
                }
                self.step(Some(input), now);
            }
            VoiceRuntimeEvent::Awaited(input, _) => {
                self.handle(VoiceRuntimeEvent::Input(input), now)
            }
            VoiceRuntimeEvent::Wake => self.step(None, now),
            VoiceRuntimeEvent::LegOpened(opened) => {
                let input = self.prepare_leg(opened);
                self.step(Some(input), now);
            }
            VoiceRuntimeEvent::AudioInputRouting(command) => self.command(command),
            VoiceRuntimeEvent::HostRequests => self.step_host_requests(now),
            VoiceRuntimeEvent::Report(reply) => {
                let _ = reply.send(self.driver.report(self.shared.time.wall()));
            }
        }
    }

    /// One step of the decision core (`None`: a wake), after every request the host made of its
    /// speeches before it: a request is ordered at its call, so whatever ends a speech, an
    /// `interrupt()` asked while the host still read it running lands first. The list stays
    /// locked until the step's views are published: a later request reads the speech's end.
    fn step(&mut self, input: Option<VoiceEvent>, now: Now) {
        let requests = self.shared.speeches.requests(self.reports.clone());
        let held = self.step_host_requests_from(&requests, now);
        let effects = self.driver.handle(input, now);
        let operations = self.publish_views(effects);
        drop(held);
        self.operate(operations);
    }

    fn step_host_requests(&mut self, now: Now) {
        let requests = self.shared.speeches.requests(self.reports.clone());
        drop(self.step_host_requests_from(&requests, now));
    }

    /// Steps the host's requests until none is left; returns the emptied list, still locked.
    fn step_host_requests_from<'a>(
        &mut self,
        requests: &'a Requests,
        now: Now,
    ) -> parking_lot::MutexGuard<'a, Vec<SpeechCommand>> {
        loop {
            let mut held = requests.lock();
            if held.is_empty() {
                return held;
            }
            let asked = std::mem::take(&mut *held);
            drop(held);
            for command in asked {
                let input = VoiceEvent::Host(Sent {
                    origin: VoiceCommandOrigin::Host,
                    command: command.into(),
                });
                let effects = self.driver.handle(Some(input), now);
                self.perform_voice(effects);
            }
        }
    }

    fn on_frame(&mut self, InputFrame { frame, head }: InputFrame, now: Now) {
        // The session recorder hears every frame the forwarder does, muted ones excepted.
        if self.forwarder.input_enabled()
            && let Some(tap) = self.shared.recorder_tap()
        {
            tap.recorder.on_input_frame(tap.now(), &frame);
        }
        let effects = self.forwarder.on_frame(frame, head, now.at);
        self.perform(effects);
    }

    /// Stop admission before retiring queued resources. Both normal close and Drop use this
    /// path: late stream results and installed sinks retain their cleanup owners until settled.
    fn close_queue(&mut self) -> Vec<VoiceEvent> {
        self.queue.close();
        self.peer.watch.close();
        let requests = self.shared.speeches.requests(self.reports.clone());
        let mut inputs: Vec<VoiceEvent> = std::mem::take(&mut *requests.lock())
            .into_iter()
            .map(|command| {
                VoiceEvent::Host(Sent { origin: VoiceCommandOrigin::Host, command: command.into() })
            })
            .collect();
        while let Ok(event) = self.queue.try_recv() {
            self.retire_queued_event(event, &mut inputs);
        }
        inputs
    }

    fn retire_queued_event(&mut self, event: VoiceRuntimeEvent, inputs: &mut Vec<VoiceEvent>) {
        match event {
            VoiceRuntimeEvent::Input(input) => inputs.push(input),
            // Its reply is kept, so the core's refusal reaches it.
            VoiceRuntimeEvent::Awaited(input, _) => inputs.push(input),
            VoiceRuntimeEvent::LegOpened(opened) => inputs.push(self.prepare_leg(opened)),
            // A microphone attached too late to be read is let go, not left in the table.
            VoiceRuntimeEvent::AudioInputRouting(AudioInputRouterCommand::SetSource(Some(
                source,
            ))) => {
                self.shared.resources.release(source);
            }
            VoiceRuntimeEvent::AudioInputRouting(command) => {
                for effect in self.forwarder.handle(command) {
                    match effect {
                        AudioInputRouterEffect::ReportToSession(input) => inputs.push(*input),
                        effect => self.perform(vec![effect]),
                    }
                }
            }
            VoiceRuntimeEvent::Report(reply) => {
                let _ = reply.send(self.driver.report(self.shared.time.wall()));
            }
            VoiceRuntimeEvent::Wake | VoiceRuntimeEvent::HostRequests => {}
        }
    }

    /// The decision core refuses what the loop never delivered and settles its own waiters once.
    fn finish(&mut self, inputs: Vec<VoiceEvent>) {
        let effects = self.driver.finish(inputs);
        self.perform_voice(effects);
        self.calls.clear();
        self.release();
        for opened in std::mem::take(&mut self.pending_legs).into_values() {
            opened.retire(&self.tasks);
        }
    }

    /// The session is over: every sink the data plane still feeds has its input ended, and the
    /// microphone is let go (its reader ends on the close).
    fn release(&mut self) {
        let effects = self.forwarder.close();
        self.perform(effects);
        self.set_source(None);
        self.set_video_input(None);
    }

    fn command(&mut self, command: AudioInputRouterCommand) {
        match command {
            AudioInputRouterCommand::SetSource(input) => self.set_source(input),
            // The host object is told, as `AgentInput.set_audio_enabled` does.
            AudioInputRouterCommand::SetInputEnabled(enabled) => {
                let resources = &self.shared.resources;
                if let Some(input) =
                    self.source.as_ref().and_then(|(id, _)| resources.audio_input(*id))
                {
                    if enabled { input.on_attached() } else { input.on_detached() }
                }
            }
            AudioInputRouterCommand::Install { .. }
            | AudioInputRouterCommand::Remove(_)
            | AudioInputRouterCommand::SetGateDiscard(_)
            | AudioInputRouterCommand::FlushSttWithSilence { .. }
            | AudioInputRouterCommand::ReportCensus => {}
        }
        let effects = self.forwarder.handle(command);
        self.perform(effects);
    }

    /// The old reader is aborted first: two sources on one forwarder look like the user talking
    /// over themselves. The old source is detached, the new one attached if input is enabled
    /// (`AgentInput.audio`'s setter).
    fn set_source(&mut self, input: Option<ResourceId>) {
        let resources = self.shared.resources.clone();
        if let Some((old, reader)) = self.source.take() {
            self.tasks.cancel(reader);
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

    fn set_video_input(&mut self, input: Option<ResourceId>) {
        if let Some(old) = self.video_forwarder.take() {
            self.tasks.cancel(old);
        }
        if let Some(old) = self.video_in.take() {
            old.on_detached();
        }
        self.video_in = input.and_then(|id| self.shared.resources.video_input(id));
        if let Some(input) = &self.video_in {
            input.on_attached();
            let pump = self.video_parts.clone().pump(Arc::clone(input));
            self.video_forwarder = Some(self.spawn("video forwarder", pump));
        }
    }

    /// Runs `work` as one of the executor's tasks.
    #[track_caller]
    fn spawn(
        &self,
        name: &str,
        work: impl std::future::Future<Output = ()> + Send + 'static,
    ) -> TaskId {
        self.tasks.spawn(self.shared.spawner.as_ref(), name, work)
    }

    fn read(&mut self, source: ResourceId) {
        let Some(input) = self.shared.resources.audio_input(source) else { return };
        let reader = read_audio(
            input,
            self.frames_tx.clone(),
            self.shared.closing.clone(),
            self.reports.clone(),
        );
        self.source = Some((source, self.spawn("audio reader", reader)));
    }

    fn perform(&mut self, effects: Vec<AudioInputRouterEffect>) {
        let resources = &self.shared.resources;
        for effect in effects {
            match effect {
                AudioInputRouterEffect::PushFrame { sink, frame, head } => {
                    if let Some(stream) = resources.audio_sink(sink)
                        && let Err(error) =
                            if head { stream.head(frame) } else { stream.push(frame) }
                        && self.push_warned.insert(sink)
                    {
                        log::warn!(
                            target: "livekit.agents",
                            "audio stream {sink:?} refused input: {error}; suppressing repeats"
                        );
                    }
                }
                AudioInputRouterEffect::EndInput(sink) => {
                    self.push_warned.remove(&sink);
                    if let Some(stream) = resources.audio_sink(sink) {
                        stream.end_input();
                    }
                    resources.release(sink);
                }
                AudioInputRouterEffect::ReportToSession(input) => {
                    let _ = self.reports.send(*input);
                }
            }
        }
    }
}

impl QueueOwner for VoiceRuntime {
    fn open(&mut self, now: Now) {
        VoiceRuntime::open(self, now);
    }

    fn run(mut self: Box<Self>) -> Option<lk_agents_abi::BoxFuture<'static, ()>> {
        Some(Box::pin(async move { VoiceRuntime::run(&mut self).await }))
    }
}

/// A runtime dropped before its close (an aborted loop, a panic) still ends every task that
/// selects on `closing`, and exports the spans its session still had open. The queue owner
/// publishes the terminal value before dropping its sender. It still releases everything both
/// cores hold, as its finish would have.
impl Drop for VoiceRuntime {
    fn drop(&mut self) {
        self.publications.closing.send_replace(true);
        drop(self.close_queue());
        self.release();
        self.pending_legs.clear();
        self.driver.release_all_providers();
        // The spans end here only when the session never closed; either way they go out below.
        let abandoned = self.driver.abandon_spans();
        if let Some(exporter) = self.shared.tracer.exporter() {
            exporter.session_ended(self.driver.spans_swept(), abandoned);
        }
        let effects = self.driver.take_outbox();
        self.perform_voice(effects);
        self.calls.clear();
        // Local admission guards keep borrowed objects alive until their own scopes end.
        self.shared.resources.clear();
    }
}

/// The microphone's reader: every frame to the queue owner, until the track ends or the session
/// closes. The input is the host's, not a `HostStream`: nothing to close, so no `Closed`.
async fn read_audio(
    input: Arc<dyn AudioInput>,
    frames: tokio::sync::mpsc::Sender<InputFrame>,
    mut closing: tokio::sync::watch::Receiver<bool>,
    queue: Tx,
) {
    // `biased` so an arrived frame is handed over first.
    let cause = loop {
        let frame = tokio::select! {
            biased;
            frame = input.next_input(&telemetry::Cx::NONE) => match frame {
                Ok(Some(frame)) => frame,
                Ok(None) => break ChannelEndCause::SourceEnded,
                Err(e) => {
                    log::warn!(target: "livekit.agents", error = e.to_string().as_str(); "audio input failed");
                    break ChannelEndCause::Error;
                }
            },
            _ = closing.wait_for(|c| *c) => break ChannelEndCause::Closing,
        };
        // A queue owner that is gone is a session that has returned: nothing to report to.
        tokio::select! {
            biased;
            sent = frames.send(frame) => if sent.is_err() { return },
            _ = closing.wait_for(|c| *c) => break ChannelEndCause::Closing,
        }
    };
    let end = ChannelEnd { leg: AUDIO_IN_LEG, cause, sink: None, closed: None, error: None };
    let _ = queue.send(end);
}

#[cfg(test)]
#[path = "../voice_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../voice_telemetry_tests.rs"]
mod telemetry_tests;
