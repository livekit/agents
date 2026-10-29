//! [`Runtime`]: the production [`QueueOwner`] of a session's two cores, the decision core
//! ([`Driver`]) and the data plane ([`AudioForwarder`]): one loop steps them ([`Runtime::run`]) and
//! performs every effect they return. A simulation steps this same owner over fake ports, choosing
//! ready events and task turns with its deterministic scheduler.

pub mod amd;
pub mod audio;
pub mod background_audio;
pub mod console;
mod effects;
pub mod generation;
mod legs;
mod ivr;
mod recorder;
pub mod remote_session;
mod storage;
pub mod workflows;

pub use legs::Executor;

use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::AudioInput;
use lk_agents_abi::{AUDIO_QUEUE_FRAMES, AbortHandle, AudioFrame, SpawnerExt};
use lk_agents_voice::agent_session::{
    Driver, Event, Now, PeerHandle, QueueOwner, SessionCores, Tx,
};
use lk_agents_voice::audio::{AUDIO_IN_LEG, ChannelEnd};
use lk_agents_voice::data_plane::{AudioForwarder, ForwarderCommand, ForwarderEffect};
use lk_agents_voice::trace::ChannelEndCause;
use std::sync::Arc;

pub use lk_agents_voice::agent_session::SessionApi;
pub use lk_agents_voice::tool_executor::VoiceAgentFrame;

pub struct Runtime {
    pub driver: Driver,
    storage: Option<Arc<dyn lk_agents_abi::storage::voice::VoiceStorage>>,
    shared: Arc<lk_agents_voice::agent_session::SessionShared>,
    /// Where host events and taps go.
    peer: PeerHandle,
    forwarder: AudioForwarder,
    /// One refused-audio warning per active stream, retired when that input ends.
    push_warned: std::collections::BTreeSet<ResourceId>,
    recorder_observers: recorder::RecorderOutputs,

    /// Every source's events, in arrival order. Frames never come this way.
    pub queue: tokio::sync::mpsc::UnboundedReceiver<Event>,
    /// The microphone's frames, straight from its reader: a second of audio. A full queue holds
    /// the reader back, as the reference's forwarding loop awaits each frame, so the loss falls to
    /// the input's own drop-oldest queue and the session never grows without limit.
    pub frames: tokio::sync::mpsc::Receiver<AudioFrame>,
    /// Where the data plane's reports and a reader's end join the queue.
    reports: Tx,
    frames_tx: tokio::sync::mpsc::Sender<AudioFrame>,
    /// The microphone, its id and its reader.
    source: Option<(ResourceId, AbortHandle)>,
    /// Where each awaited call's answer goes, until the core answers it.
    calls: std::collections::BTreeMap<
        lk_agents_voice::agent_session::RequestId,
        lk_agents_voice::agent_session::ReplyTo,
    >,
    /// The keyterm detection pass in flight, by request.
    keyterms: Option<(u64, AbortHandle)>,
    /// Each realtime speech's player, fed the socket's events (`Effect::RealtimePlayer`).
    realtime_players: std::collections::BTreeMap<
        lk_agents_voice::agent_activity::scheduling::SpeechId,
        tokio::sync::mpsc::Sender<lk_agents_abi::llm::realtime::Event>,
    >,
    /// The camera read now, its forwarder, and what a new forwarder is built from.
    video_parts: audio::VideoForwarderParts,
    video_forwarder: Option<AbortHandle>,
    video_in: Option<Arc<dyn lk_agents_abi::voice::io::VideoInput>>,
    /// Opened STT and VAD streams the core has neither activated nor discarded yet.
    pending_legs: std::collections::BTreeMap<ResourceId, legs::PendingLeg>,
    /// The turn and interruption detectors, addressed by the core's effects.
    eot: Option<Arc<dyn lk_agents_abi::EotDetector>>,
    /// The turn detector's predictions in flight, aborted at its close.
    predictions: Vec<AbortHandle>,
    interruption:
        Option<Arc<dyn lk_agents_voice::agent_activity::false_interruption::InterruptionPort>>,
}

impl Runtime {
    /// Takes the cores over: the opening legs and the microphone are read and the start's
    /// effects performed.
    fn new(cores: SessionCores, opening: Vec<legs::OpeningLeg>) -> Self {
        let SessionCores {
            driver,
            forwarder,
            queue: (reports, queue),
            source,
            eot,
            interruption,
            room_events: (room, room_io),
            video: video_in,
            shared,
            peer,
            storage,
        } = cores;
        let video_parts = audio::VideoForwarderParts {
            closing: shared.closing.subscribe(),
            queue: reports.clone(),
        };
        let (frames_tx, frames) = tokio::sync::mpsc::channel(AUDIO_QUEUE_FRAMES);
        let mut runtime = Runtime {
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
            calls: Default::default(),
            keyterms: None,
            realtime_players: Default::default(),
            video_parts,
            video_forwarder: None,
            video_in,
            eot,
            predictions: Vec::new(),
            interruption,
        };
        let ivr_detection = runtime.shared.view.borrow().config.effective.turn.ivr_detection();
        ivr::start(runtime.peer.clone(), ivr_detection);
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
        runtime.video_forwarder = runtime.video_in.clone().map(|input| {
            runtime.shared.spawner.spawn_unboxed(runtime.video_parts.clone().pump(input))
        });
        // The session's first microphone is the host's attached one: nothing to tell it.
        if let Some(source) = source {
            runtime.read(source);
        }
        let effects = runtime.driver.take_outbox();
        runtime.perform_voice(effects);
        runtime
    }

    /// Every event in arrival order, every frame as it comes, and a wake when
    /// [`Driver::wake_at`] comes, a due wake before any same-instant event.
    pub async fn run(&mut self) {
        let time = self.shared.time.clone();
        self.open(time.read());
        while !self.driver.stopped() {
            let wake_at = self.driver.wake_at();
            if wake_at.is_some_and(|at| at <= time.now()) {
                self.handle(Event::Wake, time.read());
                continue;
            }
            tokio::select! {
                biased;
                () = time.wait_until_opt(wake_at) => self.handle(Event::Wake, time.read()),
                next = self.queue.recv() => match next {
                    Some(event) => self.handle(event, time.read()),
                    None => break,
                },
                Some(frame) = self.frames.recv() => self.on_frame(frame, time.read()),
            }
        }
        // Closed first, so a command sent from now on is refused to its sender rather than lost.
        self.queue.close();
        let undelivered = std::iter::from_fn(|| self.queue.try_recv().ok()).collect();
        self.finish(undelivered);
    }

    pub fn open(&mut self, now: Now) {
        let effects = self.driver.open(now);
        self.perform_voice(effects);
    }

    /// One event: an input or a wake to the decision core, a command to the data plane.
    pub fn handle(&mut self, event: Event, now: Now) {
        match event {
            Event::Input(mut input) => {
                if let lk_agents_voice::agent_session::Input::Stt(_, provider) = &mut input {
                    *provider = self
                        .driver
                        .stt()
                        .and_then(|stt| self.shared.resources.stt(stt))
                        .map(|stt| stt.id());
                }
                if let lk_agents_voice::agent_session::Input::Eot(_, _, thresholds) = &mut input {
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

    pub fn on_frame(&mut self, frame: AudioFrame, now: Now) {
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
    /// replaced sink is ended and the census is journalled; the decision core refuses the rest.
    pub fn finish(&mut self, undelivered: Vec<Event>) {
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
                // A microphone attached too late to be read is let go, not left in the table.
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

    /// The session is over: every sink the data plane still feeds has its input ended, and the
    /// microphone is let go (its reader ends on the close).
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
            // The host object is told, as `AgentInput.set_audio_enabled` does.
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

    /// The old reader is aborted first: two sources on one forwarder look like the user talking
    /// over themselves. The old source is detached, the new one attached if input is enabled
    /// (`AgentInput.audio`'s setter).
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
        let Some(input) = self.shared.resources.audio_input(source) else { return };
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
                ForwarderEffect::Push { sink, frame } => {
                    if let Some(stream) = resources.audio_sink(sink)
                        && let Err(error) = stream.push(frame)
                        && self.push_warned.insert(sink)
                    {
                        log::warn!(
                            target: "livekit.agents",
                            "audio stream {sink:?} refused input: {error}; suppressing repeats"
                        );
                    }
                }
                ForwarderEffect::EndInput(sink) => {
                    self.push_warned.remove(&sink);
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
}

impl QueueOwner for Runtime {
    fn open(&mut self, now: Now) {
        Runtime::open(self, now);
    }

    fn run(mut self: Box<Self>) -> Option<lk_agents_abi::BoxFuture<'static, ()>> {
        Some(Box::pin(async move { Runtime::run(&mut self).await }))
    }
}

/// A runtime dropped before its close (an aborted loop, a panic) still ends every task that
/// selects on `closing`: the tasks sharing the session's state keep its sender alive. It still
/// releases everything both cores hold, as its finish would have.
impl Drop for Runtime {
    fn drop(&mut self) {
        self.shared.closing.send_replace(true);
        self.release();
        self.driver.release_all_providers();
        let effects = self.driver.take_outbox();
        self.perform_voice(effects);
    }
}

/// The microphone's reader: every frame to the queue owner, until the track ends or the session
/// closes. The input is the host's, not a `HostStream`: nothing to close, so no `Closed`.
async fn read_audio(
    input: Arc<dyn AudioInput>,
    frames: tokio::sync::mpsc::Sender<AudioFrame>,
    mut closing: tokio::sync::watch::Receiver<bool>,
    queue: Tx,
) {
    // `biased` so an arrived frame is handed over first.
    let cause = loop {
        let frame = tokio::select! {
            biased;
            frame = input.next_frame() => match frame {
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
mod tests;

#[cfg(test)]
mod api_tests;
