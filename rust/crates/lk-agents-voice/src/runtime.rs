//! [`Runtime`]: a session's two cores, the decision core ([`Driver`]) and the data plane
//! ([`AudioForwarder`]), stepped by one queue owner, and what performs the data plane's effects.
//! The queue owner only feeds it: [`Runtime::run`] in production, the fuzzer's run in tests.

mod effects;
mod storage;

use crate::agent_session::{Driver, Event, Now, Tx};
use crate::audio::{AUDIO_IN_LEG, ChannelEnd};
use crate::data_plane::{AudioForwarder, ForwarderCommand, ForwarderEffect};
use crate::trace::ChannelEndCause;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::AudioInput;
use lk_agents_abi::{AUDIO_QUEUE_FRAMES, AbortHandle, AudioFrame, SpawnerExt};
use std::sync::Arc;

#[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
pub struct Runtime {
    pub driver: Driver,
    storage: Option<Arc<dyn lk_agents_abi::storage::voice::VoiceStorage>>,
    pub(crate) shared: Arc<crate::agent_session::SessionShared>,
    events: tokio::sync::mpsc::UnboundedSender<crate::agent_session::HostEvent>,
    watch: crate::agent_session::WatchHub,
    forwarder: AudioForwarder,
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
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
    /// The keyterm detection pass in flight, by request.
    keyterms: Option<(u64, AbortHandle)>,
    /// Each realtime speech's player, fed the socket's events (`Effect::RealtimePlayer`).
    realtime_players: std::collections::BTreeMap<
        crate::agent_activity::scheduling::SpeechId,
        tokio::sync::mpsc::UnboundedSender<lk_agents_abi::llm::realtime::Event>,
    >,
    /// The camera read now, its forwarder, and what a new forwarder is built from.
    video_parts: crate::audio::VideoForwarderParts,
    video_forwarder: Option<AbortHandle>,
    video_in: Option<Arc<dyn lk_agents_abi::voice::io::VideoInput>>,
    /// Opened STT and VAD streams the core has neither activated nor discarded yet.
    pending_legs: std::collections::BTreeMap<ResourceId, effects::PendingLeg>,
    /// The turn and interruption detectors, addressed by the core's effects.
    eot: Option<Arc<lk_agents_abi::EotLeg<dyn lk_agents_abi::EotDetector>>>,
    interruption: Option<Arc<dyn crate::agent_activity::false_interruption::InterruptionPort>>,
}

impl Runtime {
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    pub(crate) fn new(
        driver: Driver,
        forwarder: AudioForwarder,
        #[allow(clippy::disallowed_types, reason = "S0: runtime migration")] queue: (
            Tx,
            tokio::sync::mpsc::UnboundedReceiver<Event>,
        ),
        source: Option<ResourceId>,
        eot: Option<Arc<lk_agents_abi::EotLeg<dyn lk_agents_abi::EotDetector>>>,
        interruption: Option<Arc<dyn crate::agent_activity::false_interruption::InterruptionPort>>,
        (video_parts, video_forwarder, video_in): (
            crate::audio::VideoForwarderParts,
            Option<AbortHandle>,
            Option<Arc<dyn lk_agents_abi::voice::io::VideoInput>>,
        ),
        shared: Arc<crate::agent_session::SessionShared>,
        events: tokio::sync::mpsc::UnboundedSender<crate::agent_session::HostEvent>,
        watch: crate::agent_session::WatchHub,
        storage: Option<Arc<dyn lk_agents_abi::storage::voice::VoiceStorage>>,
    ) -> Self {
        let (frames_tx, frames) = tokio::sync::mpsc::channel(AUDIO_QUEUE_FRAMES);
        let (reports, queue) = queue;
        let mut runtime = Runtime {
            driver,
            storage,
            shared,
            events,
            watch,
            forwarder,
            queue,
            frames,
            reports,
            frames_tx,
            source: None,
            pending_legs: Default::default(),
            keyterms: None,
            realtime_players: Default::default(),
            video_parts,
            video_forwarder,
            video_in,
            eot,
            interruption,
        };
        // The session's first microphone is the host's attached one: nothing to tell it.
        if let Some(source) = source {
            runtime.read(source);
        }
        runtime
    }

    #[allow(clippy::disallowed_macros, reason = "S0: runtime migration")]
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
                if let crate::agent_session::Input::Stt(_, provider) = &mut input {
                    *provider = self
                        .driver
                        .session
                        .stt
                        .as_ref()
                        .and_then(|stt| self.shared.resources.stt(stt.id))
                        .map(|stt| stt.id());
                }
                if let crate::agent_session::Input::Eot(_, _, thresholds) = &mut input {
                    *thresholds = self.eot.as_ref().and_then(|leg| leg.thresholds());
                }
                let effects = self.driver.handle(Some(input), now);
                self.perform_voice(effects);
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
        if self.forwarder.input_enabled
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
        if self.forwarder.input_enabled
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
                // A dead host stream is the drain's to report, not the forwarder's per frame.
                ForwarderEffect::Push { sink, frame } => {
                    if let Some(sink) = resources.audio_sink(sink) {
                        let _ = sink.push(frame);
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
}

/// A runtime dropped before its close (an aborted loop, a panic) still ends every task that
/// selects on `closing`: the tasks sharing the session's state keep its sender alive. It still
/// releases everything both cores hold, as its finish would have.
impl Drop for Runtime {
    fn drop(&mut self) {
        self.shared.closing.send_replace(true);
        self.shared.calls.close();
        self.release();
        self.driver.release_all_providers();
        let effects = std::mem::take(&mut self.driver.outbox);
        self.perform_voice(effects);
    }
}

#[allow(clippy::disallowed_macros, reason = "S0: runtime migration")]
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
                Some(frame) => frame,
                None => break ChannelEndCause::SourceEnded,
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
