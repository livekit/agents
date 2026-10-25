//! [`Runtime`]: a session's two cores, the decision core ([`Driver`]) and the data plane
//! ([`AudioForwarder`]), stepped by one queue owner, and what performs the data plane's effects.
//! The queue owner only feeds it: [`Runtime::run`] in production, the fuzzer's run in tests.

use crate::agent_session::{Driver, Event, Input, Now, Tx};
use crate::audio::{AUDIO_IN_LEG, ChannelEnd};
use crate::data_plane::{AudioForwarder, ForwarderCommand, ForwarderEffect};
use crate::trace::ChannelEndCause;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::AudioInput;
use lk_agents_abi::{AbortHandle, AudioFrame, SpawnerExt};
use std::sync::Arc;

pub struct Runtime {
    pub driver: Driver,
    forwarder: AudioForwarder,
    /// Every source's events, in arrival order. Frames never come this way.
    pub queue: tokio::sync::mpsc::UnboundedReceiver<Event>,
    /// The microphone's frames, straight from its reader.
    pub frames: tokio::sync::mpsc::UnboundedReceiver<AudioFrame>,
    /// Where the data plane's reports join the queue.
    reports: Tx<Input>,
    frames_tx: tokio::sync::mpsc::UnboundedSender<AudioFrame>,
    /// Where a reader reports that its source ended.
    ended: Tx<ChannelEnd>,
    /// The microphone, its id and its reader.
    source: Option<(ResourceId, AbortHandle)>,
}

impl Runtime {
    pub(crate) fn new(
        driver: Driver,
        forwarder: AudioForwarder,
        queue: (Tx<Input>, tokio::sync::mpsc::UnboundedReceiver<Event>),
        ended: Tx<ChannelEnd>,
        source: Option<ResourceId>,
    ) -> Self {
        let (frames_tx, frames) = tokio::sync::mpsc::unbounded_channel();
        let (reports, queue) = queue;
        let mut runtime =
            Runtime { driver, forwarder, queue, frames, reports, frames_tx, ended, source: None };
        // The session's first microphone is the host's attached one: nothing to tell it.
        if let Some(source) = source {
            runtime.read(source);
        }
        runtime
    }

    /// Every event in arrival order, every frame as it comes, and a wake when
    /// [`Driver::wake_at`] comes, a due wake before any same-instant event.
    pub async fn run(&mut self) {
        let time = self.driver.shared.time.clone();
        self.driver.open(time.read());
        while !self.driver.stopped() {
            let wake_at = self.driver.wake_at();
            if wake_at.is_some_and(|at| at <= time.now()) {
                self.handle(Event::Wake, time.read());
                continue;
            }
            let settled = self.driver.settled();
            tokio::select! {
                biased;
                () = time.wait_until_opt(wake_at) => self.handle(Event::Wake, time.read()),
                () = settled => self.handle(Event::Wake, time.read()),
                Some(frame) = self.frames.recv() => self.on_frame(frame, time.read()),
                next = self.queue.recv() => match next {
                    Some(event) => self.handle(event, time.read()),
                    None => break,
                },
            }
        }
        let undelivered = std::iter::from_fn(|| self.queue.try_recv().ok()).collect();
        self.finish(undelivered);
    }

    /// One event: an input or a wake to the decision core, a command to the data plane.
    pub fn handle(&mut self, event: Event, now: Now) {
        match event {
            Event::Input(input) => self.driver.handle(Some(input), now),
            Event::Wake => self.driver.handle(None, now),
            Event::Forwarder(command) => self.command(command),
        }
    }

    pub fn on_frame(&mut self, frame: AudioFrame, now: Now) {
        // The session recorder hears every frame the forwarder does, muted ones excepted.
        if self.forwarder.input_enabled
            && let Some(tap) = self.driver.shared.recorder_tap()
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
        self.driver.finish(inputs);
    }

    fn command(&mut self, command: ForwarderCommand) {
        match command {
            ForwarderCommand::SetSource(input) => self.set_source(input),
            // The host object is told, as `AgentInput.set_audio_enabled` does.
            ForwarderCommand::SetInputEnabled(enabled) => {
                let resources = &self.driver.shared.resources;
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
        let resources = self.driver.shared.resources.clone();
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
        let Some(input) = self.driver.shared.resources.audio_input(source) else { return };
        let reader = read_audio(
            input,
            self.frames_tx.clone(),
            self.driver.shared.closing.subscribe(),
            self.ended.clone(),
        );
        self.source = Some((source, self.driver.shared.spawner.spawn_unboxed(reader)));
    }

    fn perform(&mut self, effects: Vec<ForwarderEffect>) {
        let resources = &self.driver.shared.resources;
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

/// The microphone's reader: every frame to the queue owner, until the track ends or the session
/// closes. The input is the host's, not a `HostStream`: nothing to close, so no `Closed`.
async fn read_audio(
    input: Arc<dyn AudioInput>,
    frames: tokio::sync::mpsc::UnboundedSender<AudioFrame>,
    mut closing: tokio::sync::watch::Receiver<bool>,
    ended: Tx<ChannelEnd>,
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
        if frames.send(frame).is_err() {
            return;
        }
    };
    let end = ChannelEnd { leg: AUDIO_IN_LEG, cause, sink: None, closed: None, error: None };
    let _ = ended.send(end);
}
