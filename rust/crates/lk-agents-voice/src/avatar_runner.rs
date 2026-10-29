//! `voice/avatar/_runner.py`'s `AvatarRunner`, the avatar worker's side: the agent's audio from an
//! `AudioReceiver` into the user's `VideoGenerator`, what it generates out through the SDK's
//! `rtc.AVSynchronizer` (publishing the tracks on the first frame), and each segment's playback
//! reported back.
//!
//! [`RunnerCore`] makes every decision as a state machine: `handle(event, now) -> effects`, no
//! timers of its own. The frames are plain records ([`AvatarFrame`], [`MediaFrame`]): the core
//! decides where each one goes, never what is in it. The shared runtime queue owner
//! executes each [`IoOp`] against the host proxies and answer it with an [`Event`].

use std::collections::{BTreeMap, VecDeque};

use lk_agents_abi::Ts;
use lk_agents_abi::errors::ProviderError;
pub use lk_agents_abi::utils::video::{AudioReceiverFrame, AvatarFrame, MediaFrame};

const LOG: &str = "livekit.agents";

// ----------------------------------------------------------------------------- the host's side

pub use lk_agents_abi::voice::avatar::{AvatarAudioReceiver, AvatarMedia, AvatarVideoGenerator};
use lk_agents_abi::voice::io::AvatarPlaybackSegment;

// ----------------------------------------------------------------------------- the core

/// A host call into the runner, answered by [`Effect::Answer`] with the same id.
pub type CallId = u64;

pub use lk_agents_abi::voice::avatar::RunnerError;

/// What happened: a host call, the room, or the answer to an [`IoOp`].
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// `start()`, once the receiver has started; whether the room is connected now.
    Start {
        call: CallId,
        connected: bool,
    },
    WaitForComplete {
        call: CallId,
    },
    /// The receiver's `clear_buffer` event.
    ClearBuffer {
        epoch: u64,
    },
    /// `aclose()`.
    Close {
        call: CallId,
    },
    /// The room's connection state changed.
    Connection {
        connected: bool,
    },
    RoomFailed(ProviderError),
    AudioOpened(Result<(), ProviderError>),
    Audio(
        Result<
            (u64, bool, Option<AvatarPlaybackSegment>, Option<AudioReceiverFrame>),
            ProviderError,
        >,
    ),
    AudioPushed(Result<(), ProviderError>),
    FramesOpened(Result<(), ProviderError>),
    Frame(Result<Option<AvatarFrame>, ProviderError>),
    Published(Result<(), ProviderError>),
    Pushed(Result<(), ProviderError>),
    Reported {
        finished: bool,
        result: Result<(), ProviderError>,
    },
    GeneratorCleared {
        clear: u64,
        result: Result<(), ProviderError>,
    },
    ReceiverClosed(Result<(), ProviderError>),
    /// Every [`IoOp`] in flight was cancelled ([`Effect::CancelIo`]).
    IoCancelled,
    MediaClosed(Result<(), ProviderError>),
}

/// What the runner asks of its receiver, generator and media.
#[derive(Debug, Clone, PartialEq)]
pub enum IoOp {
    OpenAudio,
    NextAudio,
    PushAudio(Option<MediaFrame>),
    OpenFrames,
    NextFrame,
    Publish,
    Push(MediaFrame),
    ReportStarted { segment: AvatarPlaybackSegment },
    ReportFinished { segment: AvatarPlaybackSegment, position: f64, interrupted: bool },
    ClearGenerator { clear: u64 },
    CloseReceiver,
    CloseMedia,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    /// Run this; its answer comes back as an [`Event`].
    Io(IoOp),
    /// Cancel every [`IoOp`] in flight and wait for them to be gone, then [`Event::IoCancelled`].
    CancelIo,
    Answer {
        call: CallId,
        result: Result<(), RunnerError>,
    },
}

/// Where one of the two loops (`_read_audio`, `_forward_video`) is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Pump {
    #[default]
    Idle,
    /// An op in flight; its answer moves the loop on.
    Busy,
    /// Waiting for the tracks to be published before pushing the held frame.
    Publishing,
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Publish {
    Never,
    /// Asked, waiting for the room to connect.
    Waiting,
    InFlight,
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Closing {
    /// The receiver is closing; the loops still run, as the reference's do.
    Receiver,
    Cancelling,
    Media,
}

/// The runner's decisions, one event at a time.
#[derive(Debug)]
pub struct RunnerCore {
    lazy_publish: bool,
    /// `_room_connected_fut`.
    connected: bool,
    publish: Publish,
    /// The `start()` waiting on a publish.
    start_call: Option<CallId>,
    started: bool,
    read: Pump,
    forward: Pump,
    /// The frame the forward loop holds while the tracks publish, and its duration if audio.
    held: Option<(MediaFrame, Option<f64>)>,
    /// The audio frame being pushed, and its duration: reported once the push returns.
    pushing: Option<f64>,
    /// `_playback_position`.
    position: f64,
    /// The reference's `_audio_playing`, counted: the segments whose audio went into the generator
    /// and whose end has not come out of it. The generator holds a segment's end while the next
    /// segment's audio is read, and one flag for both reported the first and lost the second.
    owed: VecDeque<AvatarPlaybackSegment>,
    /// The segment the receiver is handing over has had audio.
    reading: bool,
    /// Each clear in flight and the actual segments read whose finishes it owes.
    clears: BTreeMap<u64, Clear>,
    clear_epoch: u64,
    held_audio: Option<(u64, bool, Option<AvatarPlaybackSegment>, AudioReceiverFrame)>,
    reporting: bool,
    // Pending finish I/O is bounded by the segments already read; new reads are held.
    reports: VecDeque<PendingReport>,
    frame_pull_paused: bool,
    /// A clear cut the segment the receiver was handing over: its end, read before any new
    /// audio, is the cut segment's, whose finish the clear reported. The reference pushes it on to
    /// the generator, where it reports the next segment finished before it played.
    stale_end: bool,
    next_clear: u64,
    /// An audio frame is on its way into the generator. A clear waits for it:
    /// issued alongside, the push could land after the clear and play the cut audio.
    audio_pushing: bool,
    /// The clears waiting for that push to return.
    clears_waiting: Vec<u64>,
    /// How the loops ended, as `asyncio.gather` sees them: the first failure, else both done.
    finished: u8,
    failed: Option<RunnerError>,
    waiting: Vec<CallId>,
    closing: Option<(Closing, CallId)>,
    closed: bool,
}

#[derive(Debug)]
struct Clear {
    epoch: u64,
    owed: VecDeque<AvatarPlaybackSegment>,
}

#[derive(Debug)]
struct PendingReport {
    segment: AvatarPlaybackSegment,
    position: f64,
    interrupted: bool,
}

impl RunnerCore {
    pub fn new(lazy_publish: bool) -> Self {
        RunnerCore {
            lazy_publish,
            connected: false,
            publish: Publish::Never,
            start_call: None,
            started: false,
            read: Pump::Idle,
            forward: Pump::Idle,
            held: None,
            pushing: None,
            position: 0.0,
            owed: VecDeque::new(),
            reading: false,
            clears: BTreeMap::new(),
            clear_epoch: 0,
            held_audio: None,
            reporting: false,
            reports: VecDeque::new(),
            frame_pull_paused: false,
            stale_end: false,
            next_clear: 0,
            audio_pushing: false,
            clears_waiting: Vec::new(),
            finished: 0,
            failed: None,
            waiting: Vec::new(),
            closing: None,
            closed: false,
        }
    }

    /// The runner has no timers.
    pub fn wake_at(&self) -> Option<Ts> {
        None
    }

    /// Whether the loops are being cut: no answer moves them on any more.
    fn cut(&self) -> bool {
        self.closed || matches!(self.closing, Some((Closing::Cancelling | Closing::Media, _)))
    }

    pub fn handle(&mut self, event: Event, _now: Ts) -> Vec<Effect> {
        let mut out = Vec::new();
        match event {
            Event::Start { call, connected } => {
                self.connected |= connected;
                if self.lazy_publish {
                    self.begin(&mut out);
                    out.push(Effect::Answer { call, result: Ok(()) });
                } else {
                    self.start_call = Some(call);
                    self.ask_publish(&mut out);
                }
            }
            Event::WaitForComplete { call } => {
                self.waiting.push(call);
                self.answer_waiting(&mut out);
            }
            Event::Connection { connected } => {
                if connected && !self.connected {
                    self.connected = true;
                    if self.publish == Publish::Waiting {
                        self.publish = Publish::InFlight;
                        out.push(Effect::Io(IoOp::Publish));
                    }
                }
            }
            Event::RoomFailed(error) if !self.cut() => {
                let error = RunnerError::Failed(error);
                self.failed.get_or_insert(error.clone());
                if let Some(call) = self.start_call.take() {
                    out.push(Effect::Answer { call, result: Err(error) });
                }
                self.answer_waiting(&mut out);
            }
            Event::ClearBuffer { epoch } if !self.cut() && epoch > self.clear_epoch => {
                self.clear_epoch = epoch;
                let clear = self.next_clear;
                self.next_clear += 1;
                self.clears.insert(clear, Clear { epoch, owed: std::mem::take(&mut self.owed) });
                self.stale_end = std::mem::take(&mut self.reading);
                if self.audio_pushing {
                    self.clears_waiting.push(clear);
                } else {
                    out.push(Effect::Io(IoOp::ClearGenerator { clear }));
                }
            }
            Event::GeneratorCleared { clear, result } if !self.cut() => {
                let owed =
                    self.clears.remove(&clear).map_or_else(VecDeque::new, |clear| clear.owed);
                if let Err(error) = result {
                    log_error("_handle_clear_buffer", &error);
                }
                // The receiver has cut this input even if the generator could not clear.
                for segment in owed {
                    let position = std::mem::take(&mut self.position);
                    self.report_finished(segment, position, true, &mut out);
                }
                self.release_audio(&mut out);
            }
            Event::AudioOpened(result) if !self.cut() => match result {
                Ok(()) => out.push(Effect::Io(IoOp::NextAudio)),
                Err(e) => self.loop_ended(true, Err(e), &mut out),
            },
            Event::Audio(result) if !self.cut() => match result {
                Ok((epoch, starts_segment, segment, Some(_))) if epoch < self.clear_epoch => {
                    if starts_segment {
                        if let Some(clear) =
                            self.clears.values_mut().find(|clear| clear.epoch > epoch)
                        {
                            if let Some(segment) = segment {
                                clear.owed.push_back(segment);
                            }
                        } else {
                            if let Some(segment) = segment {
                                self.report_finished(segment, 0.0, true, &mut out);
                            }
                        }
                    }
                    out.push(Effect::Io(IoOp::NextAudio));
                }
                Ok((_, _, _, Some(AudioReceiverFrame::SegmentEnd)))
                    if std::mem::take(&mut self.stale_end) =>
                {
                    out.push(Effect::Io(IoOp::NextAudio));
                }
                Ok((epoch, starts_segment, segment, Some(item)))
                    if !self.clears.is_empty() || self.reporting =>
                {
                    self.held_audio = Some((epoch, starts_segment, segment, item));
                }
                Ok((_, _, None, Some(_))) => {
                    self.loop_ended(
                        true,
                        Err(lk_agents_abi::errors::ApiError::new(
                            "avatar audio has no segment identity",
                        )
                        .into()),
                        &mut out,
                    );
                }
                Ok((_, _, Some(segment), Some(item))) => {
                    let frame = match item {
                        AudioReceiverFrame::Audio(frame) => {
                            if !std::mem::replace(&mut self.reading, true) {
                                self.owed.push_back(segment);
                            }
                            self.stale_end = false;
                            Some(MediaFrame::Audio(frame))
                        }
                        AudioReceiverFrame::SegmentEnd => {
                            self.reading = false;
                            None
                        }
                    };
                    self.audio_pushing = true;
                    out.push(Effect::Io(IoOp::PushAudio(frame)));
                }
                Ok((_, _, _, None)) => self.loop_ended(true, Ok(()), &mut out),
                Err(e) => self.loop_ended(true, Err(e), &mut out),
            },
            Event::AudioPushed(result) if !self.cut() => {
                self.audio_pushing = false;
                for clear in std::mem::take(&mut self.clears_waiting) {
                    out.push(Effect::Io(IoOp::ClearGenerator { clear }));
                }
                match result {
                    Ok(()) => out.push(Effect::Io(IoOp::NextAudio)),
                    Err(e) => self.loop_ended(true, Err(e), &mut out),
                }
            }
            Event::FramesOpened(result) if !self.cut() => match result {
                Ok(()) => self.next_frame(&mut out),
                Err(e) => self.loop_ended(false, Err(e), &mut out),
            },
            Event::Frame(result) if !self.cut() => match result {
                Ok(Some(AvatarFrame::SegmentEnd)) => {
                    // the agent hears the segment finished; the forwarding goes on
                    if let Some(segment) = self.owed.pop_front() {
                        let position = std::mem::take(&mut self.position);
                        self.report_finished(segment, position, false, &mut out);
                    }
                    self.next_frame(&mut out);
                }
                Ok(Some(AvatarFrame::Audio(frame))) => {
                    let duration = frame.duration();
                    self.forward(MediaFrame::Audio(frame), Some(duration), &mut out)
                }
                Ok(Some(AvatarFrame::Video(frame))) => {
                    self.forward(MediaFrame::Video(frame), None, &mut out)
                }
                Ok(None) => self.loop_ended(false, Ok(()), &mut out),
                Err(e) => self.loop_ended(false, Err(e), &mut out),
            },
            Event::Published(result) if !self.cut() => {
                self.publish = if result.is_ok() { Publish::Done } else { Publish::Never };
                if let Some(call) = self.start_call.take() {
                    match result {
                        Ok(()) => {
                            self.begin(&mut out);
                            out.push(Effect::Answer { call, result: Ok(()) });
                        }
                        Err(e) => {
                            out.push(Effect::Answer { call, result: Err(RunnerError::Failed(e)) })
                        }
                    }
                } else if self.forward == Pump::Publishing {
                    match result {
                        Ok(()) => {
                            if let Some((frame, duration)) = self.held.take() {
                                self.push(frame, duration, &mut out);
                            }
                        }
                        Err(e) => {
                            self.held = None;
                            self.loop_ended(false, Err(e), &mut out);
                        }
                    }
                }
            }
            Event::Pushed(result) if !self.cut() => match result {
                Ok(()) => {
                    if let Some(duration) = self.pushing.take() {
                        let started = self.position == 0.0 && duration > 0.0;
                        self.position += duration;
                        if started {
                            if let Some(segment) = self.owed.front().cloned() {
                                out.push(Effect::Io(IoOp::ReportStarted { segment }));
                            }
                        }
                    }
                    self.next_frame(&mut out);
                }
                Err(e) => {
                    self.pushing = None;
                    self.loop_ended(false, Err(e), &mut out);
                }
            },
            Event::Reported { finished, result } => {
                let delivered = match result {
                    Ok(()) => true,
                    Err(error) => {
                        log_error("the playback report", &error);
                        if finished {
                            // Delivery is unknown. Preserve the ordering barrier and held input
                            // until explicit close; a failed callback is not an acknowledgement.
                            self.failed.get_or_insert(RunnerError::Failed(error));
                            self.answer_waiting(&mut out);
                        }
                        false
                    }
                };
                if finished && delivered {
                    self.reporting = false;
                    if let Some(report) = self.reports.pop_front() {
                        self.report_finished(
                            report.segment,
                            report.position,
                            report.interrupted,
                            &mut out,
                        );
                    }
                    self.release_audio(&mut out);
                }
            }
            Event::Close { call } => match self.closing {
                _ if self.closed => out.push(Effect::Answer { call, result: Ok(()) }),
                // a second close while one is under way waits for nothing more: the first answers
                Some(_) => out.push(Effect::Answer { call, result: Ok(()) }),
                None => {
                    self.closing = Some((Closing::Receiver, call));
                    out.push(Effect::Io(IoOp::CloseReceiver));
                }
            },
            Event::ReceiverClosed(result) => {
                if let Some((Closing::Receiver, call)) = self.closing {
                    match result {
                        Ok(()) => {
                            self.closing = Some((Closing::Cancelling, call));
                            out.push(Effect::CancelIo);
                        }
                        // as the reference's `aclose`, which raises before cancelling anything
                        Err(e) => {
                            self.closing = None;
                            out.push(Effect::Answer { call, result: Err(RunnerError::Failed(e)) });
                        }
                    }
                }
            }
            Event::IoCancelled => {
                if let Some((Closing::Cancelling, call)) = self.closing {
                    self.held = None;
                    self.held_audio = None;
                    self.reports.clear();
                    self.reporting = false;
                    self.frame_pull_paused = false;
                    self.pushing = None;
                    self.clears.clear();
                    self.clears_waiting.clear();
                    for pump in [&mut self.read, &mut self.forward] {
                        if *pump != Pump::Done {
                            *pump = Pump::Done;
                            self.failed.get_or_insert(RunnerError::Closed);
                        }
                    }
                    if !self.started {
                        self.failed.get_or_insert(RunnerError::Closed);
                    }
                    self.answer_waiting(&mut out);
                    self.closing = Some((Closing::Media, call));
                    out.push(Effect::Io(IoOp::CloseMedia));
                }
            }
            Event::MediaClosed(result) => {
                if let Some((Closing::Media, call)) = self.closing.take() {
                    self.closed = true;
                    out.push(Effect::Answer { call, result: result.map_err(RunnerError::Failed) });
                }
            }
            // an answer to an op the close cut, or a report that went through
            _ => {}
        }
        out
    }

    /// Start both loops.
    fn begin(&mut self, out: &mut Vec<Effect>) {
        self.started = true;
        self.read = Pump::Busy;
        self.forward = Pump::Busy;
        out.push(Effect::Io(IoOp::OpenAudio));
        out.push(Effect::Io(IoOp::OpenFrames));
        self.answer_waiting(out);
    }

    /// One actual host report at a time, so async delivery cannot reverse segment accounting.
    fn report_finished(
        &mut self,
        segment: AvatarPlaybackSegment,
        position: f64,
        interrupted: bool,
        out: &mut Vec<Effect>,
    ) {
        if self.reporting {
            self.reports.push_back(PendingReport { segment, position, interrupted });
        } else {
            self.reporting = true;
            out.push(Effect::Io(IoOp::ReportFinished { segment, position, interrupted }));
        }
    }

    fn next_frame(&mut self, out: &mut Vec<Effect>) {
        if self.reporting || !self.clears.is_empty() {
            self.frame_pull_paused = true;
        } else {
            out.push(Effect::Io(IoOp::NextFrame));
        }
    }

    fn release_audio(&mut self, out: &mut Vec<Effect>) {
        if self.clears.is_empty() && !self.reporting {
            if let Some(observation) = self.held_audio.take() {
                out.extend(self.handle(
                    Event::Audio(Ok((
                        observation.0,
                        observation.1,
                        observation.2,
                        Some(observation.3),
                    ))),
                    Ts::ZERO,
                ));
            }
            if !self.reporting && self.frame_pull_paused {
                self.frame_pull_paused = false;
                out.push(Effect::Io(IoOp::NextFrame));
            }
        }
    }

    /// `_publish_track`: once the room has connected.
    fn ask_publish(&mut self, out: &mut Vec<Effect>) {
        if self.connected {
            self.publish = Publish::InFlight;
            out.push(Effect::Io(IoOp::Publish));
        } else {
            self.publish = Publish::Waiting;
        }
    }

    /// A generated frame: publish the tracks first if they are not up.
    fn forward(&mut self, frame: MediaFrame, duration: Option<f64>, out: &mut Vec<Effect>) {
        if self.publish == Publish::Done {
            self.push(frame, duration, out);
            return;
        }
        self.forward = Pump::Publishing;
        self.held = Some((frame, duration));
        if self.publish == Publish::Never {
            self.ask_publish(out);
        }
    }

    fn push(&mut self, frame: MediaFrame, duration: Option<f64>, out: &mut Vec<Effect>) {
        self.forward = Pump::Busy;
        self.pushing = duration;
        out.push(Effect::Io(IoOp::Push(frame)));
    }

    /// One loop's end: `read` or forward.
    fn loop_ended(&mut self, read: bool, result: Result<(), ProviderError>, out: &mut Vec<Effect>) {
        let (pump, name) = if read {
            (&mut self.read, "_read_audio")
        } else {
            (&mut self.forward, "_forward_video")
        };
        *pump = Pump::Done;
        match result {
            Ok(()) => self.finished += 1,
            Err(e) => {
                log_error(name, &e);
                self.failed.get_or_insert(RunnerError::Failed(e));
            }
        }
        self.answer_waiting(out);
    }

    /// `wait_for_complete`'s answers, once the outcome is known.
    fn answer_waiting(&mut self, out: &mut Vec<Effect>) {
        let result = match &self.failed {
            Some(e) => Err(e.clone()),
            None if !self.started && self.start_call.is_none() => Err(RunnerError::NotStarted),
            None if self.finished == 2 => Ok(()),
            None => return,
        };
        for call in self.waiting.drain(..) {
            out.push(Effect::Answer { call, result: result.clone() });
        }
    }
}

fn log_error(what: &str, e: &ProviderError) {
    log::error!(target: LOG, error = e.to_string().as_str(); "Error in {what}");
}

#[cfg(test)]
#[path = "avatar_runner_tests.rs"]
mod tests;
