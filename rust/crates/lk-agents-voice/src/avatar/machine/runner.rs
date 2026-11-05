//! `voice/avatar/_runner.py`'s `AvatarRunner`, the avatar worker's side: the agent's audio from an
//! `AudioReceiver` into the user's `VideoGenerator`, what it generates out through the SDK's
//! `rtc.AVSynchronizer` (publishing the tracks on the first frame), and each segment's playback
//! reported back.
//!
//! [`RunnerCore`] makes every decision as a state machine: `handle(event, now) -> effects`, no
//! timers of its own. The frames are plain records ([`AvatarFrame`], [`MediaFrame`]): the core
//! decides where each one goes, never what is in it. The shared runtime queue owner
//! executes each [`AvatarIoOperation`] against the host proxies and answer it with an [`AvatarEvent`].

use crate::avatar::effects::AvatarEffect;
use std::collections::{BTreeMap, VecDeque};

use lk_agents_abi::Ts;
use lk_agents_abi::errors::ProviderError;
pub use lk_agents_abi::utils::video::{AudioReceiverFrame, AvatarFrame, MediaFrame};

const LOG: &str = "livekit.agents";

/// The playback position of a segment cut before any of it played.
const NOT_PLAYED: f64 = 0.0;

// ----------------------------------------------------------------------------- the host's side

pub use lk_agents_abi::voice::avatar::{AvatarAudioReceiver, AvatarMedia, AvatarVideoGenerator};
use lk_agents_abi::voice::io::AvatarPlaybackSegment;

// ----------------------------------------------------------------------------- the core

/// A host call into the runner, answered by [`AvatarEffect::Answer`] with the same id.
pub type CallId = u64;

pub use lk_agents_abi::voice::avatar::RunnerError;

/// What happened: a host call, the room, or the answer to an [`AvatarIoOperation`].
#[derive(Debug, Clone, PartialEq)]
pub enum AvatarEvent {
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
    /// Every [`AvatarIoOperation`] in flight was cancelled ([`AvatarEffect::CancelIo`]).
    IoCancelled,
    MediaClosed(Result<(), ProviderError>),
}

impl AvatarEvent {
    /// An answer to an op the loops issued, or input that feeds them: dropped once the close
    /// cuts the loops.
    fn dropped_once_cut(&self) -> bool {
        matches!(
            self,
            AvatarEvent::RoomFailed(_)
                | AvatarEvent::ClearBuffer { .. }
                | AvatarEvent::GeneratorCleared { .. }
                | AvatarEvent::AudioOpened(_)
                | AvatarEvent::Audio(_)
                | AvatarEvent::AudioPushed(_)
                | AvatarEvent::FramesOpened(_)
                | AvatarEvent::Frame(_)
                | AvatarEvent::Published(_)
                | AvatarEvent::Pushed(_)
        )
    }
}

/// What the runner asks of its receiver, generator and media.
#[derive(Debug, Clone, PartialEq)]
pub enum AvatarIoOperation {
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

/// The runner's two loops, as the reference names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Loop {
    /// `_read_audio`: the receiver's audio into the generator.
    ReadAudio,
    /// `_forward_video`: the generator's frames out to the room.
    ForwardVideo,
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

    /// Whether the loops are being cut: no answer moves them on any more.
    fn cut(&self) -> bool {
        self.closed || matches!(self.closing, Some((Closing::Cancelling | Closing::Media, _)))
    }

    pub fn handle(&mut self, event: AvatarEvent, _now: Ts) -> Vec<AvatarEffect> {
        let mut effects = Vec::new();
        // an answer to an op the close cut
        if self.cut() && event.dropped_once_cut() {
            return effects;
        }
        let out = &mut effects;
        match event {
            AvatarEvent::Start { call, connected } => self.start(call, connected, out),
            AvatarEvent::WaitForComplete { call } => self.wait_for_complete(call, out),
            AvatarEvent::Connection { connected } => self.connection(connected, out),
            AvatarEvent::RoomFailed(error) => self.room_failed(error, out),
            AvatarEvent::ClearBuffer { epoch } => self.clear_buffer(epoch, out),
            AvatarEvent::GeneratorCleared { clear, result } => {
                self.generator_cleared(clear, result, out)
            }
            AvatarEvent::AudioOpened(result) => self.audio_opened(result, out),
            AvatarEvent::Audio(result) => self.audio_received(result, out),
            AvatarEvent::AudioPushed(result) => self.audio_pushed(result, out),
            AvatarEvent::FramesOpened(result) => self.frames_opened(result, out),
            AvatarEvent::Frame(result) => self.frame_received(result, out),
            AvatarEvent::Published(result) => self.published(result, out),
            AvatarEvent::Pushed(result) => self.pushed(result, out),
            AvatarEvent::Reported { finished, result } => self.reported(finished, result, out),
            AvatarEvent::Close { call } => self.close(call, out),
            AvatarEvent::ReceiverClosed(result) => self.receiver_closed(result, out),
            AvatarEvent::IoCancelled => self.io_cancelled(out),
            AvatarEvent::MediaClosed(result) => self.media_closed(result, out),
        }
        effects
    }

    fn start(&mut self, call: CallId, connected: bool, out: &mut Vec<AvatarEffect>) {
        self.connected |= connected;
        if self.lazy_publish {
            self.begin(out);
            out.push(AvatarEffect::Answer { call, result: Ok(()) });
        } else {
            self.start_call = Some(call);
            self.ask_publish(out);
        }
    }

    fn wait_for_complete(&mut self, call: CallId, out: &mut Vec<AvatarEffect>) {
        self.waiting.push(call);
        self.answer_waiting(out);
    }

    fn connection(&mut self, connected: bool, out: &mut Vec<AvatarEffect>) {
        if !connected || self.connected {
            return;
        }
        self.connected = true;
        if self.publish == Publish::Waiting {
            self.publish = Publish::InFlight;
            out.push(AvatarEffect::Io(AvatarIoOperation::Publish));
        }
    }

    fn room_failed(&mut self, error: ProviderError, out: &mut Vec<AvatarEffect>) {
        let error = RunnerError::Failed(error);
        self.failed.get_or_insert(error.clone());
        if let Some(call) = self.start_call.take() {
            out.push(AvatarEffect::Answer { call, result: Err(error) });
        }
        self.answer_waiting(out);
    }

    /// The receiver's `clear_buffer`: a stale epoch was already cleared.
    fn clear_buffer(&mut self, epoch: u64, out: &mut Vec<AvatarEffect>) {
        if epoch <= self.clear_epoch {
            return;
        }
        self.clear_epoch = epoch;
        let clear = self.next_clear;
        self.next_clear += 1;
        self.clears.insert(clear, Clear { epoch, owed: std::mem::take(&mut self.owed) });
        self.stale_end = std::mem::take(&mut self.reading);
        if self.audio_pushing {
            self.clears_waiting.push(clear);
        } else {
            out.push(AvatarEffect::Io(AvatarIoOperation::ClearGenerator { clear }));
        }
    }

    fn generator_cleared(
        &mut self,
        clear: u64,
        result: Result<(), ProviderError>,
        out: &mut Vec<AvatarEffect>,
    ) {
        let owed = self.clears.remove(&clear).map_or_else(VecDeque::new, |clear| clear.owed);
        if let Err(error) = result {
            log_error("_handle_clear_buffer", &error);
        }
        // The receiver has cut this input even if the generator could not clear.
        let interrupted = true;
        for segment in owed {
            let position = std::mem::take(&mut self.position);
            self.report_finished(segment, position, interrupted, out);
        }
        self.release_audio(out);
    }

    fn audio_opened(&mut self, result: Result<(), ProviderError>, out: &mut Vec<AvatarEffect>) {
        match result {
            Ok(()) => out.push(AvatarEffect::Io(AvatarIoOperation::NextAudio)),
            Err(e) => self.loop_ended(Loop::ReadAudio, Err(e), out),
        }
    }

    /// One read from the receiver. The arm order is the decision: stale epoch, stale segment
    /// end, held while clearing or reporting, no segment identity, push, end.
    fn audio_received(
        &mut self,
        result: Result<
            (u64, bool, Option<AvatarPlaybackSegment>, Option<AudioReceiverFrame>),
            ProviderError,
        >,
        out: &mut Vec<AvatarEffect>,
    ) {
        match result {
            Ok((epoch, starts_segment, segment, Some(_))) if epoch < self.clear_epoch => {
                if starts_segment && let Some(segment) = segment {
                    self.owe_stale_segment(epoch, segment, out);
                }
                out.push(AvatarEffect::Io(AvatarIoOperation::NextAudio));
            }
            Ok((_, _, _, Some(AudioReceiverFrame::SegmentEnd)))
                if std::mem::take(&mut self.stale_end) =>
            {
                out.push(AvatarEffect::Io(AvatarIoOperation::NextAudio));
            }
            Ok((epoch, starts_segment, segment, Some(item)))
                if !self.clears.is_empty() || self.reporting =>
            {
                self.held_audio = Some((epoch, starts_segment, segment, item));
            }
            Ok((_, _, None, Some(_))) => {
                let error =
                    lk_agents_abi::errors::ApiError::new("avatar audio has no segment identity");
                self.loop_ended(Loop::ReadAudio, Err(error.into()), out);
            }
            Ok((_, _, Some(segment), Some(item))) => self.push_audio(segment, item, out),
            Ok((_, _, _, None)) => self.loop_ended(Loop::ReadAudio, Ok(()), out),
            Err(e) => self.loop_ended(Loop::ReadAudio, Err(e), out),
        }
    }

    /// A segment read before the clear that cut it: the clear in flight owes its finish, else
    /// it is reported finished now.
    fn owe_stale_segment(
        &mut self,
        epoch: u64,
        segment: AvatarPlaybackSegment,
        out: &mut Vec<AvatarEffect>,
    ) {
        let (position, interrupted) = (NOT_PLAYED, true);
        match self.clears.values_mut().find(|clear| clear.epoch > epoch) {
            Some(clear) => clear.owed.push_back(segment),
            None => self.report_finished(segment, position, interrupted, out),
        }
    }

    fn push_audio(
        &mut self,
        segment: AvatarPlaybackSegment,
        item: AudioReceiverFrame,
        out: &mut Vec<AvatarEffect>,
    ) {
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
        out.push(AvatarEffect::Io(AvatarIoOperation::PushAudio(frame)));
    }

    fn audio_pushed(&mut self, result: Result<(), ProviderError>, out: &mut Vec<AvatarEffect>) {
        self.audio_pushing = false;
        for clear in std::mem::take(&mut self.clears_waiting) {
            out.push(AvatarEffect::Io(AvatarIoOperation::ClearGenerator { clear }));
        }
        match result {
            Ok(()) => out.push(AvatarEffect::Io(AvatarIoOperation::NextAudio)),
            Err(e) => self.loop_ended(Loop::ReadAudio, Err(e), out),
        }
    }

    fn frames_opened(&mut self, result: Result<(), ProviderError>, out: &mut Vec<AvatarEffect>) {
        match result {
            Ok(()) => self.next_frame(out),
            Err(e) => self.loop_ended(Loop::ForwardVideo, Err(e), out),
        }
    }

    fn frame_received(
        &mut self,
        result: Result<Option<AvatarFrame>, ProviderError>,
        out: &mut Vec<AvatarEffect>,
    ) {
        match result {
            Ok(Some(AvatarFrame::SegmentEnd)) => {
                // the agent hears the segment finished; the forwarding goes on
                if let Some(segment) = self.owed.pop_front() {
                    let position = std::mem::take(&mut self.position);
                    let interrupted = false;
                    self.report_finished(segment, position, interrupted, out);
                }
                self.next_frame(out);
            }
            Ok(Some(AvatarFrame::Audio(frame))) => {
                let duration = frame.duration();
                self.forward(MediaFrame::Audio(frame), Some(duration), out)
            }
            Ok(Some(AvatarFrame::Video(frame))) => {
                self.forward(MediaFrame::Video(frame), None, out)
            }
            Ok(None) => self.loop_ended(Loop::ForwardVideo, Ok(()), out),
            Err(e) => self.loop_ended(Loop::ForwardVideo, Err(e), out),
        }
    }

    fn published(&mut self, result: Result<(), ProviderError>, out: &mut Vec<AvatarEffect>) {
        self.publish = if result.is_ok() { Publish::Done } else { Publish::Never };
        if let Some(call) = self.start_call.take() {
            self.answer_start(call, result, out);
        } else if self.forward == Pump::Publishing {
            self.resume_held_frame(result, out);
        }
    }

    /// `start()` waited on the publish.
    fn answer_start(
        &mut self,
        call: CallId,
        result: Result<(), ProviderError>,
        out: &mut Vec<AvatarEffect>,
    ) {
        match result {
            Ok(()) => {
                self.begin(out);
                out.push(AvatarEffect::Answer { call, result: Ok(()) });
            }
            Err(e) => out.push(AvatarEffect::Answer { call, result: Err(RunnerError::Failed(e)) }),
        }
    }

    /// The forward loop held its frame while the tracks published.
    fn resume_held_frame(
        &mut self,
        result: Result<(), ProviderError>,
        out: &mut Vec<AvatarEffect>,
    ) {
        match result {
            Ok(()) => {
                if let Some((frame, duration)) = self.held.take() {
                    self.push(frame, duration, out);
                }
            }
            Err(e) => {
                self.held = None;
                self.loop_ended(Loop::ForwardVideo, Err(e), out);
            }
        }
    }

    fn pushed(&mut self, result: Result<(), ProviderError>, out: &mut Vec<AvatarEffect>) {
        if let Err(e) = result {
            self.pushing = None;
            self.loop_ended(Loop::ForwardVideo, Err(e), out);
            return;
        }
        if let Some(duration) = self.pushing.take() {
            let started = self.position == NOT_PLAYED && duration > 0.0;
            self.position += duration;
            if started && let Some(segment) = self.owed.front().cloned() {
                out.push(AvatarEffect::Io(AvatarIoOperation::ReportStarted { segment }));
            }
        }
        self.next_frame(out);
    }

    fn reported(
        &mut self,
        finished: bool,
        result: Result<(), ProviderError>,
        out: &mut Vec<AvatarEffect>,
    ) {
        if let Err(error) = result {
            log_error("the playback report", &error);
            if finished {
                // Delivery is unknown. Preserve the ordering barrier and held input
                // until explicit close; a failed callback is not an acknowledgement.
                self.failed.get_or_insert(RunnerError::Failed(error));
                self.answer_waiting(out);
            }
            return;
        }
        if !finished {
            return;
        }
        self.reporting = false;
        if let Some(report) = self.reports.pop_front() {
            self.report_finished(report.segment, report.position, report.interrupted, out);
        }
        self.release_audio(out);
    }

    fn close(&mut self, call: CallId, out: &mut Vec<AvatarEffect>) {
        match self.closing {
            _ if self.closed => out.push(AvatarEffect::Answer { call, result: Ok(()) }),
            // a second close while one is under way waits for nothing more: the first answers
            Some(_) => out.push(AvatarEffect::Answer { call, result: Ok(()) }),
            None => {
                self.closing = Some((Closing::Receiver, call));
                out.push(AvatarEffect::Io(AvatarIoOperation::CloseReceiver));
            }
        }
    }

    fn receiver_closed(&mut self, result: Result<(), ProviderError>, out: &mut Vec<AvatarEffect>) {
        let Some((Closing::Receiver, call)) = self.closing else {
            return;
        };
        match result {
            Ok(()) => {
                self.closing = Some((Closing::Cancelling, call));
                out.push(AvatarEffect::CancelIo);
            }
            // as the reference's `aclose`, which raises before cancelling anything
            Err(e) => {
                self.closing = None;
                out.push(AvatarEffect::Answer { call, result: Err(RunnerError::Failed(e)) });
            }
        }
    }

    fn io_cancelled(&mut self, out: &mut Vec<AvatarEffect>) {
        let Some((Closing::Cancelling, call)) = self.closing else {
            return;
        };
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
        self.answer_waiting(out);
        self.closing = Some((Closing::Media, call));
        out.push(AvatarEffect::Io(AvatarIoOperation::CloseMedia));
    }

    fn media_closed(&mut self, result: Result<(), ProviderError>, out: &mut Vec<AvatarEffect>) {
        if let Some((Closing::Media, call)) = self.closing.take() {
            self.closed = true;
            out.push(AvatarEffect::Answer { call, result: result.map_err(RunnerError::Failed) });
        }
    }

    /// Start both loops.
    fn begin(&mut self, out: &mut Vec<AvatarEffect>) {
        self.started = true;
        self.read = Pump::Busy;
        self.forward = Pump::Busy;
        out.push(AvatarEffect::Io(AvatarIoOperation::OpenAudio));
        out.push(AvatarEffect::Io(AvatarIoOperation::OpenFrames));
        self.answer_waiting(out);
    }

    /// One actual host report at a time, so async delivery cannot reverse segment accounting.
    fn report_finished(
        &mut self,
        segment: AvatarPlaybackSegment,
        position: f64,
        interrupted: bool,
        out: &mut Vec<AvatarEffect>,
    ) {
        if self.reporting {
            self.reports.push_back(PendingReport { segment, position, interrupted });
        } else {
            self.reporting = true;
            out.push(AvatarEffect::Io(AvatarIoOperation::ReportFinished {
                segment,
                position,
                interrupted,
            }));
        }
    }

    fn next_frame(&mut self, out: &mut Vec<AvatarEffect>) {
        if self.reporting || !self.clears.is_empty() {
            self.frame_pull_paused = true;
        } else {
            out.push(AvatarEffect::Io(AvatarIoOperation::NextFrame));
        }
    }

    fn release_audio(&mut self, out: &mut Vec<AvatarEffect>) {
        if self.clears.is_empty() && !self.reporting {
            if let Some(observation) = self.held_audio.take() {
                out.extend(self.handle(
                    AvatarEvent::Audio(Ok((
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
                out.push(AvatarEffect::Io(AvatarIoOperation::NextFrame));
            }
        }
    }

    /// `_publish_track`: once the room has connected.
    fn ask_publish(&mut self, out: &mut Vec<AvatarEffect>) {
        if self.connected {
            self.publish = Publish::InFlight;
            out.push(AvatarEffect::Io(AvatarIoOperation::Publish));
        } else {
            self.publish = Publish::Waiting;
        }
    }

    /// A generated frame: publish the tracks first if they are not up.
    fn forward(&mut self, frame: MediaFrame, duration: Option<f64>, out: &mut Vec<AvatarEffect>) {
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

    fn push(&mut self, frame: MediaFrame, duration: Option<f64>, out: &mut Vec<AvatarEffect>) {
        self.forward = Pump::Busy;
        self.pushing = duration;
        out.push(AvatarEffect::Io(AvatarIoOperation::Push(frame)));
    }

    /// One loop's end.
    fn loop_ended(
        &mut self,
        ended: Loop,
        result: Result<(), ProviderError>,
        out: &mut Vec<AvatarEffect>,
    ) {
        let (pump, name) = match ended {
            Loop::ReadAudio => (&mut self.read, "_read_audio"),
            Loop::ForwardVideo => (&mut self.forward, "_forward_video"),
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
    fn answer_waiting(&mut self, out: &mut Vec<AvatarEffect>) {
        let result = match &self.failed {
            Some(e) => Err(e.clone()),
            None if !self.started && self.start_call.is_none() => Err(RunnerError::NotStarted),
            None if self.finished == 2 => Ok(()),
            None => return,
        };
        for call in self.waiting.drain(..) {
            out.push(AvatarEffect::Answer { call, result: result.clone() });
        }
    }
}

fn log_error(what: &str, e: &ProviderError) {
    log::error!(target: LOG, error = e.to_string().as_str(); "Error in {what}");
}

#[cfg(test)]
#[path = "runner_tests.rs"]
mod tests;
