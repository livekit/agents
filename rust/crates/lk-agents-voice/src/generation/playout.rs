//! `perform_audio_forwarding`: one segment's frames pushed to the sink, and what the sink reports.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::DriverError;
use crate::agent_session::SessionShared;
use crate::generation::GenEvent;
use crate::generation::TurnAccumulator;
use crate::speech_handle::SpeechHandle;
use crate::trace::{PlayoutStep, SinkCall};
use crate::transcription::Heard;
use lk_agents_abi::AudioFrame;
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputExt, PlaybackEvent, PlaybackFinished, PlaybackObserver, PlaybackSegment,
};
use lk_agents_abi::voice::rtc::frame_duration;
use lk_agents_media::{FrameResampler, OnChange, Quality};
use lk_agents_voice::transcription::Visible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// A pushed frame's length when none has been pushed yet: what every TTS in the tree emits.
const DEFAULT_FRAME_LEN: Duration = Duration::from_millis(20);

/// Scheduler hops a cut segment gives the sink to report the cut. Hops, not time: a speech cut at
/// `Closing` has no clock advancing it.
const CUT_REPORT_HOPS: usize = 16;

/// How long a cut segment then waits on the clock for a host output's report of the cut.
const CUT_REPORT_WAIT: Duration = Duration::from_secs(1);

/// How late a sink's finish may be, once it reported all the audio played, before the playout
/// takes it as lost. The reference's `DataStreamAudioOutput(clear_buffer_timeout=2.0)`: its one
/// sink whose reports cross a transport marks the playout done after this "to avoid deadlock".
const LOST_FINISH_MARGIN: Duration = Duration::from_secs(2);

#[derive(Debug)]
pub(crate) enum Next {
    Frame(AudioFrame),
    /// Flush and keep playing (a realtime `MessageDone`).
    Boundary,
    End,
}

/// A source of one segment's frames. `next` is never raced (it may be a host generator that
/// dropping does not cancel); interruption is checked between frames, and the push is raced.
pub(crate) trait Segment {
    /// Realtime messages settle before the next message acquires its output binding.
    fn stop_at_boundary(&self) -> bool {
        false
    }

    fn next(&mut self) -> impl std::future::Future<Output = Result<Next, DriverError>> + Send;

    /// Independent control events may progress while a flushed segment awaits its receipt.
    /// This future may be dropped at any poll and must not race a host generator's `next`.
    fn playout_controls(&mut self) -> impl std::future::Future<Output = ()> + Send {
        std::future::pending()
    }

    fn text(&self) -> String;

    /// The sink reported this segment's first frame leaving for the speaker (not the first push:
    /// a paused sink plays nothing). `at` is the sink's own instant, in wall seconds. Bookkeeping
    /// only; the `speaking` edge is `GenEvent::PlaybackStarted`'s.
    fn started_speaking(&mut self, at: f64);
}

/// Poll only a source's cancellation-safe control path beside the native playback wait.
async fn with_playout_controls<S: Segment, T: Send>(
    source: &mut S,
    work: impl std::future::Future<Output = T> + Send,
) -> T {
    let mut controls = std::pin::pin!(source.playout_controls());
    let mut work = std::pin::pin!(work);
    let mut controls_done = false;
    std::future::poll_fn(|cx| {
        if !controls_done {
            controls_done = controls.as_mut().poll(cx).is_ready();
        }
        work.as_mut().poll(cx)
    })
    .await
}

/// Once per turn; not gated on interruption, since nothing is pushed for a speech cut before its
/// first frame. Bookkeeping only: the `speaking` edge is [`GenEvent::PlaybackStarted`]'s.
pub(crate) fn first_frame(shared: &Arc<SessionShared>, acc: &mut TurnAccumulator, at: f64) {
    acc.on_first_frame(Some(at), shared.time.wall_secs());
}

/// A segment whose source failed: the error, and the synchroniser's words up to where its audio
/// played, when some did and a synchroniser reported them.
pub(crate) struct SourceFailed {
    pub(crate) error: DriverError,
    pub(crate) synchronized: Option<String>,
    pub(crate) heard: Option<Heard>,
}

impl From<DriverError> for SourceFailed {
    fn from(error: DriverError) -> Self {
        SourceFailed { error, synchronized: None, heard: None }
    }
}

#[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
#[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
/// `perform_audio_forwarding` for one segment. `out: None` drains and drops the frames, reporting
/// nothing. The clean path ends with `flush`, then waits for the sink to report played out.
pub(crate) async fn perform_audio_forwarding<S: Segment>(
    shared: &Arc<SessionShared>,
    handle: &SpeechHandle,
    out: Option<&dyn AudioOutput>,
    segment: &mut S,
) -> Result<Heard, SourceFailed> {
    let Some(out) = out else {
        // Raced like the sink path's `watched`: a cut source may never end its segment.
        let complete = loop {
            let next = tokio::select! {
                biased;
                () = handle.wait_for_interrupted() => break false,
                next = segment.next() => next?,
            };
            match next {
                Next::Frame(_) => {}
                Next::Boundary if segment.stop_at_boundary() => break true,
                Next::Boundary => {}
                Next::End => break true,
            }
        };
        return Ok(silent(segment.text(), complete));
    };
    // Opened before the first push, so nothing the sink says about this segment is missed.
    let (tx, events) = mpsc::unbounded_channel();
    out.tracker().observe(Arc::new(SinkReports(tx)));
    let mut playing = Playing {
        segment_key: out.tracker().current_segment(),
        finish_due: None,
        shared,
        handle,
        out,
        events,
        chain: OutputChain::new(out.sample_rate()),
        started: false,
        owed: None,
        pushed: 0,
        pushed_len: Duration::ZERO,
        played: Duration::ZERO,
        completed: Duration::ZERO,
        reported: 0,
        batch: 0,
        first_segment: None,
        cut_below: false,
    };
    let mut ended = loop {
        if handle.interrupted() {
            break Ended::Cut;
        }
        // The next frame or the interruption, folding sink reports meanwhile: a slow provider
        // must not delay the cut, and a pushed frame's `PlaybackStarted` must not wait for the
        // next frame. Dropping `next()` here is what interruption means for it.
        let next = playing.watched(segment.next()).await;
        playing.settle(segment);
        let frame = match next {
            None => break Ended::Cut,
            Some(Ok(Next::Frame(frame))) => frame,
            Some(Ok(Next::Boundary)) if segment.stop_at_boundary() => break Ended::Audio,
            Some(Ok(Next::Boundary)) => {
                if handle.interrupted() {
                    break Ended::Cut;
                }
                // The stretch so far is a whole: its tail, its push record, its flush.
                if let Err(e) = playing.push_tail(segment).await {
                    break e;
                }
                playing.report_push();
                playing.flush_batch();
                continue;
            }
            Some(Ok(Next::End)) => break Ended::Audio,
            Some(Err(e)) => break Ended::Provider(e),
        };
        // The frame may have taken a while to arrive: a speech cut meanwhile never starts playing.
        // The launch hold: the speech's first frame waits while the user speaks.
        if handle.interrupted() || (playing.pushed == 0 && !wait_launch(shared, handle).await) {
            break Ended::Cut;
        }
        playing.drain(segment);
        let frames = match playing.chain.push(&frame) {
            Ok(frames) => frames,
            Err(e) => break Ended::Provider(e),
        };
        if let Err(e) = playing.push(segment, frames).await {
            break e;
        }
    };
    // The resampler's tail belongs to the segment on the clean path only.
    if let Ended::Audio = ended
        && let Err(e) = playing.push_tail(segment).await
    {
        ended = e;
    }
    playing.report_push();
    let text = segment.text();
    match ended {
        Ended::Cut => Ok(playing.cut(segment, text).await),
        Ended::Provider(e) => {
            // What was synthesised plays out before the caller forwards the rest as text: the
            // reference waits for the playout whether or not the TTS failed (`forward_generation`).
            let mut synchronized = None;
            if playing.pushed > 0 {
                playing.flush_batch();
                match with_playout_controls(segment, playing.played_out_report()).await {
                    Ok(Some(ev)) => {
                        playing.played_out(segment, &ev);
                        synchronized = ev.synchronized_transcript.map(|t| Visible::of(&t).text);
                    }
                    Ok(None) | Err(_) => {
                        playing.cut(segment, text).await;
                    }
                }
            }
            Err(SourceFailed { error: e, synchronized, heard: Some(playing.confirmed_heard()) })
        }
        // No audio: flushing would make `wait_for_playout` answer with the previous event.
        Ended::Audio if playing.pushed == 0 => Ok(silent(text, true)),
        Ended::Audio => {
            playing.flush_batch();
            let ev = match with_playout_controls(segment, playing.played_out_report()).await {
                Ok(Some(ev)) => ev,
                Ok(None) => return Ok(playing.cut(segment, text).await),
                Err(error) => {
                    return Err(SourceFailed {
                        error,
                        synchronized: None,
                        heard: Some(playing.confirmed_heard()),
                    });
                }
            };
            playing.played_out(segment, &ev);
            // A synchroniser's words up to the played position, not the words sent.
            Ok(Heard {
                text: synchronized(ev.synchronized_transcript, text),
                position: ev.playback_position,
                complete: !ev.interrupted && !playing.cut_below,
                frames: playing.pushed,
                played: playing.completed.max(ev.playback_position),
            })
        }
    }
}

fn synchronized(heard: Option<String>, text: String) -> String {
    heard.map(|heard| Visible::of(&heard).text).unwrap_or(text)
}

#[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
/// `false` when interrupted while parked. Every source and every sink goes through here.
async fn wait_launch(shared: &Arc<SessionShared>, handle: &SpeechHandle) -> bool {
    let speech = handle.id();
    let mut level = shared.launch.subscribe();
    if !level.borrow_and_update().contains(&speech) {
        return true;
    }
    shared.trace(crate::trace::InterruptStep::PlayoutGate {
        speech,
        op: crate::agent_session::effects::GateOp::Pause,
        cause: crate::agent_session::effects::GateCause::Launch,
    });
    loop {
        tokio::select! {
            biased;
            () = handle.wait_for_interrupted() => return false,
            changed = level.changed() => {
                // A closed level (the loop is gone) holds nothing.
                if changed.is_err() || !level.borrow_and_update().contains(&speech) {
                    shared.report(GenEvent::LaunchReleased { speech });
                    return true;
                }
            }
        }
    }
}

/// `flush()` then `clear_buffer()`: a cut segment reports `Clear`, never `Flush`+`End`. With
/// nothing pushed since the last flush there is no segment to close, and a second flush would end
/// a synchronised segment that has none of its audio.
fn clear_sink(shared: &Arc<SessionShared>, speech: SpeechId, out: &dyn AudioOutput, open: bool) {
    shared.trace(PlayoutStep::SinkCalled { speech: Some(speech), call: SinkCall::Clear });
    if open { out.cut() } else { out.clear_buffer() }
}

/// Built at the first off-rate frame. A sink that takes any rate gets the frames untouched. A rate
/// or channel change mid-segment drains the stream so far; the new one starts clean.
struct OutputChain(Option<FrameResampler>);

impl OutputChain {
    fn new(target: Option<u32>) -> Self {
        OutputChain(target.map(|rate| FrameResampler::new(rate, Quality::Medium, OnChange::Drain)))
    }

    fn push(&mut self, frame: &AudioFrame) -> Result<Vec<AudioFrame>, DriverError> {
        let Some(chain) = self.0.as_mut() else { return Ok(vec![frame.clone()]) };
        chain.push(frame).map_err(|e| DriverError::Resample(e.to_string()))
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, DriverError> {
        let Some(chain) = self.0.as_mut() else { return Ok(Vec::new()) };
        let tail = chain.flush().map_err(|e| DriverError::Resample(e.to_string()))?;
        Ok(tail.into_iter().collect())
    }
}

pub(super) fn silent(text: String, complete: bool) -> Heard {
    Heard { text, position: Duration::ZERO, complete, frames: 0, played: Duration::ZERO }
}

pub(super) fn unheard() -> Heard {
    silent(String::new(), false)
}

/// What the sink reported for this segment, on one channel, in order. `Shown`: a synchroniser
/// below released a word as of the played position; journalled, and the playout wakes at the
/// release, ahead of the words' forwarder, so a finished segment's stream closes before its words
/// reach the text output. `Reset`: the sink's playback count was reset under it. `Finished`
/// carries the tracker's number for the segment it closed.
enum SinkReport {
    Playback(PlaybackEvent),
    Finished { segment: PlaybackSegment, ev: PlaybackFinished },
    Shown(lk_agents_abi::TimedString),
    Reset,
}

#[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
struct SinkReports(mpsc::UnboundedSender<SinkReport>);

impl PlaybackObserver for SinkReports {
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        let _ = self.0.send(SinkReport::Playback(ev.clone()));
    }

    fn on_segment_finished(&self, segment: PlaybackSegment, ev: &PlaybackFinished) {
        let _ = self.0.send(SinkReport::Finished { segment, ev: ev.clone() });
    }

    fn on_transcript_shown(&self, word: &lk_agents_abi::TimedString) {
        let _ = self.0.send(SinkReport::Shown(word.clone()));
    }

    fn on_playback_reset(&self) {
        let _ = self.0.send(SinkReport::Reset);
    }

    fn is_closed(&self) -> bool {
        self.0.is_closed()
    }
}

/// A complete segment that pushed no audio but forwarded words: its audio leg ends with none, so
/// a synchroniser shows the words whole and pairs the next segment's audio with the next
/// segment's words. Nothing plays, so nothing is waited for.
pub(super) fn end_unvoiced(shared: &Arc<SessionShared>, speech: SpeechId, out: &dyn AudioOutput) {
    let shown: Arc<dyn PlaybackObserver> =
        Arc::new(ShownUnvoiced { shared: Arc::clone(shared), speech });
    out.tracker().observe(Arc::clone(&shown));
    out.flush();
    out.tracker().unobserve(&shown);
}

/// Journals the words a synchroniser shows as an unvoiced segment ends. With no audio there is no
/// playhead to be ahead of: each word counts as reached where it starts.
struct ShownUnvoiced {
    shared: Arc<SessionShared>,
    speech: SpeechId,
}

impl PlaybackObserver for ShownUnvoiced {
    fn on_playback_event(&self, _ev: &PlaybackEvent) {}

    fn on_transcript_shown(&self, word: &lk_agents_abi::TimedString) {
        self.shared.trace(PlayoutStep::TranscriptShown {
            speech: self.speech,
            text: word.text.clone(),
            start: word.start_time.unwrap_or_default(),
            end: word.end_time.unwrap_or_default(),
            played: Duration::from_secs_f64(word.start_time.unwrap_or_default().max(0.0)),
        });
    }
}

/// How the push loop ended.
enum Ended {
    Audio,
    /// Interrupted, or the sink failed: the track is gone, and pushing into it helps nobody.
    Cut,
    Provider(DriverError),
}

/// One segment on its way to the sink. Never off a push. The segment is not held here, so its
/// `next()` can be pending while reports are folded (`watched`) and settled once it is free
/// (`settle`).
struct Playing<'a> {
    segment_key: Option<PlaybackSegment>,
    finish_due: Option<lk_agents_abi::Ts>,
    shared: &'a Arc<SessionShared>,
    handle: &'a SpeechHandle,
    out: &'a dyn AudioOutput,
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    events: mpsc::UnboundedReceiver<SinkReport>,
    chain: OutputChain,
    started: bool,
    /// The first-frame instant owed to the segment's `started_speaking`, until it is free.
    owed: Option<f64>,
    /// The frame length `AudioPlayed` counts in.
    pushed: u64,
    pushed_len: Duration,
    played: Duration,
    /// What the segments the sink reported played out held: the forwarding's heard audio before
    /// the segment now playing.
    completed: Duration,
    reported: u64,
    /// Pushed since the last boundary: one `SinkCall::Push` record per stretch.
    batch: u64,
    /// The tracker's number for this forwarding's first segment, once a frame opened it.
    first_segment: Option<PlaybackSegment>,
    /// The sink reported one of this forwarding's segments cut under it (a reset): not all of
    /// it played, whatever the last segment reports.
    cut_below: bool,
}

impl Playing<'_> {
    fn trace(&self, step: PlayoutStep) {
        self.shared.trace(step);
    }

    fn sink_call(&self, call: SinkCall) {
        self.trace(PlayoutStep::SinkCalled { speech: Some(self.handle.id()), call });
    }

    fn finished(&self, position: Duration, interrupted: bool) {
        let speech = self.handle.id();
        self.trace(PlayoutStep::PlaybackFinished { speech, position, interrupted });
    }

    /// The sink reported the segment played out; its `Finished` was emitted before the wait
    /// resolved.
    fn played_out<S: Segment>(&mut self, segment: &mut S, ev: &PlaybackFinished) {
        self.drain(segment);
        self.sink_call(SinkCall::End);
        // Cut before its first frame left: never started, so no finish. Journalled as the sink
        // reported it: a segment cut below earlier makes the speech incomplete, not this finish.
        if self.started || !(ev.interrupted && ev.playback_position.is_zero()) {
            self.finished(ev.playback_position, ev.interrupted);
        }
    }

    /// The confirmed lower bound at a failed report boundary. The physical operation remains
    /// pending; this value neither clears the leaf nor credits unreported audio or words.
    fn confirmed_heard(&mut self) -> Heard {
        self.drain_reports();
        Heard {
            text: String::new(),
            position: self.played.saturating_sub(self.completed),
            complete: false,
            frames: self.pushed,
            played: self.played,
        }
    }

    fn report_push(&self) {
        if self.batch > 0 {
            self.sink_call(SinkCall::Push { frames: self.batch });
        }
    }

    fn flush_batch(&mut self) {
        if std::mem::take(&mut self.batch) > 0 {
            self.sink_call(SinkCall::Flush);
            self.out.flush();
            let remaining = self.pushed_len.saturating_sub(self.played);
            let at =
                self.shared.time.now().saturating_add(remaining).saturating_add(LOST_FINISH_MARGIN);
            self.finish_due = Some(at);
            self.trace(PlayoutStep::FinishDue { speech: self.handle.id(), at });
        }
    }

    fn on_event(&mut self, report: SinkReport) {
        let speech = self.handle.id();
        match report {
            SinkReport::Shown(word) => self.trace(PlayoutStep::TranscriptShown {
                speech,
                text: word.text,
                start: word.start_time.unwrap_or_default(),
                end: word.end_time.unwrap_or_default(),
                played: self.played,
            }),
            SinkReport::Reset => {
                self.trace(PlayoutStep::SinkReset { speech });
            }
            SinkReport::Playback(PlaybackEvent::Started(s)) => {
                if self.segment_key.is_some_and(|key| key != s.segment) {
                    return;
                }
                let time = &self.shared.time;
                let at = time.wall_secs()
                    - time.now().saturating_duration_since(s.created_at).as_secs_f64();
                self.start(at);
            }
            SinkReport::Playback(PlaybackEvent::Progressed(p)) => {
                if self.segment_key.is_some_and(|key| key != p.segment) {
                    return;
                }
                let (offset, duration) = (p.offset, p.duration);
                self.trace(PlayoutStep::PlaybackProgressed { speech, offset, duration });
                self.played += p.duration;
                self.report_played();
            }
            // Started/progressed only: a tracker reports finishes with their segment.
            SinkReport::Playback(PlaybackEvent::Finished { .. }) => {}
            SinkReport::Finished { segment, ev: f } => {
                if self.segment_key.is_some_and(|key| key != segment) {
                    return;
                }
                if f.interrupted && self.first_segment.is_some_and(|first| segment == first) {
                    self.cut_below = true;
                }
                // A sink with no `Started` (a bare recorder) still played it, unless cut at zero.
                if !(f.interrupted && f.playback_position.is_zero()) {
                    self.start(self.shared.time.wall_secs());
                }
                self.played = self.played.max(f.playback_position);
                if f.interrupted {
                } else {
                    self.completed += f.playback_position;
                }
                self.report_played();
            }
        }
    }

    fn start(&mut self, at: f64) {
        if !std::mem::replace(&mut self.started, true) {
            self.shared.report(GenEvent::PlaybackStarted { speech: self.handle.id() });
            self.owed = Some(at);
        }
    }

    fn report_played(&mut self) {
        let len =
            if self.pushed == 0 { DEFAULT_FRAME_LEN } else { self.pushed_len / self.pushed as u32 };
        let len = len.as_nanos().max(1);
        let total = ((self.played.as_nanos() + len / 2) / len) as u64;
        if total > self.reported {
            let frames = total - self.reported;
            self.trace(PlayoutStep::AudioPlayed { speech: self.handle.id(), frames });
            self.reported = total;
        }
    }

    /// Now that nothing borrows it.
    fn settle<S: Segment>(&mut self, segment: &mut S) {
        if let Some(at) = self.owed.take() {
            segment.started_speaking(at);
        }
    }

    fn drain<S: Segment>(&mut self, segment: &mut S) {
        while let Ok(ev) = self.events.try_recv() {
            self.on_event(ev);
        }
        self.settle(segment);
    }

    #[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
    /// `fut`, folding queued sink reports before the operation they may explain.
    async fn watched<T>(&mut self, fut: impl std::future::Future<Output = T>) -> Option<T> {
        tokio::pin!(fut);
        let mut listening = true;
        loop {
            tokio::select! {
                biased;
                () = self.handle.wait_for_interrupted() => return None,
                ev = self.events.recv(), if listening => match ev {
                    Some(ev) => {
                        self.on_event(ev);
                    }
                    None => listening = false,
                },
                v = &mut fut => return Some(v),
            }
        }
    }

    /// A missing report is bounded from the flush by delivered audio plus the transport
    /// allowance. Expiry reports uncertainty; it never manufactures heard audio or a leaf finish.
    async fn played_out_report(&mut self) -> Result<Option<PlaybackFinished>, DriverError> {
        let out = self.out;
        let time = self.shared.time.clone();
        let due = self.finish_due.expect("accepted audio was flushed before awaiting playback");
        let left = due.saturating_duration_since(time.now());
        match self.watched(time.timeout(left, out.wait_for_playout())).await {
            None => Ok(None),
            Some(Ok(event)) => Ok(Some(event)),
            Some(Err(_)) => {
                let position = self.played.saturating_sub(self.completed);
                self.trace(PlayoutStep::FinishLost {
                    speech: self.handle.id(),
                    segment: self.segment_key.expect("captured audio has a playback key"),
                    position,
                });
                Err(DriverError::PlaybackReportTimeout(
                    self.segment_key.expect("captured audio has a playback key"),
                ))
            }
        }
    }

    async fn push<S: Segment>(
        &mut self,
        segment: &mut S,
        frames: Vec<AudioFrame>,
    ) -> Result<(), Ended> {
        for frame in &frames {
            // Safe to race: a Rust-native future, dropping it only declines one push.
            let out = self.out;
            let pushed = self.watched(out.capture_frame(frame)).await;
            self.settle(segment);
            match pushed {
                Some(Ok(())) => {
                    if self.first_segment.is_none() {
                        self.first_segment = self.out.tracker().current_segment();
                        self.segment_key = self.first_segment;
                    }
                    self.pushed += 1;
                    self.pushed_len += frame_duration(frame);
                    self.batch += 1;
                    // The recording's agent channel, placed by the sink's own segment reports.
                    if let Some(tap) = self.shared.recorder_tap() {
                        tap.recorder.capture_frame(tap.now(), frame);
                    }
                }
                Some(Err(_)) | None => return Err(Ended::Cut),
            }
        }
        Ok(())
    }

    /// The resampler's tail, pushed: it belongs to the stretch it ends.
    async fn push_tail<S: Segment>(&mut self, segment: &mut S) -> Result<(), Ended> {
        let tail = self.chain.flush().map_err(Ended::Provider)?;
        self.push(segment, tail).await
    }

    async fn cut<S: Segment>(&mut self, segment: &mut S, text: String) -> Heard {
        // Synchronous: interruption may not wait for a lock.
        clear_sink(self.shared, self.handle.id(), self.out, self.batch > 0);
        self.drain(segment);
        let (heard, reported) = self.heard_at_cut(text).await;
        self.drain(segment);
        // Never started (a bare recorder), or the sink never said: no finish to journal.
        if let (true, Some(interrupted)) = (self.started, reported) {
            self.finished(heard.position, interrupted);
        }
        heard
    }

    /// Nothing pushed is nothing heard. Otherwise the sink's report of the cut if it lands within a
    /// few hops (never on the clock), else the words at the tracker's position. With it, whether
    /// the sink says it cut the segment (one that had played out before the clear reached it
    /// reports a whole finish); `None` when it said nothing.
    async fn heard_at_cut(&mut self, text: String) -> (Heard, Option<bool>) {
        let frames = self.pushed;
        if frames == 0 {
            // Nothing reached the sink, so no position (the tracker's would be another segment's).
            return (unheard(), Some(true));
        }
        let mut report = None;
        for _ in 0..CUT_REPORT_HOPS {
            let ev = self.shared.time.timeout(Duration::ZERO, self.out.wait_for_playout()).await;
            if let Ok(ev) = ev {
                report = Some(ev);
                break;
            }
            tokio::task::yield_now().await;
        }
        // A host output reports from its own loop, later than the hops: wait for it on the clock,
        // which runs unless the session is closing.
        if report.is_none() && !*self.shared.closing.borrow() {
            report =
                self.shared.time.timeout(CUT_REPORT_WAIT, self.out.wait_for_playout()).await.ok();
        }
        let Some(ev) = report else {
            let position = self.out.tracker().position();
            let played = self.completed + position;
            return (Heard { text, position, complete: false, frames, played }, None);
        };
        // The segments before were counted as they finished; a report that is not the cut is one
        // of them (nothing was captured since), already counted.
        self.drain_reports();
        let cut = if ev.interrupted { ev.playback_position } else { Duration::ZERO };
        let text = synchronized(ev.synchronized_transcript, text);
        let played = self.completed + cut;
        let heard = Heard { text, position: ev.playback_position, complete: false, frames, played };
        (heard, Some(ev.interrupted))
    }

    /// The sink's reports so far, folded without settling the segment (a cut has nothing owed).
    fn drain_reports(&mut self) {
        while let Ok(ev) = self.events.try_recv() {
            self.on_event(ev);
        }
    }
}
