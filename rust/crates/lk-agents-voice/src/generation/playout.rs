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
use lk_agents_media::{FrameResampler, OnChange, Quality};
use lk_agents_rtc::frame_duration;
use lk_agents_voice::io::{AudioOutput, AudioOutputExt, PlaybackEvent, PlaybackObserver};
use lk_agents_voice::transcription::Visible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// A pushed frame's length when none has been pushed yet: what every TTS in the tree emits.
const DEFAULT_FRAME_LEN: Duration = Duration::from_millis(20);

/// Scheduler hops a cut segment gives the sink to report the cut. Hops, not time: a speech cut at
/// `Closing` has no clock advancing it.
const CUT_REPORT_HOPS: usize = 16;

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
    fn next(&mut self) -> impl std::future::Future<Output = Result<Next, DriverError>> + Send;

    fn text(&self) -> String;

    /// The sink reported this segment's first frame leaving for the speaker (not the first push:
    /// a paused sink plays nothing). `at` is the sink's own instant, in wall seconds. Bookkeeping
    /// only; the `speaking` edge is `GenEvent::PlaybackStarted`'s.
    fn started_speaking(&mut self, at: f64);
}

/// Once per turn; not gated on interruption, since nothing is pushed for a speech cut before its
/// first frame. Bookkeeping only: the `speaking` edge is [`GenEvent::PlaybackStarted`]'s.
pub(crate) fn first_frame(shared: &Arc<SessionShared>, acc: &mut TurnAccumulator, at: f64) {
    acc.on_first_frame(Some(at), shared.time.wall_secs());
}

/// `perform_audio_forwarding` for one segment. `out: None` drains and drops the frames, reporting
/// nothing. The clean path ends with `flush`, then waits for the sink to report played out.
pub(crate) async fn perform_audio_forwarding<S: Segment>(
    shared: &Arc<SessionShared>,
    handle: &SpeechHandle,
    out: Option<&dyn AudioOutput>,
    segment: &mut S,
) -> Result<Heard, DriverError> {
    let Some(out) = out else {
        let complete = loop {
            if handle.interrupted() {
                break false;
            }
            match segment.next().await? {
                Next::Frame(_) | Next::Boundary => {}
                Next::End => break true,
            }
        };
        return Ok(silent(segment.text(), complete));
    };
    // Opened before the first push, so nothing the sink says about this segment is missed.
    let (tx, events) = mpsc::unbounded_channel();
    out.tracker().observe(Arc::new(SinkReports(tx)));
    let mut playing = Playing {
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
        reported: 0,
        batch: 0,
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
            // What was synthesised plays out; the caller forwards the rest as text. Not waited for.
            // The position is this segment's (zero once it is over).
            playing.flush_batch();
            playing.drain(segment);
            if playing.started {
                playing.finished(playing.played, false);
            }
            Err(e)
        }
        // No audio: flushing would make `wait_for_playout` answer with the previous event.
        Ended::Audio if playing.pushed == 0 => Ok(silent(text, true)),
        Ended::Audio => {
            playing.flush_batch();
            let Some(ev) = playing.watched(out.wait_for_playout()).await else {
                return Ok(playing.cut(segment, text).await);
            };
            // The sink's `Finished` was emitted before the wait resolved.
            playing.drain(segment);
            playing.sink_call(SinkCall::End);
            // Cut before its first frame left: never started, so no finish.
            if playing.started || !(ev.interrupted && ev.playback_position.is_zero()) {
                playing.finished(ev.playback_position, ev.interrupted);
            }
            // A synchroniser's words up to the played position, not the words sent.
            Ok(Heard {
                text: synchronized(ev.synchronized_transcript, text),
                position: ev.playback_position,
                complete: !ev.interrupted,
                frames: playing.pushed,
            })
        }
    }
}

fn synchronized(heard: Option<String>, text: String) -> String {
    heard.map(|heard| Visible::of(&heard).text).unwrap_or(text)
}

/// `false` when interrupted while parked. Every source and every sink goes through here.
async fn wait_launch(shared: &Arc<SessionShared>, handle: &SpeechHandle) -> bool {
    let speech = handle.id();
    let mut level = shared.launch.subscribe();
    if !level.borrow_and_update().contains(&speech) {
        return true;
    }
    shared.trace(crate::trace::InterruptStep::PlayoutGate {
        speech,
        op: crate::agent_session::cycle::GateOp::Pause,
        cause: crate::agent_session::cycle::GateCause::Launch,
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

/// `flush()` then `clear_buffer()`: a cut segment reports `Clear`, never `Flush`+`End`.
pub(crate) fn clear_sink(shared: &Arc<SessionShared>, speech: SpeechId, out: &dyn AudioOutput) {
    shared.trace(PlayoutStep::SinkCalled { speech: Some(speech), call: SinkCall::Clear });
    out.cut();
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
    Heard { text, position: Duration::ZERO, complete, frames: 0 }
}

pub(super) fn unheard() -> Heard {
    silent(String::new(), false)
}

/// What the sink reported for this segment, on one channel, in order. `Shown`: a synchroniser
/// below released words as of the played position; nothing to fold, but the playout wakes at the
/// release, ahead of the words' forwarder, so a finished segment's stream closes before its words
/// reach the text output. `Reset`: the sink's playback count was reset under it.
enum SinkReport {
    Playback(PlaybackEvent),
    Shown,
    Reset,
}

struct SinkReports(mpsc::UnboundedSender<SinkReport>);

impl PlaybackObserver for SinkReports {
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        let _ = self.0.send(SinkReport::Playback(ev.clone()));
    }

    fn on_transcript_shown(&self, _text: &str) {
        let _ = self.0.send(SinkReport::Shown);
    }

    fn on_playback_reset(&self) {
        let _ = self.0.send(SinkReport::Reset);
    }

    fn is_closed(&self) -> bool {
        self.0.is_closed()
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
    shared: &'a Arc<SessionShared>,
    handle: &'a SpeechHandle,
    out: &'a dyn AudioOutput,
    events: mpsc::UnboundedReceiver<SinkReport>,
    chain: OutputChain,
    started: bool,
    /// The first-frame instant owed to the segment's `started_speaking`, until it is free.
    owed: Option<f64>,
    /// The frame length `AudioPlayed` counts in.
    pushed: u64,
    pushed_len: Duration,
    played: Duration,
    reported: u64,
    /// Pushed since the last boundary: one `SinkCall::Push` record per stretch.
    batch: u64,
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

    fn report_push(&self) {
        if self.batch > 0 {
            self.sink_call(SinkCall::Push { frames: self.batch });
        }
    }

    fn flush_batch(&mut self) {
        if std::mem::take(&mut self.batch) > 0 {
            self.sink_call(SinkCall::Flush);
            self.out.flush();
        }
    }

    fn on_event(&mut self, report: SinkReport) {
        let speech = self.handle.id();
        match report {
            SinkReport::Shown => {}
            SinkReport::Reset => self.trace(PlayoutStep::SinkReset { speech }),
            SinkReport::Playback(PlaybackEvent::Started(s)) => {
                let time = &self.shared.time;
                let at = time.wall_secs()
                    - time.now().saturating_duration_since(s.created_at).as_secs_f64();
                self.start(at);
            }
            SinkReport::Playback(PlaybackEvent::Progressed(p)) => {
                let (offset, duration) = (p.offset, p.duration);
                self.trace(PlayoutStep::PlaybackProgressed { speech, offset, duration });
                self.played += p.duration;
                self.report_played();
            }
            SinkReport::Playback(PlaybackEvent::Finished(f)) => {
                // A sink with no `Started` (a bare recorder) still played it, unless cut at zero.
                if !(f.interrupted && f.playback_position.is_zero()) {
                    self.start(self.shared.time.wall_secs());
                }
                self.played = self.played.max(f.playback_position);
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

    /// `biased`, so an event already reported is folded before the push it may explain.
    async fn watched<T>(&mut self, fut: impl std::future::Future<Output = T>) -> Option<T> {
        tokio::pin!(fut);
        let mut listening = true;
        loop {
            tokio::select! {
                biased;
                () = self.handle.wait_for_interrupted() => return None,
                ev = self.events.recv(), if listening => match ev {
                    Some(ev) => self.on_event(ev),
                    None => listening = false,
                },
                v = &mut fut => return Some(v),
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
        clear_sink(self.shared, self.handle.id(), self.out);
        self.drain(segment);
        let heard = self.heard_at_cut(text).await;
        self.drain(segment);
        // Never started (a bare recorder): no finish to report.
        if self.started {
            self.finished(heard.position, true);
        }
        heard
    }

    /// Nothing pushed is nothing heard. Otherwise the sink's report of the cut if it lands within a
    /// few hops (never on the clock), else the words at the tracker's position.
    async fn heard_at_cut(&mut self, text: String) -> Heard {
        let frames = self.pushed;
        if frames == 0 {
            // Nothing reached the sink, so no position (the tracker's would be another segment's).
            return unheard();
        }
        for _ in 0..CUT_REPORT_HOPS {
            let ev = self.shared.time.timeout(Duration::ZERO, self.out.wait_for_playout()).await;
            if let Ok(ev) = ev {
                let text = synchronized(ev.synchronized_transcript, text);
                return Heard { text, position: ev.playback_position, complete: false, frames };
            }
            tokio::task::yield_now().await;
        }
        Heard { text, position: self.out.tracker().position(), complete: false, frames }
    }
}
