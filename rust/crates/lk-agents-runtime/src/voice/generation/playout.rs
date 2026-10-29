//! `perform_audio_forwarding`: one segment's frames pushed to the sink, and what the sink reports.

use lk_agents_abi::AudioFrame;
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputExt, PlaybackEvent, PlaybackFinished, PlaybackObserver, PlaybackSegment,
};
use lk_agents_abi::voice::rtc::frame_duration;
use lk_agents_media::{FrameResampler, OnChange, Quality};
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::DriverError;
use lk_agents_voice::agent_session::SessionShared;
use lk_agents_voice::generation::GenEvent;
use lk_agents_voice::generation::TurnAccumulator;
use lk_agents_voice::generation::playout::{
    Completion, CutEffect, CutEvent, CutReceipt, Playout, PlayoutEffect, SinkReport,
};
use lk_agents_voice::speech_handle::SpeechHandle;
use lk_agents_voice::trace::{PlayoutStep, SinkCall};
use lk_agents_voice::transcription::Heard;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

pub(crate) use lk_agents_voice::generation::playout::Next;
use lk_agents_voice::generation::playout::{FlowEffect, FlowEvent, SegmentFlow, drained};

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
    acc.on_forwarded(
        lk_agents_voice::generation::forward::ForwardEvent::Audio { at },
        shared.time.wall_secs(),
    );
}

pub(crate) use lk_agents_voice::generation::playout::SourceFailed;
use lk_agents_voice::generation::playout::{
    FinishEffect, FinishEvent, SegmentEnd as Ended, SegmentFinish,
};

/// `perform_audio_forwarding` for one segment. `out: None` drains and drops the frames, reporting
/// nothing. The clean path ends with `flush`, then waits for the sink to report played out.
pub(crate) async fn perform_audio_forwarding<S: Segment>(
    shared: &Arc<SessionShared>,
    handle: &SpeechHandle,
    out: Option<&dyn AudioOutput>,
    segment: &mut S,
) -> Result<Heard, SourceFailed> {
    let Some(out) = out else {
        let mut flow = SegmentFlow::new(false, segment.stop_at_boundary());
        let mut effect = FlowEffect::CheckInterruption;
        loop {
            let event = match effect {
                FlowEffect::CheckInterruption => {
                    FlowEvent::Ready { interrupted: handle.interrupted() }
                }
                FlowEffect::Read => {
                    let next = tokio::select! {
                        biased;
                        () = handle.wait_for_interrupted() => None,
                        next = segment.next() => Some(next),
                    };
                    FlowEvent::Source { next, interrupted: handle.interrupted() }
                }
                FlowEffect::Finish(end) => return drained(end, segment.text()),
                _ => unreachable!("a source with no destination requests only reads"),
            };
            effect = flow.handle(event, 0);
        }
    };
    // Opened before the first push, so nothing the sink says about this segment is missed.
    let (tx, events) = mpsc::unbounded_channel();
    out.tracker().observe(Arc::new(SinkReports(tx)));
    let mut playing = Playing {
        state: Playout::new(handle.id(), out.tracker().current_segment()),
        shared,
        handle,
        out,
        events,
        chain: OutputChain::new(out.sample_rate()),
    };
    let mut flow = SegmentFlow::new(true, segment.stop_at_boundary());
    let mut effect = FlowEffect::CheckInterruption;
    let ended = loop {
        let event = match effect {
            FlowEffect::CheckInterruption => FlowEvent::Ready { interrupted: handle.interrupted() },
            FlowEffect::Read => {
                // One pending source read, with the sink's independent reports folded beside it.
                let next = playing.watched(segment.next()).await;
                playing.settle(segment);
                FlowEvent::Source { next, interrupted: handle.interrupted() }
            }
            FlowEffect::WaitLaunch => FlowEvent::Launched(wait_launch(shared, handle).await),
            FlowEffect::Frame(frame) => {
                playing.drain(segment);
                let result = match playing.chain.push(&frame) {
                    Ok(frames) => playing.push(segment, frames).await,
                    Err(error) => Err(Ended::Provider(error)),
                };
                FlowEvent::Forwarded(result)
            }
            FlowEffect::Tail => FlowEvent::TailPushed(playing.push_tail(segment).await),
            FlowEffect::FlushBoundary => {
                playing.report_push();
                playing.flush_batch();
                FlowEvent::BoundaryFlushed
            }
            FlowEffect::Finish(end) => break end,
        };
        effect = flow.handle(event, playing.state.pushed);
    };
    playing.report_push();
    let text = segment.text();
    let (mut finish, mut effect) = SegmentFinish::begin(ended, text, &playing.state);
    loop {
        let event = match effect {
            FinishEffect::Drain => {
                playing.flush_batch();
                FinishEvent::Receipt(
                    with_playout_controls(segment, playing.played_out_report()).await,
                )
            }
            FinishEffect::Accept(receipt) => {
                playing.played_out(segment, &receipt);
                FinishEvent::Accepted(receipt)
            }
            FinishEffect::Cut(text) => FinishEvent::Cut(playing.cut(segment, text).await),
            FinishEffect::Confirm => {
                playing.drain_reports();
                FinishEvent::Confirmed
            }
            FinishEffect::Complete(result) => return result,
        };
        effect = finish.handle(event, &playing.state);
    }
}

/// `false` when interrupted while parked. Every source and every sink goes through here.
async fn wait_launch(shared: &Arc<SessionShared>, handle: &SpeechHandle) -> bool {
    let speech = handle.id();
    let mut level = shared.view.subscribe();
    if !level.borrow_and_update().launch_holds.contains(&speech) {
        return true;
    }
    shared.trace(lk_agents_voice::trace::InterruptStep::PlayoutGate {
        speech,
        op: lk_agents_voice::agent_session::effects::GateOp::Pause,
        cause: lk_agents_voice::agent_session::effects::GateCause::Launch,
    });
    loop {
        tokio::select! {
            biased;
            () = handle.wait_for_interrupted() => return false,
            changed = level.changed() => {
                // A closed level (the loop is gone) holds nothing.
                if changed.is_err() || !level.borrow_and_update().launch_holds.contains(&speech) {
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

/// One segment on its way to the sink. Never off a push. The segment is not held here, so its
/// `next()` can be pending while reports are folded (`watched`) and settled once it is free
/// (`settle`).
struct Playing<'a> {
    state: Playout,
    shared: &'a Arc<SessionShared>,
    handle: &'a SpeechHandle,
    out: &'a dyn AudioOutput,
    events: mpsc::UnboundedReceiver<SinkReport>,
    chain: OutputChain,
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
        if self.state.started || !(ev.interrupted && ev.playback_position.is_zero()) {
            self.finished(ev.playback_position, ev.interrupted);
        }
    }

    fn report_push(&self) {
        if self.state.batch > 0 {
            self.sink_call(SinkCall::Push { frames: self.state.batch });
        }
    }

    fn flush_batch(&mut self) {
        if self.state.take_flush() {
            self.sink_call(SinkCall::Flush);
            self.out.flush();
            let step = self.state.flushed(self.shared.time.now());
            self.trace(step);
        }
    }

    fn on_event(&mut self, report: SinkReport) {
        let now = self.shared.time.read();
        for effect in self.state.handle(report, now) {
            match effect {
                PlayoutEffect::Trace(step) => self.trace(step),
                PlayoutEffect::Started => {
                    self.shared.report(GenEvent::PlaybackStarted { speech: self.handle.id() });
                }
            }
        }
    }

    /// Now that nothing borrows it.
    fn settle<S: Segment>(&mut self, segment: &mut S) {
        if let Some(at) = self.state.take_started() {
            segment.started_speaking(at);
        }
    }

    fn drain<S: Segment>(&mut self, segment: &mut S) {
        while let Ok(ev) = self.events.try_recv() {
            self.on_event(ev);
        }
        self.settle(segment);
    }

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
        let due =
            self.state.finish_due.expect("accepted audio was flushed before awaiting playback");
        let left = due.saturating_duration_since(time.now());
        match self.watched(time.timeout(left, out.wait_for_playout())).await {
            None => Ok(None),
            Some(Ok(event)) => Ok(Some(event)),
            Some(Err(_)) => {
                let position = self.state.played.saturating_sub(self.state.completed);
                self.trace(PlayoutStep::FinishLost {
                    speech: self.handle.id(),
                    segment: self.state.segment_key.expect("captured audio has a playback key"),
                    position,
                });
                Err(DriverError::PlaybackReportTimeout(
                    self.state.segment_key.expect("captured audio has a playback key"),
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
                    self.state
                        .captured(self.out.tracker().current_segment(), frame_duration(frame));
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
        clear_sink(self.shared, self.handle.id(), self.out, self.state.batch > 0);
        self.drain(segment);
        let (heard, reported) = self.heard_at_cut(text).await;
        self.drain(segment);
        // Never started (a bare recorder), or the sink never said: no finish to journal.
        if let (true, Some(interrupted)) = (self.state.started, reported) {
            self.finished(heard.position, interrupted);
        }
        heard
    }

    /// Nothing pushed is nothing heard. Otherwise the sink's report of the cut if it lands within a
    /// few hops (never on the clock), else the words at the tracker's position. With it, whether
    /// the sink says it cut the segment (one that had played out before the clear reached it
    /// reports a whole finish); `None` when it said nothing.
    async fn heard_at_cut(&mut self, text: String) -> (Heard, Option<bool>) {
        let (mut receipt, mut effect) = CutReceipt::begin(self.state.pushed);
        let report = loop {
            let event = match effect {
                CutEffect::Probe => {
                    let report = self
                        .shared
                        .time
                        .timeout(Duration::ZERO, self.out.wait_for_playout())
                        .await
                        .ok();
                    CutEvent::Probed(report)
                }
                CutEffect::Yield => {
                    let mut yielded = false;
                    std::future::poll_fn(|cx| {
                        if std::mem::replace(&mut yielded, true) {
                            std::task::Poll::Ready(())
                        } else {
                            cx.waker().wake_by_ref();
                            std::task::Poll::Pending
                        }
                    })
                    .await;
                    CutEvent::Yielded { closing: *self.shared.closing.borrow() }
                }
                CutEffect::WaitUntil(at) => {
                    let left = at.saturating_duration_since(self.shared.time.now());
                    let report =
                        self.shared.time.timeout(left, self.out.wait_for_playout()).await.ok();
                    CutEvent::Waited(report)
                }
                CutEffect::Complete(report) => break report,
                CutEffect::Unheard => return (unheard(), Some(true)),
            };
            effect = receipt.handle(event, self.shared.time.now());
        };
        let Some(ev) = report else {
            let position = self.out.tracker().position();
            return (self.state.heard(text, Completion::CutUnreported { position }), None);
        };
        // Prior segments are credited by their own receipts before projecting this final cut.
        self.drain_reports();
        let interrupted = ev.interrupted;
        (self.state.heard(text, Completion::Cut(ev)), Some(interrupted))
    }

    /// The sink's reports so far, folded without settling the segment (a cut has nothing owed).
    fn drain_reports(&mut self) {
        while let Ok(ev) = self.events.try_recv() {
            self.on_event(ev);
        }
    }
}
