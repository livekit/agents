//! `perform_audio_forwarding`: one segment's frames pushed to the sink, and what the sink reports.

use crate::voice::session::SessionShared;
use crate::voice::speech_handle::SpeechHandle;
use lk_agents_abi::AudioFrame;
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputExt, PlaybackEvent, PlaybackFinished, PlaybackObserver, PlaybackSegment,
};
use lk_agents_abi::voice::rtc::frame_duration;
use lk_agents_media::{AudioResampler, OnChange, Quality};
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::DriverError;
use lk_agents_voice::generation::GenEvent;
use lk_agents_voice::generation::TurnAccumulator;
use lk_agents_voice::generation::effects::AudioPlaybackEffect;
use lk_agents_voice::generation::machine::audio_playback::SinkReport;
use lk_agents_voice::generation::machine::audio_playback::{AudioPlayback, AudioPlaybackEvent};
use lk_agents_voice::trace::{PlayoutStep, SinkCall};
use lk_agents_voice::transcription::Heard;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

pub(crate) use lk_agents_voice::generation::machine::audio_playback::Next;

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
        lk_agents_voice::generation::machine::forward::TurnAccumulatorOutputEvent::Audio { at },
        shared.time.wall_secs(),
    );
}

use lk_agents_voice::generation::machine::audio_playback::SegmentEnd as Ended;
pub(crate) use lk_agents_voice::generation::machine::audio_playback::SourceFailed;

/// `perform_audio_forwarding` for one segment. `out: None` drains and drops the frames, reporting
/// nothing. The clean path ends with `flush`, then waits for the sink to report played out.
pub(crate) async fn perform_audio_forwarding<S: Segment>(
    shared: &Arc<SessionShared>,
    handle: &SpeechHandle,
    out: Option<&dyn AudioOutput>,
    segment: &mut S,
) -> Result<Heard, Box<SourceFailed>> {
    let Some(out) = out else {
        let mut playback = AudioPlayback::new(handle.id(), None, false, segment.stop_at_boundary());
        let mut effect = playback.begin();
        loop {
            let event = match effect {
                AudioPlaybackEffect::CheckInterruption => {
                    AudioPlaybackEvent::InterruptionChecked { interrupted: handle.interrupted() }
                }
                AudioPlaybackEffect::ReadSource => {
                    let next = tokio::select! {
                        biased;
                        () = handle.wait_for_interrupted() => None,
                        next = segment.next() => Some(next),
                    };
                    AudioPlaybackEvent::SourceRead { next, interrupted: handle.interrupted() }
                }
                AudioPlaybackEffect::ReadSourceTranscript => {
                    AudioPlaybackEvent::SourceTranscriptRead(segment.text())
                }
                AudioPlaybackEffect::Complete(result) => return result,
                _ => unreachable!("playback without a sink only reads its source"),
            };
            effect = playback
                .handle(event, shared.time.now())
                .expect("source playback has not completed");
        }
    };
    // Opened before the first push, so nothing the sink says about this segment is missed.
    let (tx, events) = mpsc::unbounded_channel();
    out.tracker().observe(Arc::new(SinkReports(tx)));
    let mut playing = Playing {
        state: AudioPlayback::new(
            handle.id(),
            out.tracker().current_segment(),
            true,
            segment.stop_at_boundary(),
        ),
        shared,
        handle,
        out,
        events,
        chain: OutputChain::new(out.sample_rate()),
    };
    let mut effect = playing.state.begin();
    loop {
        let event = match effect {
            AudioPlaybackEffect::CheckInterruption => {
                AudioPlaybackEvent::InterruptionChecked { interrupted: handle.interrupted() }
            }
            AudioPlaybackEffect::ReadSource => {
                // One pending source read, with sink reports folded beside it. Source callbacks
                // settle only after that borrow has ended.
                let next = playing.watched(segment.next()).await;
                playing.settle(segment);
                AudioPlaybackEvent::SourceRead { next, interrupted: handle.interrupted() }
            }
            AudioPlaybackEffect::WaitForLaunchPermission => {
                AudioPlaybackEvent::LaunchPermissionResolved {
                    allowed: wait_launch(shared, handle).await,
                }
            }
            AudioPlaybackEffect::ForwardFrame(frame) => {
                playing.drain(segment);
                let result = match playing.chain.push(&frame) {
                    Ok(frames) => playing.push(segment, frames).await,
                    Err(error) => Err(Ended::Provider(error)),
                };
                AudioPlaybackEvent::FrameForwarded(result)
            }
            AudioPlaybackEffect::ForwardResamplerTail => {
                AudioPlaybackEvent::ResamplerTailForwarded(playing.push_tail(segment).await)
            }
            AudioPlaybackEffect::FlushBoundary => {
                playing.report_push();
                playing.flush_batch();
                AudioPlaybackEvent::BoundaryFlushed
            }
            AudioPlaybackEffect::ReadSourceTranscript => {
                playing.report_push();
                AudioPlaybackEvent::SourceTranscriptRead(segment.text())
            }
            AudioPlaybackEffect::WaitForPlaybackReceipt => {
                playing.flush_batch();
                AudioPlaybackEvent::PlaybackReceiptReceived(
                    with_playout_controls(segment, playing.played_out_report()).await,
                )
            }
            AudioPlaybackEffect::AcceptPlaybackReceipt(receipt) => {
                playing.played_out(segment, &receipt);
                AudioPlaybackEvent::PlaybackReceiptAccepted(receipt)
            }
            AudioPlaybackEffect::ClearOutput => {
                // Interruption may not wait for a lock. Clear before any receipt polling.
                clear_sink(shared, handle.id(), out, playing.state.pending_frames() > 0);
                playing.drain(segment);
                AudioPlaybackEvent::OutputCleared
            }
            AudioPlaybackEffect::PollCutReceipt => {
                let report = shared.time.timeout(Duration::ZERO, out.wait_for_playout()).await.ok();
                AudioPlaybackEvent::CutReceiptPolled(report)
            }
            AudioPlaybackEffect::YieldForCutReceipt => {
                crate::process::yield_now().await;
                AudioPlaybackEvent::CutReceiptYielded { closing: *shared.closing.borrow() }
            }
            AudioPlaybackEffect::WaitForCutReceiptUntil(at) => {
                let left = at.saturating_duration_since(shared.time.now());
                let report = shared.time.timeout(left, out.wait_for_playout()).await.ok();
                AudioPlaybackEvent::CutReceiptWaited(report)
            }
            AudioPlaybackEffect::ResolveCutReceipt(report) => {
                if report.is_some() {
                    // Prior segments are credited before projecting this final cut.
                    playing.drain_reports();
                }
                AudioPlaybackEvent::CutReceiptResolved {
                    report,
                    position: out.tracker().position(),
                }
            }
            AudioPlaybackEffect::FinishCut { heard, reported } => {
                playing.drain(segment);
                // Journal the physical receipt, even if clear arrived after playback finished.
                if let (true, Some(interrupted)) = (playing.state.has_started(), reported) {
                    playing.finished(heard.position, interrupted);
                }
                AudioPlaybackEvent::CutFinished(heard)
            }
            AudioPlaybackEffect::ConfirmPlayed => {
                playing.drain_reports();
                AudioPlaybackEvent::PlayedConfirmed
            }
            AudioPlaybackEffect::Complete(result) => return result,
            AudioPlaybackEffect::Trace(_) | AudioPlaybackEffect::Started => {
                unreachable!("sink observations are performed when received")
            }
        };
        effect = playing
            .state
            .handle(event, shared.time.now())
            .expect("audio playback has not completed");
    }
}

/// `false` when interrupted while parked. Every source and every sink goes through here.
async fn wait_launch(shared: &Arc<SessionShared>, handle: &SpeechHandle) -> bool {
    let speech = handle.id();
    let mut level = shared.view.clone();
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
struct OutputChain(Option<AudioResampler>);

impl OutputChain {
    fn new(target: Option<u32>) -> Self {
        OutputChain(target.map(|rate| AudioResampler::new(rate, Quality::Medium, OnChange::Drain)))
    }

    fn push(&mut self, frame: &AudioFrame) -> Result<Vec<AudioFrame>, DriverError> {
        let Some(chain) = self.0.as_mut() else { return Ok(vec![frame.clone()]) };
        chain.push_frame(frame).map_err(|e| DriverError::Resample(e.to_string()))
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

    fn on_transcript_shown(
        &self,
        segment: Option<PlaybackSegment>,
        word: &lk_agents_abi::TranscriptChunk,
    ) {
        let _ = self.0.send(SinkReport::Shown(segment, word.clone()));
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

    fn on_transcript_shown(
        &self,
        _: Option<PlaybackSegment>,
        word: &lk_agents_abi::TranscriptChunk,
    ) {
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
    state: AudioPlayback,
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
        if self.state.has_started() || !(ev.interrupted && ev.playback_position.is_zero()) {
            self.finished(ev.playback_position, ev.interrupted);
        }
    }

    fn report_push(&self) {
        if self.state.pending_frames() > 0 {
            self.sink_call(SinkCall::Push { frames: self.state.pending_frames() });
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
        for effect in self.state.on_report(report, now) {
            match effect {
                AudioPlaybackEffect::Trace(step) => self.trace(step),
                AudioPlaybackEffect::Started => {
                    self.shared.report(GenEvent::PlaybackStarted { speech: self.handle.id() });
                }
                _ => unreachable!("a sink report only publishes accounting observations"),
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
        let due = self
            .state
            .finish_deadline()
            .expect("accepted audio was flushed before awaiting playback");
        let left = due.saturating_duration_since(time.now());
        match self.watched(time.timeout(left, out.wait_for_playout())).await {
            None => Ok(None),
            Some(Ok(event)) => Ok(Some(event)),
            Some(Err(_)) => {
                let position =
                    self.state.played_audio().saturating_sub(self.state.completed_audio());
                self.trace(PlayoutStep::FinishLost {
                    speech: self.handle.id(),
                    segment: self.state.segment_key().expect("captured audio has a playback key"),
                    position,
                });
                Err(DriverError::PlaybackReportTimeout(
                    self.state.segment_key().expect("captured audio has a playback key"),
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

    /// The sink's reports so far, folded without settling the segment (a cut has nothing owed).
    fn drain_reports(&mut self) {
        while let Ok(ev) = self.events.try_recv() {
            self.on_event(ev);
        }
    }
}
