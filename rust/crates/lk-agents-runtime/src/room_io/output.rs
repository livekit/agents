//! `voice/room_io/_output.py`: `_ParticipantAudioOutput`, the agent's audio on its published track,
//! and its host object; the transcription output is [`transcription`].

use lk_agents_abi::voice::io::{AudioOutput, PlaybackFinished, PlaybackSegment, PlaybackTracker};
use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::room_io::AudioOutputConfig;
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::voice::rtc::{AudioSource, PlaybackInfo, RtcError, RtcRoom, TrackSid};
use lk_agents_abi::{AbortHandle, Closed, HOST_TEARDOWN_TIMEOUT, timeout};
use lk_agents_abi::{AudioFrame, Clock, Spawner, SpawnerExt, Ts};
use lk_agents_voice::room_io::output::{
    CaptureAdmission, Playback, PlaybackReport, capture_chunks,
};
use lk_agents_voice::room_io::publication::Publication;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

pub mod transcription;

const PROGRESS_TICK: Duration = Duration::from_millis(20);

struct State {
    playback: Playback,
    source: Option<Arc<dyn AudioSource>>,
    publication: Publication,
    cleanup_finished: bool,
    cleanup_outcome: Option<Closed>,
    cleanup_deadline: Option<Ts>,
}

struct Shared {
    clock: Arc<dyn Clock>,
    tracker: Arc<PlaybackTracker>,
    state: Mutex<State>,
    /// One wake for every waiter; each re-checks its own condition.
    wake: Notify,
    sample_rate: u32,
}

impl Shared {
    /// Never held across an await, nor across a call into the source: a host source takes the GIL,
    /// which a host thread calling into this leg already holds.
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap()
    }

    /// `clear_queue` on the source, closing the open run at what passed the speaker. A pause
    /// (`park`) resumes the timeline past the re-fed discard, which is where this run ended; a cut
    /// resumes where the source does. Returns what the cut discarded.
    fn clear_source(
        &self,
        src: &dyn AudioSource,
        park: bool,
    ) -> std::result::Result<(Option<PlaybackReport>, Duration), RtcError> {
        let before = src.playback_info()?;
        src.clear_queue()?;
        let after = src.playback_info()?;
        let now = self.clock.now();
        Ok(self.lock().playback.cleared(before, after, now, park))
    }

    fn source_result<T>(
        &self,
        source: &dyn AudioSource,
        result: std::result::Result<T, RtcError>,
    ) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                self.fail_source(source, error.into(), None);
                None
            }
        }
    }

    fn report(&self, report: Option<PlaybackReport>) {
        let Some(report) = report else { return };
        match report {
            PlaybackReport::Started { segment, at } => {
                self.tracker.playback_started(segment, at);
            }
            PlaybackReport::Progressed { segment, progress } => {
                self.tracker.playback_progressed(
                    segment,
                    progress.started_at,
                    progress.offset,
                    progress.duration,
                );
            }
            PlaybackReport::Finished { segment, event } => {
                if self.tracker.current_segment() == Some(segment) {
                    self.tracker.note_flush();
                }
                self.tracker.playback_finished(segment, event);
            }
        }
    }

    fn fail_source(
        &self,
        source: &dyn AudioSource,
        error: IoError,
        source_played: Option<Duration>,
    ) {
        if !self.fail(error, source_played) {
            return;
        }
        if let Err(error) = source.clear_queue() {
            self.lock().cleanup_outcome = Some(Closed::Failed);
            log::error!(target: "livekit.agents", "room audio queue cleanup failed: {error}");
        }
    }

    fn fail(&self, error: IoError, source_played: Option<Duration>) -> bool {
        let now = self.clock.now();
        let reports = {
            let mut state = self.lock();
            if state.playback.closed()
                || state.publication.outcome().is_some_and(|outcome| outcome.is_err())
            {
                return false;
            }
            state.publication.fail(error.clone());
            state.playback.interrupted(source_played, now)
        };
        log::error!(target: "livekit.agents", "room audio source failed: {error}");
        for report in reports {
            self.report(Some(report));
        }
        self.wake.notify_waiters();
        true
    }

    /// Park until `ready` holds. Armed before the check so a wake between isn't lost.
    async fn wait_until(&self, ready: impl Fn(&State) -> bool) {
        loop {
            let parked = self.wake.notified();
            tokio::pin!(parked);
            parked.as_mut().enable();
            if ready(&self.lock()) {
                return;
            }
            parked.await;
        }
    }

    /// `_forward_audio`: one frame at a time from the backlog into the source, paced by the source.
    async fn forward_audio(self: Arc<Self>) {
        loop {
            let frame = loop {
                let parked = self.wake.notified();
                tokio::pin!(parked);
                parked.as_mut().enable();
                {
                    let mut st = self.lock();
                    if st.playback.closed() {
                        return;
                    }
                    if let Some(frame) = st.playback.next_frame() {
                        break frame;
                    }
                }
                parked.await;
            };

            let src = self.lock().source.clone();
            if let Some(src) = src {
                self.submit(&src, frame).await;
            }
            self.lock().playback.capture_settled();
            self.wake.notify_waiters();
        }
    }

    async fn submit(&self, src: &Arc<dyn AudioSource>, frame: AudioFrame) {
        let now = self.clock.now();
        let (started, dry) = {
            let mut st = self.lock();
            (st.playback.submitted(now), st.playback.dry_at())
        };
        self.report(started);
        if let Some(dry) = dry.filter(|dry| now >= *dry) {
            let Some(info) = self.source_result(src.as_ref(), src.playback_info()) else { return };
            let progress = self.lock().playback.progressed(info, dry, false);
            self.report(progress);
        }

        // The await that paces everything.
        if let Err(error) = src.capture(&frame).await {
            if !self.lock().playback.closed() {
                let (failure, played) = match src.playback_info() {
                    Ok(info) => (error.into(), Some(info.played)),
                    Err(position_error) => (
                        IoError::Rtc(format!(
                            "{error}; playback position failed: {position_error}"
                        )),
                        None,
                    ),
                };
                self.fail_source(src.as_ref(), failure, played);
            }
            return;
        }

        let Some(queued) = self.source_result(src.as_ref(), src.queued_duration()) else { return };
        let (paused, cutting) = {
            let mut st = self.lock();
            st.playback.captured(frame.clone(), queued);
            (st.playback.paused(), st.playback.cutting())
        };
        if paused {
            let Some((report, discarded)) =
                self.source_result(src.as_ref(), self.clear_source(src.as_ref(), true))
            else {
                return;
            };
            self.report(report);
            self.lock().playback.park_captured(frame, discarded);
        } else if cutting {
            if self.source_result(src.as_ref(), src.clear_queue()).is_none() {
                return;
            }
            let Some(info) = self.source_result(src.as_ref(), src.playback_info()) else { return };
            self.lock().playback.cut_captured(info);
        }
        let Some(queued) = self.source_result(src.as_ref(), src.queued_duration()) else { return };
        self.lock().playback.source_queued(self.clock.now(), queued);
    }

    /// While a segment plays, its progress every [`PROGRESS_TICK`], as the run so far. The source
    /// is the playhead's one origin; a pause, a cut and the finish report their own runs.
    async fn report_progress(self: Arc<Self>) {
        let playing = |s: &State| s.playback.playing();
        loop {
            self.wait_until(|s| s.playback.closed() || playing(s)).await;
            let at = self.clock.now().saturating_add(PROGRESS_TICK);
            self.clock.wait_until(at).await;
            let src = {
                let st = self.lock();
                if st.playback.closed() {
                    return;
                }
                if !playing(&st) {
                    continue;
                }
                st.source.clone()
            };
            let Some(src) = src else { continue };
            let Some(info) = self.source_result(src.as_ref(), src.playback_info()) else {
                continue;
            };
            let now = self.clock.now();
            let run = {
                let mut st = self.lock();
                if !playing(&st) {
                    continue;
                }
                st.playback.progressed(info, now, true)
            };
            self.report(run);
        }
    }

    /// `_wait_for_playout`, per closed segment: wait for the drain, then finish the segment.
    async fn wait_for_playout(self: Arc<Self>) {
        loop {
            self.wait_until(|s| s.playback.closed() || s.playback.ready_to_finish()).await;
            if self.lock().playback.closed() {
                return;
            }
            // Summed over waits: a pause can make the source run dry more than once.
            let mut info = PlaybackInfo::default();
            let segment = self.lock().playback.segment();
            loop {
                self.wait_until(|s| s.playback.closed() || s.playback.drained()).await;
                let src = {
                    let st = self.lock();
                    if st.playback.closed() {
                        return;
                    }
                    st.source.clone()
                };
                let more = match src {
                    Some(src) => {
                        let Some(info) =
                            self.source_result(src.as_ref(), src.wait_for_playout().await)
                        else {
                            break;
                        };
                        info
                    }
                    None => PlaybackInfo::default(),
                };
                info.played += more.played;
                info.discarded += more.discarded;
                if !self.lock().playback.settled(segment, more) {
                    break;
                }
                // A pause holds the finish until the sink plays again or the pause becomes a cut,
                // unless the pause put audio back; then measure again until nothing is queued.
                self.wait_until(|s| s.playback.closed() || !s.playback.finish_held()).await;
                let (drained, src) = {
                    let st = self.lock();
                    if st.playback.closed() {
                        return;
                    }
                    (st.playback.drained(), st.source.clone())
                };
                if !drained {
                    continue;
                }
                if let Some(src) = src {
                    let Some(queued) = self.source_result(src.as_ref(), src.queued_duration())
                    else {
                        break;
                    };
                    if !queued.is_zero() {
                        continue;
                    }
                }
                let now = self.clock.now();
                let reports = self.lock().playback.finish(segment, info, now);
                for report in reports {
                    self.report(Some(report));
                }
                // Idle again: held captures may retry.
                self.wake.notify_waiters();
                break;
            }
        }
    }
}

/// `_ParticipantAudioOutput`: frames rebuffered into a backlog, forwarded into the track's source by
/// one pump, each flushed segment finished by one watcher.
pub struct ParticipantAudioOutput {
    room: Arc<dyn RtcRoom>,
    spawner: Arc<dyn Spawner>,
    config: AudioOutputConfig,
    shared: Arc<Shared>,
    /// The pump, the watcher and the progress reporter: for the life of the output, not one per
    /// segment.
    tasks: Mutex<Vec<AbortHandle>>,
}

impl ParticipantAudioOutput {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        config: AudioOutputConfig,
    ) -> Arc<Self> {
        let sample_rate = config.publish.sample_rate;
        let num_channels = config.publish.num_channels;
        let state = State {
            playback: Playback::new(sample_rate, num_channels, config.publish.queue_size_ms),
            source: None,
            publication: Publication::default(),
            cleanup_finished: false,
            cleanup_outcome: None,
            cleanup_deadline: None,
        };
        Arc::new(ParticipantAudioOutput {
            room,
            spawner,
            shared: Arc::new(Shared {
                clock,
                tracker: Arc::new(PlaybackTracker::new()),
                state: Mutex::new(state),
                wake: Notify::new(),
                sample_rate,
            }),
            tasks: Mutex::new(Vec::new()),
            config,
        })
    }

    fn begin_start(&self) {
        let mut tasks = self.tasks.lock().unwrap();
        if self.shared.lock().publication.begin() {
            tasks.push(self.spawner.spawn_unboxed(self.shared.clone().forward_audio()));
            tasks.push(self.spawner.spawn_unboxed(self.shared.clone().wait_for_playout()));
            tasks.push(self.spawner.spawn_unboxed(self.shared.clone().report_progress()));
            let room = self.room.clone();
            let shared = self.shared.clone();
            let options = self.config.publish.clone();
            let self_spawner = self.spawner.clone();
            // Creation belongs to the output, even when the caller abandons start(). Close drains
            // this task before releasing its handles, so a late SID can always be unpublished.
            tasks.push(self.spawner.spawn_unboxed(async move {
                let (source, publication) = match room.local().publish_audio_track(options).await {
                    Ok(published) => published,
                    Err(error) => {
                        shared.lock().publication.fail(error.into());
                        shared.wake.notify_waiters();
                        return;
                    }
                };
                let adopted = {
                    let mut state = shared.lock();
                    if state.publication.adopt(publication.sid().clone()) {
                        state.source = Some(source.clone());
                        true
                    } else {
                        false
                    }
                };
                if !adopted {
                    let outcome = retire_publication(
                        room.clone(),
                        source,
                        publication.sid().clone(),
                        self_spawner,
                    )
                    .await;
                    let mut state = shared.lock();
                    state.cleanup_outcome =
                        Some(state.cleanup_outcome.unwrap_or(Closed::Clean).and(outcome));
                    drop(state);
                    shared.lock().publication.creation_retired();
                    shared.wake.notify_waiters();
                    return;
                }
                shared.wake.notify_waiters();
                match publication.wait_for_subscription().await {
                    Ok(()) => shared.lock().publication.ready(),
                    Err(error) => shared.lock().publication.fail(error.into()),
                }
                shared.wake.notify_waiters();
            }));
        }
    }

    pub async fn start(&self) -> Result<()> {
        self.begin_start();
        self.shared.wait_until(|s| s.publication.outcome().is_some()).await;
        self.shared.lock().publication.outcome().expect("readiness settled")
    }

    async fn capture_chunk(
        &self,
        segment: Option<PlaybackSegment>,
        frame: &AudioFrame,
    ) -> Result<()> {
        self.shared.wait_until(|s| s.publication.outcome().is_some()).await;
        loop {
            let parked = self.shared.wake.notified();
            tokio::pin!(parked);
            parked.as_mut().enable();
            let queued = {
                let mut st = self.shared.lock();
                if let Some(Err(error)) = st.publication.outcome() {
                    drop(st);
                    // capture_frame may already have opened a segment for this refused frame.
                    if let Some(segment) = self.shared.tracker.current_segment() {
                        self.shared.tracker.note_flush();
                        self.shared.tracker.playback_finished(
                            segment,
                            PlaybackFinished {
                                playback_position: Duration::ZERO,
                                interrupted: true,
                                synchronized_transcript: None,
                            },
                        );
                    }
                    return Err(error);
                }
                match st.playback.capture(segment, frame) {
                    CaptureAdmission::Accepted { wake } => Some(wake),
                    CaptureAdmission::Wait => None,
                }
            };
            match queued {
                Some(woke) => {
                    if woke {
                        self.shared.wake.notify_waiters();
                    }
                    return Ok(());
                }
                None => parked.await,
            }
        }
    }

    pub async fn aclose(&self) -> Closed {
        let requested_deadline = self.shared.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        let (first, captured, deadline) = {
            let mut state = self.shared.lock();
            let first = !state.playback.closed();
            let captured = state.playback.close();
            state.publication.close();
            let deadline = *state.cleanup_deadline.get_or_insert(requested_deadline);
            (first, captured, deadline)
        };
        self.shared.wake.notify_waiters();
        if first {
            // Stop frame/progress producers before the terminal playback event. Creation keeps its
            // separate obligation to deliver or retire the track, even if start() was cancelled.
            {
                let mut tasks = self.tasks.lock().unwrap();
                let workers = tasks.len().min(3);
                for handle in tasks.drain(..workers) {
                    handle.abort();
                }
            }
            let shared = self.shared.clone();
            let room = self.room.clone();
            let spawner = self.spawner.clone();
            self.spawner.spawn_unboxed(async move {
                shared.wait_until(|s| !s.publication.creating()).await;
                let (source, sid) = {
                    let mut state = shared.lock();
                    (state.source.take(), state.publication.retire())
                };
                let mut outcome = Closed::Clean;
                if let Some(source) = source {
                    if captured {
                        match shared.clear_source(source.as_ref(), false) {
                            Ok((run, _)) => shared.report(run),
                            Err(error) => {
                                outcome = outcome.and(cleanup_outcome(Err(error), "audio queue"))
                            }
                        }
                    }
                    if captured {
                        let report = shared.lock().playback.close_finished();
                        shared.report(report);
                    }
                    if let Some(sid) = sid {
                        outcome = outcome.and(retire_publication(room, source, sid, spawner).await);
                    } else {
                        outcome =
                            outcome.and(cleanup_outcome(source.close().await, "audio source"));
                    }
                }
                {
                    let mut state = shared.lock();
                    state.cleanup_finished = true;
                    state.cleanup_outcome =
                        Some(state.cleanup_outcome.unwrap_or(Closed::Clean).and(outcome));
                }
                shared.wake.notify_waiters();
            });
        }
        let remaining = deadline.saturating_duration_since(self.shared.clock.now());
        let settled = timeout(
            self.shared.clock.as_ref(),
            remaining,
            self.shared.wait_until(|s| s.cleanup_finished),
        )
        .await
        .is_ok();
        let outcome = {
            let mut state = self.shared.lock();
            if !settled {
                state.cleanup_outcome = Some(Closed::TimedOut);
            }
            state.cleanup_outcome.unwrap_or(Closed::TimedOut)
        };
        if settled {
            for handle in self.tasks.lock().unwrap().drain(..) {
                handle.abort();
            }
        }
        outcome
    }
}

async fn retire_publication(
    room: Arc<dyn RtcRoom>,
    source: Arc<dyn AudioSource>,
    sid: TrackSid,
    spawner: Arc<dyn Spawner>,
) -> Closed {
    // Source closure and SID retirement are separate obligations: either SDK call may hang.
    // This task retains both receipts after a close waiter reports its deadline outcome.
    let (closed, closing) = tokio::sync::oneshot::channel();
    spawner.spawn_unboxed(async move {
        let outcome = cleanup_outcome(source.close().await, "audio source");
        let _ = closed.send(outcome);
    });
    let unpublished =
        cleanup_outcome(room.local().unpublish_track(&sid).await, "audio publication");
    unpublished.and(closing.await.unwrap_or(Closed::Failed))
}

fn cleanup_outcome(result: std::result::Result<(), RtcError>, resource: &str) -> Closed {
    match result {
        Ok(()) => Closed::Clean,
        Err(error) => {
            log::error!(target: "livekit.agents", "room {resource} cleanup failed: {error}");
            Closed::Failed
        }
    }
}

impl AudioOutput for ParticipantAudioOutput {
    fn label(&self) -> &str {
        "RoomIO"
    }

    fn tracker(&self) -> &Arc<PlaybackTracker> {
        &self.shared.tracker
    }

    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: true }
    }

    fn sample_rate(&self) -> Option<u32> {
        Some(self.shared.sample_rate)
    }

    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let segment = self.shared.tracker.current_segment();
            self.shared.lock().playback.validate(frame)?;
            for chunk in capture_chunks(frame) {
                self.capture_chunk(segment, &chunk).await?;
            }
            Ok(())
        })
    }

    fn begin_segment(&self, segment: PlaybackSegment) -> Result<()> {
        if !self.shared.tracker.begin_segment(segment) {
            return Err(lk_agents_abi::voice::io::IoError::Rtc(
                "playback segment is already active".into(),
            ));
        }
        self.shared.lock().playback.begin_segment(segment);
        Ok(())
    }

    fn flush_segment(&self) {
        if self.shared.lock().playback.flush() {
            self.shared.wake.notify_waiters();
        }
    }

    fn clear_buffer(&self) {
        let src = {
            let mut st = self.shared.lock();
            if !st.playback.clear() {
                return;
            }
            st.source.clone()
        };
        self.shared.wake.notify_waiters();
        if let Some(src) = src {
            if let Some((report, _)) = self
                .shared
                .source_result(src.as_ref(), self.shared.clear_source(src.as_ref(), false))
            {
                self.shared.report(report);
            }
        }
    }

    fn pause(&self) {
        let src = {
            let mut st = self.shared.lock();
            if !st.playback.pause() {
                return;
            }
            st.source.clone()
        };
        let Some(src) = src else { return };
        let Some((report, discarded)) =
            self.shared.source_result(src.as_ref(), self.shared.clear_source(src.as_ref(), true))
        else {
            return;
        };
        self.shared.report(report);
        self.shared.lock().playback.paused_clear(discarded);
    }

    fn resume(&self) {
        if self.shared.lock().playback.resume() {
            self.shared.wake.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests;

mod segment_ids;
