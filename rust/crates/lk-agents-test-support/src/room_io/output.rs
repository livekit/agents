//! Simulated source execution for the room output's production buffering and playout core.

use lk_agents_abi::utils::spawner::TaskSet;
use lk_agents_abi::voice::io::{
    AudioOutput, PlaybackFinished, PlaybackSegment, PlaybackTracker, Result,
};
use lk_agents_abi::voice::room_io::AudioOutputConfig;
use lk_agents_abi::voice::rtc::{AudioOutputCapabilities, AudioSource, PlaybackInfo, RtcRoom};
use lk_agents_abi::{AudioFrame, BoxFuture, Clock, Closed, Monotone, Spawner, SpawnerExt, Ts};
use lk_agents_voice::room_io::output::{CaptureAdmission, Playback, PlaybackReport};
use lk_agents_voice::room_io::publication::Publication;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

struct State {
    playback: Playback,
    publication: Publication,
    source: Option<Arc<dyn AudioSource>>,
    cleanup_deadline: Option<Ts>,
    cleanup_outcome: Option<Closed>,
}

pub struct RoomAudioOutput {
    room: Arc<dyn RtcRoom>,
    clock: Arc<dyn Clock>,
    state: Mutex<State>,
    changed: Notify,
    tracker: Arc<PlaybackTracker>,
    config: AudioOutputConfig,
    spawner: Arc<dyn Spawner>,
    workers: Arc<TaskSet>,
    creation: Arc<TaskSet>,
    cleanup_done: Monotone<bool>,
}

impl RoomAudioOutput {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        config: AudioOutputConfig,
    ) -> Arc<Self> {
        let output = Arc::new(Self {
            room,
            clock,
            state: Mutex::new(State {
                playback: Playback::new(
                    config.publish.sample_rate,
                    config.publish.num_channels,
                    config.publish.queue_size_ms,
                ),
                publication: Publication::default(),
                source: None,
                cleanup_deadline: None,
                cleanup_outcome: None,
            }),
            changed: Notify::new(),
            tracker: Arc::new(PlaybackTracker::new()),
            config,
            spawner: spawner.clone(),
            workers: Arc::default(),
            creation: Arc::default(),
            cleanup_done: Monotone::new(false),
        });
        let publishing = output.clone();
        output.creation.spawn(spawner.as_ref(), "room audio publication", async move {
            publishing.publish().await;
        });
        let forwarding = output.clone();
        output.workers.spawn(spawner.as_ref(), "room audio forwarding", async move {
            forwarding.forward().await;
        });
        let waiting = output.clone();
        output.workers.spawn(spawner.as_ref(), "room audio playout", async move {
            waiting.finish().await;
        });
        let observing = output.clone();
        output.workers.spawn(spawner.as_ref(), "room audio progress", async move {
            observing.progress().await;
        });
        output
    }

    fn report(&self, report: Option<PlaybackReport>) {
        match report {
            Some(PlaybackReport::Started { segment, at }) => {
                self.tracker.playback_started(segment, at);
            }
            Some(PlaybackReport::Progressed { segment, progress }) => {
                self.tracker.playback_progressed(
                    segment,
                    progress.started_at,
                    progress.offset,
                    progress.duration,
                );
            }
            Some(PlaybackReport::Finished { segment, event }) => {
                if self.tracker.current_segment() == Some(segment) {
                    self.tracker.note_flush();
                }
                self.tracker.playback_finished(segment, event);
            }
            None => {}
        }
    }

    fn failed(&self, error: lk_agents_abi::voice::rtc::RtcError) {
        let reports = {
            let mut state = self.state.lock().unwrap();
            state.publication.fail(error.clone().into());
            state.playback.interrupted(None, self.clock.now())
        };
        for report in reports {
            self.report(Some(report));
        }
        self.changed.notify_waiters();
    }

    async fn publish(self: Arc<Self>) {
        self.until(|state| state.playback.closed() || state.publication.creating()).await;
        if !self.state.lock().unwrap().publication.creating() {
            return;
        }
        match self.room.local().publish_audio_track(self.config.publish.clone()).await {
            Ok((source, publication)) => {
                let adopted = {
                    let mut state = self.state.lock().unwrap();
                    let adopted = state.publication.adopt(publication.sid().clone());
                    if adopted {
                        state.source = Some(source.clone());
                    }
                    adopted
                };
                if adopted {
                    let subscribing = self.clone();
                    self.workers.try_spawn(
                        self.spawner.as_ref(),
                        "room audio subscription",
                        async move {
                            match publication.wait_for_subscription().await {
                                Ok(()) => subscribing.state.lock().unwrap().publication.ready(),
                                Err(error) => subscribing.failed(error),
                            }
                            subscribing.changed.notify_waiters();
                        },
                    );
                } else {
                    let outcome = self.retire(Some(source), Some(publication.sid().clone())).await;
                    let mut state = self.state.lock().unwrap();
                    state.cleanup_outcome =
                        Some(state.cleanup_outcome.unwrap_or(Closed::Clean).and(outcome));
                    state.publication.creation_retired();
                }
            }
            Err(error) => self.failed(error),
        }
        self.changed.notify_waiters();
    }

    pub async fn start(&self) -> Result<()> {
        self.state.lock().unwrap().publication.begin();
        self.changed.notify_waiters();
        self.until(|state| state.publication.outcome().is_some()).await;
        self.state.lock().unwrap().publication.outcome().unwrap()
    }

    async fn until(&self, ready: impl Fn(&State) -> bool) {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if ready(&self.state.lock().unwrap()) {
                return;
            }
            changed.await;
        }
    }

    fn clear_source(&self, source: &dyn AudioSource, park: bool) -> Option<Duration> {
        let result = (|| {
            let before = source.playback_info()?;
            source.clear_queue()?;
            let after = source.playback_info()?;
            Ok::<_, lk_agents_abi::voice::rtc::RtcError>((before, after))
        })();
        let (before, after) = match result {
            Ok(observations) => observations,
            Err(error) => {
                self.failed(error);
                return None;
            }
        };
        let (report, discarded) =
            self.state.lock().unwrap().playback.cleared(before, after, self.clock.now(), park);
        self.report(report);
        Some(discarded)
    }

    async fn forward(&self) {
        loop {
            self.until(|state| {
                state.playback.closed() || (!state.playback.paused() && !state.playback.drained())
            })
            .await;
            let (frame, source, report) = {
                let mut state = self.state.lock().unwrap();
                if state.playback.closed() {
                    return;
                }
                let Some(frame) = state.playback.next_frame() else {
                    continue;
                };
                let report = state.playback.submitted(self.clock.now());
                (frame, state.source.clone(), report)
            };
            self.report(report);
            let Some(source) = source else {
                self.state.lock().unwrap().playback.capture_settled();
                continue;
            };
            let dry = self.state.lock().unwrap().playback.dry_at();
            if let Some(dry) = dry.filter(|dry| self.clock.now() >= *dry) {
                match source.playback_info() {
                    Ok(info) => {
                        let report =
                            self.state.lock().unwrap().playback.progressed(info, dry, false);
                        self.report(report);
                    }
                    Err(error) => self.failed(error),
                }
            }
            match source.capture(&frame).await {
                Ok(()) => match source.queued_duration() {
                    Ok(queued) => {
                        let (paused, cutting) = {
                            let mut state = self.state.lock().unwrap();
                            state.playback.captured(frame.clone(), queued);
                            (state.playback.paused(), state.playback.cutting())
                        };
                        if paused {
                            if let Some(discarded) = self.clear_source(source.as_ref(), true) {
                                self.state.lock().unwrap().playback.park_captured(frame, discarded);
                            }
                        } else if cutting {
                            self.clear_source(source.as_ref(), false);
                            if let Ok(info) = source.playback_info() {
                                self.state.lock().unwrap().playback.cut_captured(info);
                            }
                        }
                        match source.queued_duration() {
                            Ok(queued) => self
                                .state
                                .lock()
                                .unwrap()
                                .playback
                                .source_queued(self.clock.now(), queued),
                            Err(error) => self.failed(error),
                        }
                    }
                    Err(error) => self.failed(error),
                },
                Err(error) => self.failed(error),
            }
            self.state.lock().unwrap().playback.capture_settled();
            self.changed.notify_waiters();
        }
    }

    async fn finish(&self) {
        loop {
            self.until(|state| state.playback.closed() || state.playback.ready_to_finish()).await;
            let segment = self.state.lock().unwrap().playback.segment();
            let mut info = PlaybackInfo::default();
            loop {
                self.until(|state| state.playback.closed() || state.playback.drained()).await;
                let source = {
                    let state = self.state.lock().unwrap();
                    if state.playback.closed() {
                        return;
                    }
                    state.source.clone()
                };
                let more = match &source {
                    Some(source) => match source.wait_for_playout().await {
                        Ok(info) => info,
                        Err(error) => {
                            self.failed(error);
                            break;
                        }
                    },
                    None => PlaybackInfo::default(),
                };
                info.played += more.played;
                info.discarded += more.discarded;
                if !self.state.lock().unwrap().playback.settled(segment, more) {
                    break;
                }
                self.until(|state| state.playback.closed() || !state.playback.finish_held()).await;
                {
                    let state = self.state.lock().unwrap();
                    if state.playback.closed() {
                        return;
                    }
                    if !state.playback.drained() {
                        continue;
                    }
                }
                if let Some(source) = source {
                    match source.queued_duration() {
                        Ok(queued) if !queued.is_zero() => continue,
                        Ok(_) => {}
                        Err(error) => {
                            self.failed(error);
                            break;
                        }
                    }
                }
                let reports =
                    self.state.lock().unwrap().playback.finish(segment, info, self.clock.now());
                for report in reports {
                    self.report(Some(report));
                }
                self.changed.notify_waiters();
                break;
            }
        }
    }

    async fn progress(&self) {
        loop {
            self.until(|state| state.playback.closed() || state.playback.playing()).await;
            if self.state.lock().unwrap().playback.closed() {
                return;
            }
            self.clock.wait_until(self.clock.now().saturating_add(Duration::from_millis(20))).await;
            let source = {
                let state = self.state.lock().unwrap();
                if state.playback.closed() {
                    return;
                }
                if !state.playback.playing() {
                    continue;
                }
                state.source.clone()
            };
            if let Some(source) = source {
                match source.playback_info() {
                    Ok(info) => {
                        let report = self.state.lock().unwrap().playback.progressed(
                            info,
                            self.clock.now(),
                            true,
                        );
                        self.report(report);
                    }
                    Err(error) => self.failed(error),
                }
            }
        }
    }

    async fn retire(
        &self,
        source: Option<Arc<dyn AudioSource>>,
        sid: Option<lk_agents_abi::voice::rtc::TrackSid>,
    ) -> Closed {
        let local = self.room.local();
        let closing = async {
            match source {
                Some(source) => source.close().await.is_err(),
                None => false,
            }
        };
        let unpublishing = async {
            match sid {
                Some(sid) => local.unpublish_track(&sid).await.is_err(),
                None => false,
            }
        };
        let (failed_close, failed_unpublish) = tokio::join!(closing, unpublishing);
        if failed_close || failed_unpublish { Closed::Failed } else { Closed::Clean }
    }

    pub async fn aclose(self: &Arc<Self>) -> Closed {
        let requested_deadline =
            self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let (first, captured, deadline) = {
            let mut state = self.state.lock().unwrap();
            let first = !state.playback.closed();
            let captured = state.playback.close();
            state.publication.close();
            let deadline = *state.cleanup_deadline.get_or_insert(requested_deadline);
            (first, captured, deadline)
        };
        self.changed.notify_waiters();
        if first {
            self.workers.close();
            let owner = self.clone();
            self.spawner.spawn_unboxed(async move {
                owner.workers.wait_empty(owner.clock.as_ref(), None).await;
                owner.creation.wait_empty(owner.clock.as_ref(), None).await;
                let (source, sid) = {
                    let mut state = owner.state.lock().unwrap();
                    (state.source.take(), state.publication.retire())
                };
                if captured && let Some(source) = source.as_ref() {
                    owner.clear_source(source.as_ref(), false);
                    let report = owner.state.lock().unwrap().playback.close_finished();
                    owner.report(report);
                }
                let outcome = owner.retire(source, sid).await;
                {
                    let mut state = owner.state.lock().unwrap();
                    state.cleanup_outcome =
                        Some(state.cleanup_outcome.unwrap_or(Closed::Clean).and(outcome));
                }
                owner.cleanup_done.set(true);
            });
        }
        let settled = lk_agents_abi::before(
            self.clock.as_ref(),
            Some(deadline),
            self.cleanup_done.wait_for(|done| *done),
        )
        .await
        .is_ok();
        let mut state = self.state.lock().unwrap();
        if !settled {
            state.cleanup_outcome = Some(Closed::TimedOut);
        }
        state.cleanup_outcome.unwrap_or(Closed::TimedOut)
    }
}

impl AudioOutput for RoomAudioOutput {
    fn label(&self) -> &str {
        "RoomIO"
    }
    fn tracker(&self) -> &Arc<PlaybackTracker> {
        &self.tracker
    }
    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: true }
    }
    fn sample_rate(&self) -> Option<u32> {
        Some(self.config.publish.sample_rate)
    }
    fn begin_segment(&self, segment: PlaybackSegment) -> Result<()> {
        if !self.tracker.begin_segment(segment) {
            return Err(lk_agents_abi::voice::io::IoError::Rtc(
                "playback segment is already active".into(),
            ));
        }
        self.state.lock().unwrap().playback.begin_segment(segment);
        Ok(())
    }

    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.state.lock().unwrap().playback.validate(frame)?;
            let segment = self.tracker.current_segment();
            if let Err(error) = self.start().await {
                if let Some(segment) = self.tracker.current_segment() {
                    self.tracker.note_flush();
                    self.tracker.playback_finished(
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
            let sample_bytes = usize::from(frame.num_channels) * 2;
            let chunk_bytes = (frame.sample_rate as usize / 20).max(1) * sample_bytes;
            for data in frame.data.chunks(chunk_bytes) {
                let chunk = AudioFrame::new(
                    data.to_vec(),
                    frame.sample_rate,
                    frame.num_channels,
                    (data.len() / sample_bytes) as u32,
                );
                loop {
                    let changed = self.changed.notified();
                    tokio::pin!(changed);
                    changed.as_mut().enable();
                    let admitted = {
                        let mut state = self.state.lock().unwrap();
                        state.publication.outcome().unwrap()?;
                        matches!(
                            state.playback.capture(segment, &chunk),
                            CaptureAdmission::Accepted { .. }
                        )
                    };
                    if admitted {
                        self.changed.notify_waiters();
                        break;
                    }
                    changed.await;
                }
            }
            Ok(())
        })
    }
    fn flush_segment(&self) {
        if self.state.lock().unwrap().playback.flush() {
            self.changed.notify_waiters();
        }
    }
    fn clear_buffer(&self) {
        let source = {
            let mut state = self.state.lock().unwrap();
            if !state.playback.clear() {
                return;
            }
            state.source.clone()
        };
        if let Some(source) = source {
            self.clear_source(source.as_ref(), false);
        }
        self.changed.notify_waiters();
    }
    fn pause(&self) {
        let source = {
            let mut state = self.state.lock().unwrap();
            if !state.playback.pause() {
                return;
            }
            state.source.clone()
        };
        if let Some(source) = source
            && let Some(discarded) = self.clear_source(source.as_ref(), true)
        {
            self.state.lock().unwrap().playback.paused_clear(discarded);
        }
    }
    fn resume(&self) {
        if self.state.lock().unwrap().playback.resume() {
            self.changed.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtc::FakeRtc;
    use lk_agents_abi::TokioSpawner;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::Poll;

    async fn pending<T>(mut future: Pin<&mut impl Future<Output = T>>) {
        std::future::poll_fn(|context| {
            assert!(future.as_mut().poll(context).is_pending());
            Poll::Ready(())
        })
        .await;
    }

    #[tokio::test]
    async fn closing_during_publication_joins_one_late_retirement() {
        let (rtc, clock) = FakeRtc::with_virtual_clock();
        let room = rtc.room("late-room-output");
        let local = room.local_participant();
        local.flow_publications(false);
        let output = RoomAudioOutput::new(
            room,
            clock,
            Arc::new(TokioSpawner::current()),
            AudioOutputConfig::default(),
        );
        tokio::task::yield_now().await;
        assert!(local.publications().is_empty(), "construction does not publish before start");
        let mut starting = Box::pin(output.start());
        pending(starting.as_mut()).await;
        tokio::task::yield_now().await;
        let mut first = Box::pin(output.aclose());
        let mut second = Box::pin(output.aclose());
        pending(first.as_mut()).await;
        pending(second.as_mut()).await;
        local.flow_publications(true);
        let (first, second) =
            tokio::time::timeout(Duration::from_secs(1), async { tokio::join!(first, second) })
                .await
                .expect("late publication must settle cleanup");
        assert_eq!((first, second), (Closed::Clean, Closed::Clean));
        assert!(starting.await.is_err());
        assert!(local.publications().is_empty(), "the late SID was unpublished");
        assert_eq!(output.aclose().await, Closed::Clean);
    }
}
