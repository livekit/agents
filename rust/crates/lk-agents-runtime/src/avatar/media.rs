//! Avatar track execution over the shared RTC ports. The voice plan supplies publication options.

use crate::rtc::synchronizer::AVSynchronizer;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::video::MediaFrame;
use lk_agents_abi::voice::avatar::AvatarMedia;
use lk_agents_abi::voice::rtc::media::{AVSynchronizer as _, VideoTrackOptions};
use lk_agents_abi::voice::rtc::{RtcError, RtcRoom, TrackPublication, TrackPublishOptions};
use lk_agents_abi::{Clock, Spawner, SpawnerExt};
use lk_agents_net::task::TaskSet;
use lk_agents_voice::av_sync::MediaPlan;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::watch;

pub struct Media {
    room: Arc<dyn RtcRoom>,
    pub synchronizer: Arc<AVSynchronizer>,
    audio: Arc<dyn lk_agents_abi::voice::rtc::AudioSource>,
    video: Arc<dyn lk_agents_abi::voice::rtc::VideoSource>,
    audio_options: TrackPublishOptions,
    video_options: VideoTrackOptions,
    publications: Arc<Mutex<Vec<Arc<dyn TrackPublication>>>>,
    closing: AtomicBool,
    closed: watch::Sender<Option<Result<(), RtcError>>>,
    close_tasks: Arc<TaskSet>,
    spawner: Arc<dyn Spawner>,
}

impl Media {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        plan: MediaPlan,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Result<Arc<Self>, RtcError> {
        let video = room.create_video_source(plan.video.clone())?;
        let audio = match room.create_audio_source(plan.audio.clone()) {
            Ok(audio) => audio,
            Err(error) => {
                // The constructor cannot await. A cleanup operation owns the source until close
                // settles, including when the caller receives the initialization failure.
                spawner.spawn_detached(async move {
                    if let Err(error) = video.close().await {
                        log::warn!("RTC source rollback: {error}");
                    }
                });
                return Err(error);
            }
        };
        let synchronizer = AVSynchronizer::new(
            audio.clone(),
            video.clone(),
            plan.pacer,
            plan.capacity,
            clock,
            spawner.clone(),
        );
        Ok(Arc::new(Self {
            room,
            synchronizer,
            audio,
            video,
            audio_options: plan.audio,
            video_options: plan.video,
            publications: Arc::default(),
            closing: AtomicBool::new(false),
            closed: watch::channel(None).0,
            close_tasks: Arc::default(),
            spawner,
        }))
    }

    async fn publish_tracks(&self) -> Result<(), RtcError> {
        let local = self.room.local();
        let audio =
            local.publish_audio_source(self.audio.clone(), self.audio_options.clone()).await?;
        self.publications.lock().push(audio.clone());
        audio.wait_for_subscription().await?;
        let video =
            local.publish_video_source(self.video.clone(), self.video_options.clone()).await?;
        self.publications.lock().push(video);
        Ok(())
    }

    async fn close(&self) -> Result<(), RtcError> {
        let mut closed = self.closed.subscribe();
        if !self.closing.swap(true, Ordering::SeqCst) {
            let (publications, synchronizer, room, closed) = (
                self.publications.clone(),
                self.synchronizer.clone(),
                self.room.clone(),
                self.closed.clone(),
            );
            self.close_tasks.spawn(&*self.spawner, "avatar RTC tracks close", async move {
                let publications = std::mem::take(&mut *publications.lock());
                let mut result = synchronizer.aclose().await;
                for publication in publications {
                    if let Err(error) = room.local().unpublish_track(publication.sid()).await
                        && result.is_ok()
                    {
                        result = Err(error);
                    }
                }
                closed.send_replace(Some(result));
            });
        }
        loop {
            if let Some(result) = closed.borrow_and_update().clone() {
                return result;
            }
            if closed.changed().await.is_err() {
                return Err(RtcError::Closed("avatar media"));
            }
        }
    }
}

impl AvatarMedia for Media {
    fn publish(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move {
            if let Err(error) = self.publish_tracks().await {
                if let Err(cleanup) = self.close().await {
                    log::warn!("avatar publication rollback: {cleanup}");
                }
                return Err(provider_error(error));
            }
            Ok(())
        })
    }
    fn push(&self, frame: MediaFrame) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { self.synchronizer.push(frame, None).await.map_err(provider_error) })
    }
    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { self.close().await.map_err(provider_error) })
    }
}

fn provider_error(error: RtcError) -> ProviderError {
    match error {
        RtcError::Host(error) => {
            ProviderError::Host { error, api_kind: None, retryable: false, status_code: None }
        }
        RtcError::Stream(error) => error,
        error => lk_agents_abi::errors::ApiError::new(error.to_string()).into(),
    }
}
