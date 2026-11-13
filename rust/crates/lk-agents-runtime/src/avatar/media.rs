use crate::rtc::synchronizer::AVSynchronizer;
use crate::telemetry;
use lk_agents_abi::Spawner;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::video::MediaFrame;
use lk_agents_abi::voice::avatar::AvatarMedia;
use lk_agents_abi::voice::rtc::media::{AVSynchronizer as _, VideoTrackOptions};
use lk_agents_abi::voice::rtc::{RtcError, RtcRoom, TrackPublication, TrackPublishOptions};
use lk_agents_net::owned::Background;
use lk_agents_net::signal;
use lk_agents_net::task::TaskSet;
use lk_agents_voice::av_sync::MediaPlan;
use parking_lot::Mutex;
use std::sync::Arc;

/// The close's outcome, once it ran.
type Closing = Option<Result<(), RtcError>>;

pub struct Media {
    room: Arc<dyn RtcRoom>,
    pub synchronizer: Arc<AVSynchronizer>,
    audio: Arc<dyn lk_agents_abi::voice::rtc::AudioSource>,
    video: Arc<dyn lk_agents_abi::voice::rtc::VideoSource>,
    audio_options: TrackPublishOptions,
    video_options: VideoTrackOptions,
    publications: Arc<Mutex<Vec<Arc<dyn TrackPublication>>>>,
    /// Taken by the first close, whose task holds it.
    close: Mutex<Option<signal::Producer<Closing>>>,
    closed: signal::Cell<Closing>,
    close_tasks: Arc<TaskSet>,
    spawner: Arc<dyn Spawner>,
}

impl Media {
    /// `background` runs a rollback no one can await: the process's tasks.
    pub fn new(
        room: Arc<dyn RtcRoom>,
        plan: MediaPlan,
        background: Background,
    ) -> Result<Arc<Self>, RtcError> {
        let video = room.create_video_source(plan.video.clone())?;
        let audio = match room.create_audio_source(plan.audio.clone()) {
            Ok(audio) => audio,
            Err(error) => {
                // The constructor cannot await: the close outlives it.
                background.cleanup("rtc source rollback", async move {
                    if let Err(error) = video.close(&telemetry::Cx::NONE).await {
                        log::warn!(target: lk_agents_abi::LOG_TARGET, "RTC source rollback: {error}");
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
            background.clock,
            background.spawner.clone(),
        );
        let (close, closed) = signal::once();
        Ok(Arc::new(Self {
            room,
            synchronizer,
            audio,
            video,
            audio_options: plan.audio,
            video_options: plan.video,
            publications: Arc::default(),
            close: Mutex::new(Some(close)),
            closed,
            close_tasks: Arc::default(),
            spawner: background.spawner,
        }))
    }

    async fn publish_tracks(&self) -> Result<(), RtcError> {
        let local = self.room.local();
        let audio = local
            .publish_audio_source(
                &telemetry::Cx::NONE,
                self.audio.clone(),
                self.audio_options.clone(),
            )
            .await?;
        self.publications.lock().push(audio.clone());
        audio.wait_for_subscription(&telemetry::Cx::NONE).await?;
        let video = local
            .publish_video_source(
                &telemetry::Cx::NONE,
                self.video.clone(),
                self.video_options.clone(),
            )
            .await?;
        self.publications.lock().push(video);
        Ok(())
    }

    async fn close(&self) -> Result<(), RtcError> {
        let close = self.close.lock().take();
        if let Some(closed) = close {
            let (publications, synchronizer, room) =
                (self.publications.clone(), self.synchronizer.clone(), self.room.clone());
            self.close_tasks.spawn(&*self.spawner, "avatar RTC tracks close", async move {
                let publications = std::mem::take(&mut *publications.lock());
                let mut result = synchronizer.aclose().await;
                for publication in publications {
                    if let Err(error) =
                        room.local().unpublish_track(&telemetry::Cx::NONE, publication.sid()).await
                        && result.is_ok()
                    {
                        result = Err(error);
                    }
                }
                closed.resolve(result);
            });
        }
        let closed = self.closed.wait().await;
        closed.unwrap_or(Err(RtcError::Closed("avatar media")))
    }
}

impl AvatarMedia for Media {
    fn publish(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move {
            if let Err(error) = self.publish_tracks().await {
                if let Err(cleanup) = self.close().await {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, "avatar publication rollback: {cleanup}");
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
        RtcError::Host(error) => error.into(),
        RtcError::Stream(error) => error,
        error => lk_agents_abi::errors::ApiError::new(
            lk_agents_abi::errors::ErrorKind::Connection,
            error.to_string(),
        )
        .into(),
    }
}
