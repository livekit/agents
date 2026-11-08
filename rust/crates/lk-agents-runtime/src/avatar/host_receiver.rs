//! `HostAudioReceiver`: the avatar runner's receiver over a host `AudioReceiver`. Its audio is
//! read from a native queue: `QueueAudioOutput`'s, or one the host's iterator is pumped into,
//! stamped with the clear epoch it was captured in.

use crate::telemetry;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::utils::video::AudioReceiverFrame;
use lk_agents_abi::voice::avatar::{
    self, AudioReceiverHost, AudioReceiverObservation, AvatarAudio, AvatarAudioReceiver,
};
use lk_agents_abi::voice::io::AvatarPlaybackSegment;
use lk_agents_abi::{BoxFuture, OwnedTask, Spawner, SpawnerExt};
use lk_agents_media::media_queue::{self, AudioReceiver, AudioSender, Item, Policy};
use lk_agents_net::env::ProcessEnv;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::watch;

/// How far a host receiver's audio may run ahead of the runner before its oldest is dropped.
const HOST_AUDIO_POLICY: Policy = Policy::CatchUp(Duration::from_secs(30));

pub struct HostAudioReceiver {
    host: Arc<dyn AudioReceiverHost>,
    audio: Audio,
}

enum Audio {
    /// `QueueAudioOutput`'s queue, which stamps and reports its own clears.
    Native(Arc<dyn AvatarAudio>),
    Pumped(Pumped),
}

/// What the pump queues beside the audio, in order with it.
enum Event {
    SegmentEnd,
    /// A clear: the audio after it is stamped with this epoch.
    Cleared(u64),
    Failed(ProviderError),
    End,
}

struct Pumped {
    sender: Arc<AudioSender<Event>>,
    /// The queue, the epoch of the audio it yields, and whether it ended.
    receiver: tokio::sync::Mutex<(AudioReceiver<Event>, u64, bool)>,
    clears: watch::Sender<u64>,
    spawner: Arc<dyn Spawner>,
    /// Pulls the host's iterator from the first read on.
    pump: OnceLock<OwnedTask>,
}

impl avatar::HostAudioReceiver for HostAudioReceiver {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        receiver: Arc<dyn AudioReceiverHost>,
        audio: Option<Arc<dyn AvatarAudio>>,
    ) -> Result<Self, ProviderError> {
        let audio = match audio {
            Some(audio) => Audio::Native(audio),
            None => {
                let spawner = (env.spawner)().map_err(|e| ApiError::new(e.to_string()))?;
                let (sender, queue) = media_queue::audio(
                    HOST_AUDIO_POLICY,
                    telemetry::AudioConsumer::AvatarReceiver,
                    env.clock.clone(),
                );
                Audio::Pumped(Pumped {
                    sender: Arc::new(sender),
                    receiver: tokio::sync::Mutex::new((queue, 0, false)),
                    clears: watch::Sender::new(0),
                    spawner,
                    pump: OnceLock::new(),
                })
            }
        };
        Ok(Self { host: receiver, audio })
    }

    fn clear_buffer(&self) {
        let Audio::Pumped(pumped) = &self.audio else { return };
        pumped.clears.send_modify(|epoch| *epoch += 1);
        let epoch = *pumped.clears.borrow();
        if pumped.sender.flush(Event::Cleared(epoch)).is_err() {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "avatar audio receiver stopped draining; clear_buffer dropped");
        }
    }
}

async fn pump(host: Arc<dyn AudioReceiverHost>, sender: Arc<AudioSender<Event>>) {
    let end = loop {
        break match host.next_frame(&telemetry::Cx::NONE).await {
            Ok(Some(AudioReceiverFrame::Audio(frame))) => {
                sender.push(frame);
                continue;
            }
            Ok(Some(AudioReceiverFrame::SegmentEnd)) => {
                if sender.event(Event::SegmentEnd).is_ok() {
                    continue;
                }
                Event::Failed(ApiError::new("the avatar runner stopped reading audio").into())
            }
            Ok(None) => Event::End,
            Err(error) => Event::Failed(error),
        };
    };
    let _ = sender.flush(end);
}

impl Pumped {
    async fn next_audio(
        &self,
        host: &Arc<dyn AudioReceiverHost>,
    ) -> Result<AudioReceiverObservation, ProviderError> {
        self.pump.get_or_init(|| self.spawner.spawn(pump(host.clone(), self.sender.clone())));
        let mut queue = self.receiver.lock().await;
        let (receiver, epoch, ended) = &mut *queue;
        let observation =
            |epoch, frame| AudioReceiverObservation { epoch, frame, ..Default::default() };
        while !*ended {
            let frame = match receiver.recv().await {
                Some(Item::Media(frame)) => AudioReceiverFrame::Audio(frame),
                Some(Item::Gap(_)) => continue,
                Some(Item::Event(Event::Cleared(cleared))) => {
                    *epoch = cleared;
                    continue;
                }
                Some(Item::Event(Event::SegmentEnd)) => AudioReceiverFrame::SegmentEnd,
                Some(Item::Event(Event::Failed(error))) => {
                    *ended = true;
                    return Err(error);
                }
                Some(Item::Event(Event::End)) | None => break,
            };
            return Ok(observation(*epoch, Some(frame)));
        }
        *ended = true;
        Ok(observation(*epoch, None))
    }
}

impl AvatarAudio for HostAudioReceiver {
    fn next_audio(&self) -> BoxFuture<'_, Result<AudioReceiverObservation, ProviderError>> {
        match &self.audio {
            Audio::Native(audio) => audio.next_audio(),
            Audio::Pumped(pumped) => Box::pin(pumped.next_audio(&self.host)),
        }
    }

    fn next_clear(&self, after: u64) -> BoxFuture<'_, Option<u64>> {
        match &self.audio {
            Audio::Native(audio) => audio.next_clear(after),
            Audio::Pumped(pumped) => Box::pin(async move {
                let mut clears = pumped.clears.subscribe();
                clears.wait_for(|epoch| *epoch > after).await.ok().map(|epoch| *epoch)
            }),
        }
    }
}

impl AvatarAudioReceiver for HostAudioReceiver {
    fn notify_playback_started(
        &self,
        segment: AvatarPlaybackSegment,
    ) -> BoxFuture<'_, Result<(), ProviderError>> {
        self.host.notify_playback_started(&telemetry::Cx::NONE, segment)
    }

    fn notify_playback_finished(
        &self,
        playback_position: f64,
        interrupted: bool,
        segment: AvatarPlaybackSegment,
    ) -> BoxFuture<'_, Result<(), ProviderError>> {
        self.host.notify_playback_finished(
            &telemetry::Cx::NONE,
            playback_position,
            interrupted,
            segment,
        )
    }

    fn aclose(&self) -> BoxFuture<'_, Result<(), ProviderError>> {
        self.host.aclose(&telemetry::Cx::NONE)
    }
}
