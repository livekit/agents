//! Runtime construction and typed receipts for the avatar's native audio receiver.

use std::sync::Arc;

use crate::avatar::receiver::DataStreamAudioReceiver as Receiver;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::utils::video::AudioReceiverFrame;
use lk_agents_abi::voice::avatar::{self, AvatarAudioReceiver, ClearBufferListener};
use lk_agents_abi::voice::io::AvatarPlaybackSegment;
use lk_agents_abi::voice::rtc::RtcRoom;
use lk_agents_abi::{BoxFuture, TokioSpawner};
use lk_agents_voice::avatar::ReceiverEnd;

type AudioReceipt = (u64, bool, Option<AvatarPlaybackSegment>, Option<AudioReceiverFrame>);

pub struct DataStreamAudioReceiver {
    receiver: Receiver,
}

fn failure(error: ReceiverEnd) -> ProviderError {
    ApiError::new(error.to_string()).into()
}

impl avatar::DataStreamAudioReceiver for DataStreamAudioReceiver {
    fn new(
        room: Arc<dyn RtcRoom>,
        sender_identity: Option<String>,
        frame_size_ms: u32,
        rpc_max_retries: u32,
        on_clear_buffer: Option<Arc<dyn ClearBufferListener>>,
    ) -> Result<Self, ProviderError> {
        if frame_size_ms == 0 {
            return Err(ApiError::new("frame_size_ms must be positive").into());
        }
        let handle = crate::process::handle().map_err(|error| ApiError::new(error.to_string()))?;
        let on_clear = on_clear_buffer.map(|listener| {
            Arc::new(move || listener.clear_buffer()) as Arc<dyn Fn() + Send + Sync>
        });
        Ok(Self {
            receiver: Receiver::new(
                crate::rtc::room::Room::wrap(room),
                sender_identity,
                frame_size_ms,
                rpc_max_retries,
                on_clear,
                crate::process::host_clock(),
                Arc::new(TokioSpawner::new(handle)),
            ),
        })
    }

    async fn start(&self) -> Result<(), ProviderError> {
        self.receiver.start().await.map_err(failure)
    }

    async fn next_audio(&self) -> Result<AudioReceipt, ProviderError> {
        match self.receiver.recv().await {
            Ok(receipt) => Ok(receipt),
            Err(ReceiverEnd::Closed) => Ok((0, false, None, None)),
            Err(error) => Err(failure(error)),
        }
    }

    fn notify_playback_started(&self, segment: AvatarPlaybackSegment) -> Result<(), ProviderError> {
        self.receiver.notify_playback_started(segment).map_err(failure)
    }

    fn notify_playback_finished(
        &self,
        segment: AvatarPlaybackSegment,
        playback_position: f64,
        interrupted: bool,
    ) -> Result<(), ProviderError> {
        self.receiver
            .notify_playback_finished(segment, playback_position, interrupted)
            .map_err(failure)
    }

    fn aclose(&self) -> Result<(), ProviderError> {
        self.receiver.aclose();
        Ok(())
    }
}

impl AvatarAudioReceiver for DataStreamAudioReceiver {
    fn open_audio(&self) -> BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(avatar::DataStreamAudioReceiver::start(self))
    }
    fn next_audio(&self) -> BoxFuture<'_, Result<AudioReceipt, ProviderError>> {
        Box::pin(avatar::DataStreamAudioReceiver::next_audio(self))
    }
    fn notify_playback_started(
        &self,
        segment: AvatarPlaybackSegment,
    ) -> BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(
            async move { avatar::DataStreamAudioReceiver::notify_playback_started(self, segment) },
        )
    }
    fn notify_playback_finished(
        &self,
        segment: AvatarPlaybackSegment,
        playback_position: f64,
        interrupted: bool,
    ) -> BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move {
            avatar::DataStreamAudioReceiver::notify_playback_finished(
                self,
                segment,
                playback_position,
                interrupted,
            )
        })
    }
    fn aclose(&self) -> BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { avatar::DataStreamAudioReceiver::aclose(self) })
    }
}

impl Drop for DataStreamAudioReceiver {
    fn drop(&mut self) {
        self.receiver.aclose();
    }
}
