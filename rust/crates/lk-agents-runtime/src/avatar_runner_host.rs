//! Native class construction uses the existing process composition; runner execution is injected.

use crate::avatar_runner::{AvatarRunner, RunnerError};
use lk_agents_abi::voice::avatar::{AvatarAudioReceiver, AvatarMedia, AvatarVideoGenerator};
use lk_agents_net::env::Env;
use std::sync::Arc;

impl lk_agents_abi::voice::avatar::AvatarRunner for AvatarRunner {
    fn new(
        room: lk_agents_abi::voice::room_owner::RoomSession,
        receiver: Arc<dyn AvatarAudioReceiver>,
        generator: Arc<dyn AvatarVideoGenerator>,
        media: Arc<dyn AvatarMedia>,
        lazy_publish: bool,
    ) -> Result<Self, lk_agents_abi::voice::rtc::RtcError> {
        let spawner = crate::process::Process::spawner()
            .map_err(lk_agents_abi::voice::rtc::RtcError::host)?;
        Self::new(
            room,
            receiver,
            generator,
            media,
            lazy_publish,
            crate::process::Process::clock(),
            spawner,
        )
    }
    async fn start(&self) -> Result<(), RunnerError> {
        Self::start(self).await
    }
    async fn wait_for_complete(&self) -> Result<(), RunnerError> {
        Self::wait_for_complete(self).await
    }
    fn clear_buffer(&self, epoch: u64) {
        Self::clear_buffer(self, epoch);
    }
    async fn aclose(&self) -> Result<(), RunnerError> {
        Self::aclose(self).await
    }
}
