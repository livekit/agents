//! I/O for a session with no RTC attachment or room resources.

use lk_agents_abi::voice::io::{AudioInput, AudioOutput, Result, TextOutput};
use lk_agents_abi::voice::room_owner;
use lk_agents_abi::voice::rtc::ParticipantIdentity;
use std::sync::Arc;

pub fn room_io() -> Arc<dyn room_owner::RoomIo> {
    Arc::new(DetachedRoomIo)
}

struct DetachedRoomIo;

impl room_owner::RoomIo for DetachedRoomIo {
    fn forward_events(
        self: Arc<Self>,
        _: Arc<dyn Fn(room_owner::RoomIoEvent) -> bool + Send + Sync>,
        _: Arc<dyn Fn(lk_agents_abi::voice::rtc::RoomEvent) -> bool + Send + Sync>,
    ) {
    }
    fn audio_input(&self) -> Option<Arc<dyn AudioInput>> {
        None
    }
    fn audio_input_leg(&self) -> Option<Arc<dyn room_owner::RoomAudioInputPort>> {
        None
    }
    fn video_input(&self) -> Option<Arc<dyn room_owner::RoomVideoInputPort>> {
        None
    }
    fn audio_output(&self) -> Option<Arc<dyn AudioOutput>> {
        None
    }
    fn transcription_output(&self) -> Option<Arc<dyn TextOutput>> {
        None
    }
    fn wants_transcript_sync(&self, _: bool) -> bool {
        false
    }
    fn linked_participant(&self) -> Option<ParticipantIdentity> {
        None
    }
    fn start(self: Arc<Self>, _: lk_agents_abi::telemetry::record::TraceContext) {}
    fn on_agent_state_changed(self: Arc<Self>, _: &str) {}
    fn wait_for_ready<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(()) })
    }
    fn set_participant<'a>(
        &'a self,
        identity: Option<ParticipantIdentity>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if identity.is_some() {
                Err(lk_agents_abi::voice::io::IoError::Rtc("the session has no RTC room".into()))
            } else {
                Ok(())
            }
        })
    }
    fn unset_participant<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(()) })
    }
    fn on_user_input_transcribed<'a>(
        &'a self,
        _: &'a str,
        _: bool,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(()) })
    }
    fn aclose(self: Arc<Self>) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        Box::pin(async move { lk_agents_abi::Closed::Clean })
    }
}
