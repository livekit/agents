//! The voice consumer's room ports with one binding's activation and cleanup authority.

use super::bindings::ActiveBinding;
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::voice::io::{AudioInput, AudioOutput, IoError, Result, TranscriptionOutput};
use lk_agents_abi::voice::room_owner::{
    RoomAudioInputPort, RoomIo, RoomIoEvent, RoomVideoInputPort,
};
use lk_agents_abi::voice::rtc::{ParticipantIdentity, RoomEvent};
use lk_agents_abi::{BoxFuture, Closed};
use std::sync::Arc;

pub(super) struct BoundRoomIo(pub ActiveBinding<Arc<dyn RoomIo>>);

impl BoundRoomIo {
    fn inner(&self) -> Arc<dyn RoomIo> {
        (*self.0).clone()
    }
}

impl RoomIo for BoundRoomIo {
    fn forward_events(
        self: Arc<Self>,
        listener: Arc<dyn Fn(RoomIoEvent) -> bool + Send + Sync>,
        rtc_listener: Arc<dyn Fn(RoomEvent) -> bool + Send + Sync>,
    ) {
        self.inner().forward_events(listener, rtc_listener);
    }

    fn audio_input(&self) -> Option<Arc<dyn AudioInput>> {
        self.0.audio_input()
    }

    fn audio_input_leg(&self) -> Option<Arc<dyn RoomAudioInputPort>> {
        self.0.audio_input_leg()
    }

    fn video_input(&self) -> Option<Arc<dyn RoomVideoInputPort>> {
        self.0.video_input()
    }

    fn audio_output(&self) -> Option<Arc<dyn AudioOutput>> {
        self.0.audio_output()
    }

    fn transcription_output(&self) -> Option<Arc<dyn TranscriptionOutput>> {
        self.0.transcription_output()
    }

    fn wants_transcript_sync(&self, external_audio: bool) -> bool {
        self.0.wants_transcript_sync(external_audio)
    }

    fn linked_participant(&self) -> Option<ParticipantIdentity> {
        self.0.linked_participant()
    }

    fn start(self: Arc<Self>, startup: crate::telemetry::Cx) {
        self.inner().start(startup);
    }

    fn session_started(&self) -> Result<()> {
        self.0.started().map_err(|error| IoError::Rtc(error.to_string()))
    }

    fn on_agent_state_changed(self: Arc<Self>, state: AgentState) {
        self.inner().on_agent_state_changed(state);
    }

    fn wait_for_ready(&self) -> BoxFuture<'_, Result<()>> {
        (**self.0).wait_for_ready()
    }

    fn set_participant(&self, identity: Option<ParticipantIdentity>) -> BoxFuture<'_, Result<()>> {
        (**self.0).set_participant(identity)
    }

    fn unset_participant(&self) -> BoxFuture<'_, Result<()>> {
        (**self.0).unset_participant()
    }

    fn on_user_input_transcribed<'a>(
        &'a self,
        transcript: &'a str,
        is_final: bool,
        segment: &'a str,
    ) -> BoxFuture<'a, Result<()>> {
        (**self.0).on_user_input_transcribed(transcript, is_final, segment)
    }

    fn on_user_input_corrected(&self, segment: &str, text: &str) {
        self.0.on_user_input_corrected(segment, text);
    }

    fn aclose(self: Arc<Self>) -> BoxFuture<'static, Closed> {
        self.0.request_close();
        Box::pin(async move { self.0.closed().await })
    }

    fn aclose_by(self: Arc<Self>, deadline: lk_agents_abi::Ts) -> BoxFuture<'static, Closed> {
        self.0.request_close_by(deadline);
        Box::pin(async move { self.0.closed().await })
    }
}
