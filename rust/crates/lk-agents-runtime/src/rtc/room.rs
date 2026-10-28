//! A supplied SDK room plus runtime-owned subscriptions and waits.

use super::*;
use lk_agents_abi::BoxFuture;
use lk_agents_voice::room_io::wait::{Ready, WaitFor};
use std::sync::Arc;

pub struct Room {
    sdk: Arc<dyn RtcRoom>,
}

impl Room {
    pub fn wrap(sdk: Arc<dyn RtcRoom>) -> Arc<dyn RtcRoom> {
        Arc::new(Self { sdk })
    }
}

impl RtcRoom for Room {
    fn name(&self) -> Result<String> {
        self.sdk.name()
    }
    fn connection_state(&self) -> Result<ConnectionState> {
        self.sdk.connection_state()
    }
    fn local(&self) -> &Arc<dyn LocalParticipant> {
        self.sdk.local()
    }
    fn participant(
        &self,
        identity: &ParticipantIdentity,
    ) -> Result<Option<Arc<dyn RemoteParticipant>>> {
        self.sdk.participant(identity)
    }
    fn remote_participants(&self) -> Result<Vec<Arc<dyn RemoteParticipant>>> {
        self.sdk.remote_participants()
    }
    fn event_bus(&self) -> &RoomEventBus {
        self.sdk.event_bus()
    }
    fn register_byte_stream_handler(&self, topic: &str, streams: ByteStreamSender) -> Result<()> {
        self.sdk.register_byte_stream_handler(topic, streams)
    }
    fn unregister_byte_stream_handler(&self, topic: &str) -> Result<()> {
        self.sdk.unregister_byte_stream_handler(topic)
    }
    fn register_text_stream_handler(&self, topic: &str, streams: TextStreamSender) -> Result<()> {
        self.sdk.register_text_stream_handler(topic, streams)
    }
    fn unregister_text_stream_handler(&self, topic: &str) -> Result<()> {
        self.sdk.unregister_text_stream_handler(topic)
    }
    fn byte_streams(&self, topic: &str) -> Result<Receiver<ByteStream>> {
        let (sender, receiver) = events::ByteStreamSender::channel();
        self.register_byte_stream_handler(topic, sender)?;
        Ok(receiver)
    }
    fn text_streams(&self, topic: &str) -> Result<Receiver<TextStream>> {
        let (sender, receiver) = events::TextStreamSender::channel();
        self.register_text_stream_handler(topic, sender)?;
        Ok(receiver)
    }
    fn wait_for_participant<'a>(
        &'a self,
        identity: Option<String>,
        kinds: Option<Vec<i32>>,
        include_local: bool,
    ) -> BoxFuture<'a, Result<Option<ParticipantIdentity>>> {
        Box::pin(async move {
            match wait::perform(self, WaitFor::Participant { identity, kinds, include_local })
                .await?
            {
                Ready::Participant(identity) => Ok(identity),
                _ => unreachable!(),
            }
        })
    }
    fn wait_for_agent<'a>(
        &'a self,
        name: Option<String>,
    ) -> BoxFuture<'a, Result<ParticipantIdentity>> {
        Box::pin(async move {
            match wait::perform(self, WaitFor::Agent { name }).await? {
                Ready::Agent(identity) => Ok(identity),
                _ => unreachable!(),
            }
        })
    }
    fn wait_for_participant_attribute<'a>(
        &'a self,
        identity: String,
        attribute: String,
        value: String,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            match wait::perform(self, WaitFor::Attribute { identity, attribute, value }).await? {
                Ready::Attribute => Ok(()),
                _ => unreachable!(),
            }
        })
    }
    fn wait_for_track_publication<'a>(
        &'a self,
        identity: Option<String>,
        kinds: Option<Vec<i32>>,
        include_local: bool,
        subscribed: bool,
    ) -> BoxFuture<'a, Result<(Option<ParticipantIdentity>, TrackSid)>> {
        Box::pin(async move {
            match wait::perform(self, WaitFor::Track { identity, kinds, include_local, subscribed })
                .await?
            {
                Ready::Track(identity, sid) => Ok((identity, sid)),
                _ => unreachable!(),
            }
        })
    }
}
