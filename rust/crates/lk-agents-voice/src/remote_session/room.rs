//! The room's `lk.agent.session` byte streams as a [`SessionTransport`] (`RoomSessionTransport`):
//! one message per stream, read in the order the streams opened, written one stream at a time.

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};

use lk_agents_abi::errors::HostError;
use lk_agents_abi::voice::rtc::{ByteStream, Receiver, RtcRoom, StreamBytesOptions};
use lk_agents_abi::{Clock, Closed, HOST_TEARDOWN_TIMEOUT, Spawner, SpawnerExt};
use prost::Message as _;
#[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
use tokio::sync::Mutex as AsyncMutex;

use super::{SessionTransport, TOPIC_SESSION_MESSAGES, proto};

/// The Python logger every line here lands on.
const LOG: &str = "livekit.agents";

pub struct RoomTransport(Arc<Inner>);

struct Inner {
    room: Arc<dyn RtcRoom>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    state: Mutex<State>,
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    /// The streams the topic delivers; only the receive loop reads.
    incoming: AsyncMutex<Option<Receiver<ByteStream>>>,
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    /// One stream at a time: concurrent writers interleave on the same topic.
    sending: AsyncMutex<()>,
}

#[derive(Default)]
struct State {
    registered: bool,
    closed: bool,
}

impl RoomTransport {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pub fn new(room: Arc<dyn RtcRoom>, clock: Arc<dyn Clock>, spawner: Arc<dyn Spawner>) -> Self {
        RoomTransport(Arc::new(Inner {
            room,
            clock,
            spawner,
            state: Mutex::default(),
            incoming: AsyncMutex::new(None),
            sending: AsyncMutex::new(()),
        }))
    }
}

impl Inner {
    fn closed(&self) -> bool {
        self.state.lock().expect("transport state").closed
    }

    async fn send(&self, msg: &proto::AgentSessionMessage) -> Result<(), HostError> {
        let connection =
            self.room.connection_state().map_err(|error| HostError::new(error.to_string()))?;
        if self.closed() || !connection.is_connected() {
            return Err(HostError::new("room session transport is closed"));
        }
        let data = msg.encode_to_vec();
        let _turn = self.sending.lock().await;
        let opts = StreamBytesOptions {
            name: stream_name(),
            topic: TOPIC_SESSION_MESSAGES.to_owned(),
            ..StreamBytesOptions::default()
        };
        let sent = async {
            let writer = self.room.local().stream_bytes(opts).await?;
            let written = writer.write(&data).await;
            let closed = writer.close().await;
            match (written, closed) {
                (Ok(()), result) | (result, Ok(())) => result,
                (Err(error), Err(cleanup)) => Err(lk_agents_abi::voice::rtc::RtcError::host(
                    format!("{error}; stream cleanup failed: {cleanup}"),
                )),
            }
        };
        sent.await.map_err(|e| HostError::new(format!("failed to send binary stream message: {e}")))
    }
}

impl SessionTransport for RoomTransport {
    fn start(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move {
            let this = &self.0;
            {
                let mut state = this.state.lock().expect("transport state");
                if state.registered || state.closed {
                    return Ok(());
                }
                state.registered = true;
            }
            let streams = this
                .room
                .byte_streams(TOPIC_SESSION_MESSAGES)
                .map_err(|e| HostError::new(e.to_string()))?;
            *this.incoming.lock().await = Some(streams);
            Ok(())
        })
    }

    /// A dropped message is a request the client waits out in full, so a failure is raised.
    fn send_message<'a>(
        &'a self,
        msg: &'a proto::AgentSessionMessage,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async move { self.0.send(msg).await })
    }

    fn send_message_nowait(&self, msg: &proto::AgentSessionMessage) {
        let (this, msg) = (Arc::clone(&self.0), msg.clone());
        self.0.spawner.spawn_unboxed(async move {
            let _ = this.send(&msg).await;
        });
    }

    fn recv(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<proto::AgentSessionMessage>, HostError>> {
        Box::pin(async move {
            let mut incoming = self.0.incoming.lock().await;
            let Some(streams) = incoming.as_mut() else { return Ok(None) };
            while let Some(mut stream) =
                streams.next().await.map_err(|error| HostError::new(error.to_string()))?
            {
                let read = async {
                    let mut data = Vec::new();
                    while let Some(chunk) = stream
                        .chunks
                        .next()
                        .await
                        .map_err(|error| HostError::new(error.to_string()))?
                    {
                        if chunk.len() > super::MAX_MESSAGE_SIZE.saturating_sub(data.len()) {
                            return Err(HostError::new(
                                "room session message exceeds its byte limit",
                            ));
                        }
                        data.extend_from_slice(&chunk);
                    }
                    Ok(data)
                }
                .await;
                let deadline = self.0.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
                let closed = stream.close(deadline).await;
                let data = match (read, closed) {
                    (Ok(data), Closed::Clean) => data,
                    (Err(error), Closed::Clean) => return Err(error),
                    (read, cleanup) => {
                        return Err(HostError::new(format!(
                            "{}room session stream cleanup failed: {cleanup:?}",
                            read.err().map(|error| format!("{error}; ")).unwrap_or_default()
                        )));
                    }
                };
                match proto::AgentSessionMessage::decode(data.as_slice()) {
                    Ok(msg) => return Ok(Some(msg)),
                    Err(e) => {
                        log::warn!(target: LOG, error = e.to_string().as_str();
                        "failed to read binary stream message");
                    }
                }
            }
            Ok(None)
        })
    }

    fn close(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move {
            let registered = {
                let mut state = self.0.state.lock().expect("transport state");
                if state.closed {
                    return;
                }
                state.closed = true;
                std::mem::take(&mut state.registered)
            };
            // The handler's sender goes with it, which ends a receive waiting on the next stream.
            if registered {
                if let Err(error) =
                    self.0.room.unregister_byte_stream_handler(TOPIC_SESSION_MESSAGES)
                {
                    log::error!(target: LOG, "room session handler cleanup failed: {error}");
                }
            }
            if let Ok(mut incoming) = self.0.incoming.try_lock() {
                let source = incoming.take();
                drop(incoming);
                if let Some(source) = source {
                    let deadline = self.0.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
                    let closed = source.close(deadline).await;
                    if closed != Closed::Clean {
                        log::error!(target: LOG, "room session receiver cleanup failed: {closed:?}");
                    }
                }
            }
        })
    }
}

#[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
/// `utils.shortuuid("AS_")`.
fn stream_name() -> String {
    use std::hash::BuildHasher;
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = std::collections::hash_map::RandomState::new()
        .hash_one(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    format!("AS_{:012x}", n & 0xffff_ffff_ffff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_test_support::rtc::{ConnectionState, FakeRtc, StreamInfo};
    use proto::agent_session_message::Message;

    fn ping(id: &str) -> proto::AgentSessionMessage {
        proto::AgentSessionMessage {
            message: Some(Message::Request(proto::SessionRequest {
                request_id: id.to_owned(),
                request: Some(proto::session_request::Request::Ping(
                    proto::session_request::Ping {},
                )),
            })),
        }
    }

    // Catches: messages read out of order or lost to a malformed one, a sent message not on the
    // topic as one closed stream, a send on a disconnected room not raised, or a close leaving
    // the topic registered or a receive waiting.
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[tokio::test]
    async fn the_room_carries_one_message_per_stream() {
        let (rtc, clock) = FakeRtc::with_virtual_clock();
        let room = rtc.room("call");
        let transport = RoomTransport::new(
            room.clone() as Arc<dyn RtcRoom>,
            clock,
            Arc::new(lk_agents_abi::TokioSpawner::current()),
        );
        transport.start().await.unwrap();
        transport.start().await.expect("idempotent");
        let info = || StreamInfo { topic: TOPIC_SESSION_MESSAGES.into(), ..StreamInfo::default() };
        let first = ping("req_1").encode_to_vec();
        let (head, tail) = first.split_at(3);
        room.send_byte_stream("cli", info(), [head.to_vec(), tail.to_vec()]).unwrap();
        room.send_byte_stream("cli", info(), [vec![0xff, 0xff]]).unwrap();
        room.send_byte_stream("cli", info(), [ping("req_2").encode_to_vec()]).unwrap();
        let id = |m: proto::AgentSessionMessage| match m.message {
            Some(Message::Request(r)) => r.request_id,
            _ => panic!("not a request"),
        };
        assert_eq!(id(transport.recv().await.unwrap().unwrap()), "req_1");
        assert_eq!(id(transport.recv().await.unwrap().unwrap()), "req_2");

        transport.send_message(&ping("req_3")).await.unwrap();
        let sent = room.local_participant().byte_streams();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].opts.topic, TOPIC_SESSION_MESSAGES);
        assert!(sent[0].opts.name.starts_with("AS_") && sent[0].closed);
        assert_eq!(
            id(proto::AgentSessionMessage::decode(sent[0].chunks.concat().as_slice()).unwrap()),
            "req_3"
        );

        room.set_connection_state(ConnectionState::Disconnected);
        assert!(transport.send_message(&ping("req_4")).await.is_err());

        let waiting = {
            let transport = RoomTransport(Arc::clone(&transport.0));
            tokio::spawn(async move { transport.recv().await })
        };
        tokio::task::yield_now().await;
        transport.close().await;
        assert!(waiting.await.unwrap().unwrap().is_none());
        room.byte_streams(TOPIC_SESSION_MESSAGES).expect("the topic is free again");
    }
}
