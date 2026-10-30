//! The room's `lk.agent.session` byte streams as a [`SessionTransport`] (`RoomSessionTransport`):
//! one message per stream, read in the order the streams opened, written one stream at a time.

use parking_lot::Mutex;
use std::sync::Arc;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::voice::rtc::{ByteStream, Receiver, RtcRoom, StreamBytesOptions};
use lk_agents_abi::{Clock, Closed, HOST_TEARDOWN_TIMEOUT};
use prost::Message as _;
use tokio::sync::Mutex as AsyncMutex;

use super::{SessionTransport, TOPIC_SESSION_MESSAGES, proto};

/// The Python logger every line here lands on.
const LOG: &str = "livekit.agents";

pub struct RoomTransport(Arc<Inner>);

struct Inner {
    room: Arc<dyn RtcRoom>,
    clock: Arc<dyn Clock>,
    state: Mutex<State>,
    /// The streams the topic delivers; only the receive loop reads.
    incoming: AsyncMutex<Option<Receiver<ByteStream>>>,
    /// One stream at a time: concurrent writers interleave on the same topic.
    sending: AsyncMutex<()>,
}

#[derive(Default)]
struct State {
    registered: bool,
    closed: bool,
}

impl RoomTransport {
    pub fn new(room: Arc<dyn RtcRoom>, clock: Arc<dyn Clock>) -> Self {
        RoomTransport(Arc::new(Inner {
            room,
            clock,
            state: Mutex::default(),
            incoming: AsyncMutex::new(None),
            sending: AsyncMutex::new(()),
        }))
    }
}

impl Inner {
    fn closed(&self) -> bool {
        self.state.lock().closed
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
                let mut state = this.state.lock();
                if state.registered || state.closed {
                    return Ok(());
                }
                state.registered = true;
            }
            let streams =
                crate::rtc::events::byte_streams(this.room.as_ref(), TOPIC_SESSION_MESSAGES)
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
                let mut state = self.0.state.lock();
                if state.closed {
                    return;
                }
                state.closed = true;
                std::mem::take(&mut state.registered)
            };
            // The handler's sender goes with it, which ends a receive waiting on the next stream.
            if registered
                && let Err(error) =
                    self.0.room.unregister_byte_stream_handler(TOPIC_SESSION_MESSAGES)
            {
                log::error!(target: LOG, "room session handler cleanup failed: {error}");
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

/// `utils.shortuuid("AS_")`.
fn stream_name() -> String {
    use std::hash::BuildHasher;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = std::collections::hash_map::RandomState::new()
        .hash_one(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    format!("AS_{:012x}", n & 0xffff_ffff_ffff)
}
