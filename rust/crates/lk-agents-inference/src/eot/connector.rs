//! Binary EOT protocol over the shared WebSocket transport.

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use lk_agents_abi::errors::ApiError;
use lk_agents_net::websocket::Message;

use crate::eot::cloud::{WebSocketConnector, WebSocketSink, WebSocketStream};
use lk_agents_net::transport::TransportError;

type Socket = lk_agents_net::websocket::RawWebSocket;

/// `aiohttp.ClientSession.ws_connect(url, headers=...)` (`transports.py:_connect_ws`).
pub struct TungsteniteEotConnector;

#[async_trait]
impl WebSocketConnector for TungsteniteEotConnector {
    /// No timeout here: `CloudTransport::connect` bounds it with `conn_options.connect_timeout`.
    async fn connect(
        &self,
        url: &str,
        headers: &[(String, String)],
    ) -> Result<(Box<dyn WebSocketSink>, Box<dyn WebSocketStream>), ApiError> {
        let socket = lk_agents_net::websocket::connect_with_limit(url, headers.to_vec(), None)
            .await
            .map_err(translate)?;
        let (sink, stream) = socket.split();
        Ok((Box::new(TungsteniteEotSink { sink }), Box::new(TungsteniteEotStream { stream })))
    }
}

/// Every connect failure is non-retryable, as in `_connect_ws`: the cloud->local fallback is the
/// recovery.
fn translate(err: TransportError) -> ApiError {
    lk_agents_net::transport::connect_error("LiveKit turn detector", err).with_retryable(false)
}

struct TungsteniteEotSink {
    sink: futures_util::stream::SplitSink<Socket, Message>,
}

impl WebSocketSink for TungsteniteEotSink {
    fn send(&mut self, data: Vec<u8>) -> lk_agents_abi::BoxFuture<'_, Result<(), ApiError>> {
        Box::pin(async move {
            self.sink
                .send(Message::binary(data))
                .await
                .map_err(|e| translate(lk_agents_net::websocket::classify_tungstenite(e)))
        })
    }
}

struct TungsteniteEotStream {
    stream: futures_util::stream::SplitStream<Socket>,
}

impl WebSocketStream for TungsteniteEotStream {
    /// `None` is the peer closing. Text, ping and pong are skipped (tungstenite answers pings).
    fn recv(&mut self) -> lk_agents_abi::BoxFuture<'_, Option<Result<Vec<u8>, ApiError>>> {
        Box::pin(async move {
            loop {
                return match self.stream.next().await {
                    Some(Ok(Message::Binary(data))) => Some(Ok(data.to_vec())),
                    Some(Ok(Message::Close(_))) | None => None,
                    Some(Ok(_)) => continue,
                    Some(Err(e)) => {
                        Some(Err(translate(lk_agents_net::websocket::classify_tungstenite(e))))
                    }
                };
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_tungstenite::tungstenite::Error as WsError;

    // Catches: a connect failure arriving as `retryable = true`, delaying the local fallback.
    #[test]
    fn every_connect_failure_is_non_retryable() {
        for err in [
            TransportError::Timeout,
            TransportError::Connect("dns".to_owned()),
            TransportError::Closed,
            TransportError::Io("reset".to_owned()),
        ] {
            assert!(!translate(err).retryable);
        }
    }

    // Catches: a rejected handshake losing its HTTP status.
    #[test]
    fn a_rejected_handshake_keeps_its_status_and_its_body() {
        let response = tokio_tungstenite::tungstenite::http::Response::builder()
            .status(401)
            .body(Some(b"invalid inference grant".to_vec()))
            .expect("a response");
        let err = translate(lk_agents_net::websocket::classify_tungstenite(WsError::Http(
            Box::new(response),
        )));
        assert_eq!(err.status_code, Some(401));
        assert!(err.message.contains("invalid inference grant"));
        assert!(!err.retryable);
    }

    #[tokio::test]
    async fn an_unusable_header_fails_the_connection_rather_than_being_skipped() {
        let error = TungsteniteEotConnector
            .connect("ws://localhost", &[("invalid\nheader".into(), "value".into())])
            .await
            .err()
            .expect("invalid header fails before connecting");
        assert!(!error.retryable);
    }
}
