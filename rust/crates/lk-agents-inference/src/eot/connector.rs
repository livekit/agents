//! The one [`WebSocketConnector`] that opens a real socket, over `tokio-tungstenite`. Binary
//! (protobuf), so separate from [`crate::native`]'s text transport, but it shares
//! [`classify_tungstenite`](crate::native::classify_tungstenite) for handshake statuses.

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use lk_agents_abi::errors::ApiError;
use tokio_tungstenite::tungstenite::Message;

use crate::eot::cloud::{WebSocketConnector, WebSocketSink, WebSocketStream};
use crate::transport::TransportError;

type Socket = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

/// `aiohttp.ClientSession.ws_connect(url, headers=...)` (`transports.py:_connect_ws`).
pub struct TungsteniteEotConnector;

#[async_trait]
impl WebSocketConnector for TungsteniteEotConnector {
    /// No timeout here: `CloudTransport::connect` bounds it with `conn_options.timeout`.
    async fn connect(
        &self,
        url: &str,
        headers: &[(String, String)],
    ) -> Result<(Box<dyn WebSocketSink>, Box<dyn WebSocketStream>), ApiError> {
        use tokio_tungstenite::tungstenite::client::IntoClientRequest;
        use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};

        let mut request = url.into_client_request().map_err(|e| connect_failed(url, e))?;
        for (name, value) in headers {
            let name = HeaderName::from_bytes(name.as_bytes()).map_err(header_failed)?;
            let value = HeaderValue::from_str(value).map_err(header_failed)?;
            request.headers_mut().insert(name, value);
        }

        let (socket, _response) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| translate(crate::native::classify_tungstenite(e)))?;
        let (sink, stream) = socket.split();
        Ok((Box::new(TungsteniteEotSink { sink }), Box::new(TungsteniteEotStream { stream })))
    }
}

/// Every connect failure is non-retryable, as in `_connect_ws`: the cloud->local fallback is the
/// recovery.
fn translate(err: TransportError) -> ApiError {
    crate::transport::connect_error("LiveKit turn detector", err).with_retryable(false)
}

fn connect_failed(url: &str, err: impl std::fmt::Display) -> ApiError {
    translate(TransportError::Connect(format!("{url}: {err}")))
}

/// A header the host built that `http` will not accept; reported, not skipped.
fn header_failed(err: impl std::fmt::Display) -> ApiError {
    translate(TransportError::Connect(format!("invalid inference header: {err}")))
}

struct TungsteniteEotSink {
    sink: futures_util::stream::SplitSink<Socket, Message>,
}

#[async_trait]
impl WebSocketSink for TungsteniteEotSink {
    async fn send(&mut self, data: Vec<u8>) -> Result<(), ApiError> {
        self.sink
            .send(Message::binary(data))
            .await
            .map_err(|e| translate(crate::native::classify_tungstenite(e)))
    }
}

struct TungsteniteEotStream {
    stream: futures_util::stream::SplitStream<Socket>,
}

#[async_trait]
impl WebSocketStream for TungsteniteEotStream {
    /// `None` is the peer closing. Text, ping and pong are skipped (tungstenite answers pings).
    async fn recv(&mut self) -> Option<Result<Vec<u8>, ApiError>> {
        loop {
            return match self.stream.next().await {
                Some(Ok(Message::Binary(data))) => Some(Ok(data.to_vec())),
                Some(Ok(Message::Close(_))) | None => None,
                Some(Ok(_)) => continue,
                Some(Err(e)) => Some(Err(translate(crate::native::classify_tungstenite(e)))),
            };
        }
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
        let err = translate(crate::native::classify_tungstenite(WsError::Http(Box::new(response))));
        assert_eq!(err.status_code, Some(401));
        assert!(err.message.contains("invalid inference grant"));
        assert!(!err.retryable);
    }

    // Catches: a header the host built being dropped instead of reported.
    #[test]
    fn an_unusable_header_fails_the_connection_rather_than_being_skipped() {
        let err = header_failed("invalid HTTP header name");
        assert!(err.message.contains("invalid inference header"));
        assert!(!err.retryable);
    }
}
