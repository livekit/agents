//! rmcp joins its HTTP worker but logs and swallows DELETE failures. Observe that SDK seam
//! so joining the worker cannot turn a remote session left behind into a Clean receipt.

use std::collections::HashMap;
use std::sync::Arc;

use futures_util::stream::BoxStream;
use reqwest::header::{HeaderName, HeaderValue};
use rmcp::model::ClientJsonRpcMessage;
use rmcp::transport::streamable_http_client::{
    StreamableHttpClient, StreamableHttpError, StreamableHttpPostResponse,
};
use sse_stream::{Error as SseError, Sse};

use super::McpError;
use super::transport::TransportReceipt;

#[derive(Clone)]
pub(super) struct CleanupClient {
    pub client: reqwest::Client,
    pub receipt: Arc<TransportReceipt>,
}

impl CleanupClient {
    fn observe_response(&self, response: StreamableHttpPostResponse) -> StreamableHttpPostResponse {
        match response {
            StreamableHttpPostResponse::Sse(stream, session) => {
                StreamableHttpPostResponse::Sse(self.receipt.track_reader(stream), session)
            }
            other => other,
        }
    }
}

impl StreamableHttpClient for CleanupClient {
    type Error = reqwest::Error;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        self.client
            .post_message(uri, message, session_id, auth_header, headers)
            .await
            .map(|response| self.observe_response(response))
    }

    async fn post_message_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        self.client
            .post_message_with_max_sse_event_size(
                uri,
                message,
                session_id,
                auth_header,
                headers,
                max_sse_event_size,
            )
            .await
            .map(|response| self.observe_response(response))
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        auth_header: Option<String>,
        headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), StreamableHttpError<Self::Error>> {
        self.receipt.update(|state| state.remote_session(true));
        let result = self.client.delete_session(uri, session_id, auth_header, headers).await;
        match result {
            Ok(()) => {
                self.receipt.update(|state| state.remote_session(false));
                Ok(())
            }
            // Per MCP this method is optional. Match the SDK's documented successful close.
            Err(error @ StreamableHttpError::ServerDoesNotSupportDeleteSession) => {
                self.receipt.update(|state| state.remote_session(false));
                Err(error)
            }
            Err(error) => {
                let cause = Arc::new(error);
                self.receipt.update(|state| state.fail(McpError::Transport(cause.clone())));
                Err(StreamableHttpError::Io(std::io::Error::other(cause)))
            }
        }
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, StreamableHttpError<Self::Error>> {
        self.client
            .get_stream(uri, session_id, last_event_id, auth_header, headers)
            .await
            .map(|stream| self.receipt.track_reader(stream))
    }

    async fn get_stream_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, StreamableHttpError<Self::Error>> {
        self.client
            .get_stream_with_max_sse_event_size(
                uri,
                session_id,
                last_event_id,
                auth_header,
                headers,
                max_sse_event_size,
            )
            .await
            .map(|stream| self.receipt.track_reader(stream))
    }
}
