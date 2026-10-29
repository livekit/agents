//! MCP's HTTP protocol adapter over the shared exchange and SSE decoder. The SDK owns session
//! recovery; transport attempts do not replay a tool POST with an uncertain outcome.

use std::collections::HashMap;
use std::sync::Arc;

use futures_util::{StreamExt, stream::BoxStream};
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::{ApiConnectOptions, Clock};
use lk_agents_net::http::GatewayHttp;
use lk_agents_net::http::exchange::{Exchange, HttpEvent, ResponsePolicy};
use lk_agents_net::transport::HttpRequest;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{Method, StatusCode};
use rmcp::model::{ClientJsonRpcMessage, ClientRequest, ErrorData, ServerJsonRpcMessage};
use rmcp::transport::streamable_http_client::{
    AuthRequiredError, InsufficientScopeError, StreamableHttpClient, StreamableHttpError,
    StreamableHttpPostResponse,
};
use sse_stream::{Error as SseError, Sse};

use super::McpError;
use super::transport::TransportReceipt;

type Error = StreamableHttpError<ApiError>;
const BODY_LIMIT: usize = 16 << 20;
const SESSION: &str = "Mcp-Session-Id";

#[derive(Clone)]
pub(super) struct CleanupClient {
    pub http: Arc<GatewayHttp>,
    pub headers: HeaderMap,
    pub options: ApiConnectOptions,
    pub clock: Arc<dyn Clock>,
    pub receipt: Arc<TransportReceipt>,
}

struct Response {
    exchange: Exchange,
    status: StatusCode,
    headers: HeaderMap,
}

impl CleanupClient {
    async fn request(
        &self,
        method: Method,
        uri: &str,
        body: Vec<u8>,
        headers: HeaderMap,
    ) -> Result<Response, Error> {
        let mut combined = self.headers.clone();
        combined.extend(headers);
        let headers = combined
            .iter()
            .map(|(name, value)| {
                Ok((
                    name.to_string(),
                    value
                        .to_str()
                        .map_err(|error| Error::UnexpectedServerResponse(error.to_string().into()))?
                        .to_owned(),
                ))
            })
            .collect::<Result<_, Error>>()?;
        let options = ApiConnectOptions { max_retry: 0, ..self.options };
        let mut exchange = Exchange::with_response_policy(
            self.http.clone(),
            method,
            HttpRequest { url: uri.into(), body, headers },
            &options,
            self.clock.clone(),
            ResponsePolicy::Protocol,
        );
        loop {
            match exchange.next().await.map_err(Error::Client)? {
                HttpEvent::Headers(headers) => {
                    let status = exchange.status().expect("response headers have a status");
                    return Ok(Response { exchange, status, headers });
                }
                HttpEvent::Started => {}
                _ => unreachable!("headers precede the response body"),
            }
        }
    }

    pub async fn legacy_get(&self, uri: &str) -> Result<lk_agents_net::sse::EventStream, Error> {
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::ACCEPT, HeaderValue::from_static("text/event-stream"));
        headers.insert(reqwest::header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
        let response = self.request(Method::GET, uri, Vec::new(), headers).await?;
        response.check_status()?;
        Ok(lk_agents_net::sse::EventStream::new(response.exchange, BODY_LIMIT))
    }

    pub async fn legacy_post(&self, uri: &str, message: ClientJsonRpcMessage) -> Result<(), Error> {
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let response =
            self.request(Method::POST, uri, serde_json::to_vec(&message)?, headers).await?;
        response.check_status()?;
        response.body().await.map(|_| ())
    }

    fn events(
        &self,
        response: Response,
        limit: usize,
        passive: bool,
    ) -> BoxStream<'static, Result<Sse, SseError>> {
        let mut stream = lk_agents_net::sse::EventStream::new(response.exchange, limit);
        stream.expect_response(!passive);
        let events = stream
            .into_stream()
            .map(|event| {
                event
                    .map(|event| Sse {
                        event: Some(event.event),
                        data: (!event.data.is_empty()).then_some(event.data),
                        id: event.id,
                        retry: event.retry,
                    })
                    .map_err(|error| SseError::Body(Box::new(error)))
            })
            .boxed();
        self.receipt.track_reader(events)
    }
}

impl Response {
    fn check_status(&self) -> Result<(), Error> {
        if self.status == StatusCode::UNAUTHORIZED || self.status == StatusCode::FORBIDDEN {
            if let Some(challenge) = self.headers.get(reqwest::header::WWW_AUTHENTICATE) {
                let challenge = challenge
                    .to_str()
                    .map_err(|error| Error::UnexpectedServerResponse(error.to_string().into()))?
                    .to_owned();
                return if self.status == StatusCode::UNAUTHORIZED {
                    Err(Error::AuthRequired(AuthRequiredError::new(challenge)))
                } else {
                    Err(Error::InsufficientScope(InsufficientScopeError::new(
                        challenge.clone(),
                        challenge_scope(&challenge),
                    )))
                };
            }
        }
        if self.status.is_success() {
            Ok(())
        } else {
            Err(Error::Client(ApiError::from_http(
                self.status.as_str(),
                self.status.as_u16().into(),
                None,
            )))
        }
    }

    async fn body(mut self) -> Result<Vec<u8>, Error> {
        let mut body = Vec::new();
        loop {
            match self.exchange.next().await.map_err(Error::Client)? {
                HttpEvent::Bytes(bytes) => {
                    if bytes.len() > BODY_LIMIT.saturating_sub(body.len()) {
                        return Err(Error::UnexpectedServerResponse(
                            "MCP response exceeds body limit".into(),
                        ));
                    }
                    body.extend_from_slice(&bytes);
                }
                HttpEvent::End => return Ok(body),
                _ => unreachable!("one MCP attempt, headers already consumed"),
            }
        }
    }
}

fn challenge_scope(header: &str) -> Option<String> {
    let offset = header.to_ascii_lowercase().find("scope=")? + "scope=".len();
    let value = &header[offset..];
    let scope = if let Some(quoted) = value.strip_prefix('"') {
        quoted.split_once('"')?.0
    } else {
        value.split([',', ';', ' ', '\t', '\r', '\n']).next()?
    };
    (!scope.is_empty()).then(|| scope.to_owned())
}

fn headers(
    session: Option<&str>,
    last_event: Option<String>,
    auth: Option<String>,
    custom: HashMap<HeaderName, HeaderValue>,
) -> Result<HeaderMap, Error> {
    let mut headers = HeaderMap::new();
    headers.insert(
        reqwest::header::ACCEPT,
        HeaderValue::from_static("text/event-stream, application/json"),
    );
    headers.insert(reqwest::header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    for (name, value) in [
        (SESSION, session.map(str::to_owned)),
        ("Last-Event-Id", last_event),
        ("Authorization", auth.map(|auth| format!("Bearer {auth}"))),
    ] {
        if let Some(value) = value {
            headers.insert(
                HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(&value)
                    .map_err(|error| Error::UnexpectedServerResponse(error.to_string().into()))?,
            );
        }
    }
    for (name, value) in custom {
        if ["accept", "mcp-session-id", "last-event-id"]
            .iter()
            .any(|reserved| name.as_str().eq_ignore_ascii_case(reserved))
        {
            return Err(Error::ReservedHeaderConflict(name.to_string()));
        }
        headers.insert(name, value);
    }
    Ok(headers)
}

impl StreamableHttpClient for CleanupClient {
    type Error = ApiError;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session: Option<Arc<str>>,
        auth: Option<String>,
        custom: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, Error> {
        self.post_message_with_max_sse_event_size(uri, message, session, auth, custom, BODY_LIMIT)
            .await
    }

    async fn post_message_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session: Option<Arc<str>>,
        auth: Option<String>,
        custom: HashMap<HeaderName, HeaderValue>,
        limit: usize,
    ) -> Result<StreamableHttpPostResponse, Error> {
        let headers = headers(session.as_deref(), None, auth, custom)?;
        let response =
            self.request(Method::POST, &uri, serde_json::to_vec(&message)?, headers).await?;
        if matches!(response.status, StatusCode::ACCEPTED | StatusCode::NO_CONTENT) {
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        if response.status == StatusCode::NOT_FOUND && session.is_some() {
            return Err(Error::SessionExpired);
        }
        let session_id =
            response.headers.get(SESSION).and_then(|value| value.to_str().ok()).map(str::to_owned);
        let content_type = response
            .headers
            .get(reqwest::header::CONTENT_TYPE)
            .map(|value| value.to_str().unwrap_or_default().to_owned());
        let no_reply = !matches!(message, ClientJsonRpcMessage::Request(_));
        if response.status.is_success()
            && no_reply
            && response
                .headers
                .get(reqwest::header::CONTENT_LENGTH)
                .is_some_and(|value| value == "0")
        {
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        if let Err(error) = response.check_status() {
            if matches!(error, Error::AuthRequired(_) | Error::InsufficientScope(_)) {
                return Err(error);
            }
            let status = response.status;
            let body = response.body().await?;
            let parsed = serde_json::from_slice::<ServerJsonRpcMessage>(&body);
            // An old server may reject discovery before a session exists; the SDK then probes
            // the legacy initialization protocol. Correlate that rejection with the probe.
            if session.is_none()
                && status.is_client_error()
                && !matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
            {
                if let ClientJsonRpcMessage::Request(request) = &message {
                    if matches!(request.request, ClientRequest::DiscoverRequest(_)) {
                        let error = match parsed {
                            Ok(ServerJsonRpcMessage::Error(error)) => error.error,
                            _ => ErrorData::invalid_request(
                                format!(
                                    "server/discover rejected with HTTP {status}: {}",
                                    String::from_utf8_lossy(&body)
                                ),
                                None,
                            ),
                        };
                        return Ok(StreamableHttpPostResponse::Json(
                            ServerJsonRpcMessage::error(error, Some(request.id.clone())),
                            None,
                        ));
                    }
                }
            }
            if content_type.as_deref().is_some_and(|value| value.starts_with("application/json")) {
                if let Ok(message @ ServerJsonRpcMessage::Error(_)) = parsed {
                    return Ok(StreamableHttpPostResponse::Json(message, session_id));
                }
            }
            return Err(Error::UnexpectedServerResponse(
                format!("HTTP {status}: {}", String::from_utf8_lossy(&body)).into(),
            ));
        }
        match content_type.as_deref() {
            Some(value) if value.starts_with("text/event-stream") => {
                Ok(StreamableHttpPostResponse::Sse(self.events(response, limit, false), session_id))
            }
            Some(value) if value.starts_with("application/json") => {
                let body = response.body().await?;
                match serde_json::from_slice(&body) {
                    Ok(message) => Ok(StreamableHttpPostResponse::Json(message, session_id)),
                    Err(_) if no_reply => Ok(StreamableHttpPostResponse::Accepted),
                    Err(error) => Err(Error::UnexpectedServerResponse(
                        format!(
                            "could not parse MCP response: {error}: {}",
                            String::from_utf8_lossy(&body[..body.len().min(256)])
                        )
                        .into(),
                    )),
                }
            }
            _ => Err(Error::UnexpectedContentType(content_type)),
        }
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session: Arc<str>,
        auth: Option<String>,
        custom: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), Error> {
        self.receipt.update(|state| state.remote_session(true));
        let result = async {
            let headers = headers(Some(&session), None, auth, custom)?;
            let response = self.request(Method::DELETE, &uri, Vec::new(), headers).await?;
            if response.status == StatusCode::METHOD_NOT_ALLOWED {
                return Ok(());
            }
            response.check_status()?;
            response.body().await.map(|_| ())
        }
        .await;
        match result {
            Ok(()) => {
                self.receipt.update(|state| state.remote_session(false));
                Ok(())
            }
            Err(error) => {
                let cause = Arc::new(error);
                self.receipt.update(|state| state.fail(McpError::Transport(cause.clone())));
                Err(Error::Io(std::io::Error::other(cause)))
            }
        }
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session: Option<Arc<str>>,
        last_event: Option<String>,
        auth: Option<String>,
        custom: HashMap<HeaderName, HeaderValue>,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, Error> {
        self.get_stream_with_max_sse_event_size(uri, session, last_event, auth, custom, BODY_LIMIT)
            .await
    }

    async fn get_stream_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        session: Option<Arc<str>>,
        last_event: Option<String>,
        auth: Option<String>,
        custom: HashMap<HeaderName, HeaderValue>,
        limit: usize,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, Error> {
        let headers = headers(session.as_deref(), last_event, auth, custom)?;
        let response = self.request(Method::GET, &uri, Vec::new(), headers).await?;
        if response.status == StatusCode::METHOD_NOT_ALLOWED {
            return Err(Error::ServerDoesNotSupportSse);
        }
        response.check_status()?;
        let content_type = response
            .headers
            .get(reqwest::header::CONTENT_TYPE)
            .map(|value| value.to_str().unwrap_or_default().to_owned());
        if !content_type.as_deref().is_some_and(|value| {
            value.starts_with("text/event-stream") || value.starts_with("application/json")
        }) {
            return Err(Error::UnexpectedContentType(content_type));
        }
        Ok(self.events(response, limit, true))
    }
}

#[cfg(test)]
mod tests;
