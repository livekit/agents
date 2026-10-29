//! MCP's HTTP protocol adapter over the shared exchange and SSE decoder. The SDK owns session
//! recovery; transport attempts do not replay a tool POST with an uncertain outcome.

use std::collections::HashMap;
use std::sync::Arc;

use futures_util::{StreamExt, stream::BoxStream};
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::{ApiConnectOptions, Clock};
use lk_agents_core::mcp::http::{
    Challenge, Failure, MessageKind, PostReply, PostResponse, Rejected, SESSION_HEADER,
    StreamReply, check_status, delete_status, post_reply, reserved_header, stream_reply,
};
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
        response.check(check_status(response.status.as_u16(), response.challenge()?))?;
        Ok(lk_agents_net::sse::EventStream::new(response.exchange, BODY_LIMIT))
    }

    pub async fn legacy_post(&self, uri: &str, message: ClientJsonRpcMessage) -> Result<(), Error> {
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let response =
            self.request(Method::POST, uri, serde_json::to_vec(&message)?, headers).await?;
        response.check(check_status(response.status.as_u16(), response.challenge()?))?;
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

fn challenge_error(challenge: Challenge) -> Error {
    match challenge {
        Challenge::AuthRequired(challenge) => {
            Error::AuthRequired(AuthRequiredError::new(challenge))
        }
        Challenge::InsufficientScope { challenge, scope } => {
            Error::InsufficientScope(InsufficientScopeError::new(challenge, scope))
        }
    }
}

impl Response {
    fn header(&self, name: impl reqwest::header::AsHeaderName) -> Option<&str> {
        self.headers.get(name).map(|value| value.to_str().unwrap_or_default())
    }

    fn challenge(&self) -> Result<Option<&str>, Error> {
        self.headers
            .get(reqwest::header::WWW_AUTHENTICATE)
            .map(|value| {
                value
                    .to_str()
                    .map_err(|error| Error::UnexpectedServerResponse(error.to_string().into()))
            })
            .transpose()
    }

    fn failure(&self, failure: Failure) -> Error {
        match failure {
            Failure::Challenge(challenge) => challenge_error(challenge),
            Failure::Status => Error::Client(ApiError::from_http(
                self.status.as_str(),
                self.status.as_u16().into(),
                None,
            )),
        }
    }

    fn check(&self, status: Result<(), Failure>) -> Result<(), Error> {
        status.map_err(|failure| self.failure(failure))
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
        (SESSION_HEADER, session.map(str::to_owned)),
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
        if reserved_header(name.as_str()) {
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
        let session_id = response.header(SESSION_HEADER).map(str::to_owned);
        let reply = post_reply(PostResponse {
            status: response.status.as_u16(),
            has_session: session.is_some(),
            message: match &message {
                ClientJsonRpcMessage::Request(request) => match request.request {
                    ClientRequest::DiscoverRequest(_) => MessageKind::Discover,
                    _ => MessageKind::Request,
                },
                _ => MessageKind::NoReply,
            },
            content_type: response.header(reqwest::header::CONTENT_TYPE),
            content_length: response.header(reqwest::header::CONTENT_LENGTH),
            www_authenticate: response.challenge()?,
        });
        match reply {
            PostReply::Accepted => Ok(StreamableHttpPostResponse::Accepted),
            PostReply::SessionExpired => Err(Error::SessionExpired),
            PostReply::Challenge(challenge) => Err(challenge_error(challenge)),
            PostReply::Rejected(rejected) => {
                let status = response.status;
                let body = response.body().await?;
                let parsed = serde_json::from_slice::<ServerJsonRpcMessage>(&body);
                match (rejected, parsed, &message) {
                    (Rejected::DiscoveryProbe, parsed, ClientJsonRpcMessage::Request(request)) => {
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
                        Ok(StreamableHttpPostResponse::Json(
                            ServerJsonRpcMessage::error(error, Some(request.id.clone())),
                            None,
                        ))
                    }
                    (Rejected::JsonError, Ok(message @ ServerJsonRpcMessage::Error(_)), _) => {
                        Ok(StreamableHttpPostResponse::Json(message, session_id))
                    }
                    _ => Err(Error::UnexpectedServerResponse(
                        format!("HTTP {status}: {}", String::from_utf8_lossy(&body)).into(),
                    )),
                }
            }
            PostReply::Events => {
                Ok(StreamableHttpPostResponse::Sse(self.events(response, limit, false), session_id))
            }
            PostReply::Json { unparsed_accepted } => {
                let body = response.body().await?;
                match serde_json::from_slice(&body) {
                    Ok(message) => Ok(StreamableHttpPostResponse::Json(message, session_id)),
                    Err(_) if unparsed_accepted => Ok(StreamableHttpPostResponse::Accepted),
                    Err(error) => Err(Error::UnexpectedServerResponse(
                        format!(
                            "could not parse MCP response: {error}: {}",
                            String::from_utf8_lossy(&body[..body.len().min(256)])
                        )
                        .into(),
                    )),
                }
            }
            PostReply::UnexpectedContentType => Err(Error::UnexpectedContentType(
                response.header(reqwest::header::CONTENT_TYPE).map(str::to_owned),
            )),
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
            response.check(delete_status(response.status.as_u16(), response.challenge()?))?;
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
        let content_type = response.header(reqwest::header::CONTENT_TYPE);
        match stream_reply(response.status.as_u16(), content_type, response.challenge()?) {
            StreamReply::Events => Ok(self.events(response, limit, true)),
            StreamReply::Unsupported => Err(Error::ServerDoesNotSupportSse),
            StreamReply::Failed(failure) => Err(response.failure(failure)),
            StreamReply::UnexpectedContentType => {
                Err(Error::UnexpectedContentType(content_type.map(str::to_owned)))
            }
        }
    }
}

#[cfg(test)]
mod tests;
