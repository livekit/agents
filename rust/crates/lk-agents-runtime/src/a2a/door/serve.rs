//! An A2A endpoint over one HTTP exchange: its card and its JSON-RPC methods under
//! `/<endpoint>`, the caller authenticated first, each answer as JSON or as an SSE stream.

use lk_agents_abi::a2a::auth::{AuthError, AuthPrincipal, Credentials as _, PrincipalKind};
use lk_agents_abi::a2a::card::EndpointDescriptor;
use lk_agents_abi::a2a::door::{DoorError, TextServing};
use lk_agents_abi::a2a::metadata::{AGENT_CARD_PATH, EXTENSIONS_HEADER, GRANT_HEADER, RPC_PATH};
use lk_agents_abi::errors::HostError;
use lk_agents_abi::worker::http::{
    ExchangeEvent, ExchangeMessage, HttpExchange, HttpScheme, RequestHead,
};
use lk_agents_net::a2a_auth::Credentials;
use serde_json::json;

use super::{Door, MAX_PAYLOAD_BYTES};

type Headers = Vec<(Vec<u8>, Vec<u8>)>;

enum Route {
    Card,
    Rpc,
}

/// Answer one exchange. A refused caller gets `401` or `403`, an unknown path `404` and another
/// method `405`, each with a JSON `{"error": message}` body.
pub async fn serve<E: HttpExchange>(
    door: &Door,
    exchange: &E,
    serving: TextServing,
) -> Result<(), HostError> {
    let head = exchange.head();
    let path =
        head.path.strip_prefix('/').and_then(|path| path.strip_prefix(door.endpoint.name.as_str()));
    let route = match (path, head.method.as_str()) {
        (Some(AGENT_CARD_PATH), "GET" | "HEAD") => Route::Card,
        (Some(RPC_PATH), "POST") => Route::Rpc,
        (Some(AGENT_CARD_PATH | RPC_PATH), _) => {
            return answer(exchange, 405, error("Method Not Allowed"), Vec::new()).await;
        }
        _ => return answer(exchange, 404, error("Not Found"), Vec::new()).await,
    };
    let principal = match caller(&head.headers, &serving) {
        Ok(principal) => principal,
        Err(refused) => return refuse(exchange, &refused).await,
    };
    match route {
        Route::Card => card(door, exchange, &head, &serving).await,
        Route::Rpc => rpc(door, exchange, &head, principal, serving).await,
    }
}

/// The caller, in order: the console's listener vouches for everyone; a worker holding the
/// project's credentials verifies the bearer, the node's verdict standing for a project
/// credential it stripped; without credentials only the node's verdict admits, as the project.
fn caller(headers: &Headers, serving: &TextServing) -> Result<AuthPrincipal, AuthError> {
    let request = &serving.request;
    if request.unauthenticated {
        return Ok(lk_agents_core::a2a::auth::unauthenticated());
    }
    match &serving.credentials {
        Some((key, secret)) => Credentials::new(key.clone(), secret.clone()).principal(
            header(headers, "authorization"),
            header(headers, GRANT_HEADER),
            request.authorized,
            serving.agent_name.clone(),
            serving.deployment.clone(),
        ),
        None if request.authorized => Ok(AuthPrincipal {
            kind: PrincipalKind::Project,
            grant: None,
            identity: String::new(),
            claims: Default::default(),
            endpoint_call: true,
        }),
        None => Err(AuthError::Unauthorized("a LiveKit bearer token is required".into())),
    }
}

/// The card, its URL where the request reached the endpoint: Cloud's, else the worker's path on
/// the request's own address.
async fn card<E: HttpExchange>(
    door: &Door,
    exchange: &E,
    head: &RequestHead,
    serving: &TextServing,
) -> Result<(), HostError> {
    let root = match serving.endpoint_url.as_deref() {
        Some(url) if !url.starts_with('/') => url.to_owned(),
        path => {
            let scheme = match serving.request.scheme {
                HttpScheme::Http => "http",
                HttpScheme::Https => "https",
            };
            let host = header(&head.headers, "host").unwrap_or_default();
            let fallback = format!("/{}", door.endpoint.name.as_str());
            format!("{scheme}://{host}{}", path.unwrap_or(&fallback))
        }
    };
    let descriptor = EndpointDescriptor {
        url: format!("{}{RPC_PATH}", root.trim_end_matches('/')),
        ..door.endpoint.clone()
    };
    let card = lk_agents_core::a2a::card::agent_card(descriptor)?;
    answer(exchange, 200, card.to_string(), Vec::new()).await
}

/// One JSON-RPC request: its body read up to the payload budget, then the door's first frame as
/// a JSON body, or every frame as an SSE stream for a streaming method.
async fn rpc<E: HttpExchange>(
    door: &Door,
    exchange: &E,
    head: &RequestHead,
    principal: AuthPrincipal,
    serving: TextServing,
) -> Result<(), HostError> {
    let mut body = Vec::new();
    let mut received = 0usize;
    loop {
        match exchange.receive().await.map_err(host)? {
            ExchangeEvent::Body { data, more } => {
                received = received.saturating_add(data.len());
                if received > MAX_PAYLOAD_BYTES {
                    break;
                }
                body.extend_from_slice(&data);
                if !more {
                    break;
                }
            }
            // The client left before its body ended: nothing to answer.
            _ => return Ok(()),
        }
    }
    let mut headers = Headers::new();
    let requested = header(&head.headers, EXTENSIONS_HEADER);
    if let Some(negotiated) = lk_agents_core::a2a::rpc::extensions_header(requested.as_deref()) {
        headers.push((EXTENSIONS_HEADER.to_ascii_lowercase().into_bytes(), negotiated.into()));
    }
    let responses = door
        .dispatch(
            body,
            principal,
            header(&head.headers, "a2a-version"),
            header(&head.headers, "last-event-id"),
            Some(received),
            serving.host_context,
            header(&head.headers, "traceparent"),
        )
        .await;
    let first = match responses.next().await {
        Ok(Some(first)) => first,
        Ok(None) => return Err(HostError::new("the A2A door produced no response")),
        Err(DoorError::Auth(refused)) => return refuse(exchange, &refused).await,
        Err(DoorError::Storage(error)) => return Err(HostError::new(error.to_string())),
    };
    if !responses.streaming() || first.is_error {
        return answer(exchange, 200, first.json, headers).await;
    }
    headers.push((b"content-type".to_vec(), b"text/event-stream".to_vec()));
    headers.push((b"cache-control".to_vec(), b"no-cache".to_vec()));
    send(exchange, ExchangeMessage::Start { status: 200, headers }).await?;
    let mut frame = first;
    loop {
        let data = frame.sse.into_bytes();
        send(exchange, ExchangeMessage::Body { data, more: true }).await?;
        frame = match responses.next().await {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            Err(error) => return Err(HostError::new(error.to_string())),
        };
    }
    responses.aclose().await;
    send(exchange, ExchangeMessage::Body { data: Vec::new(), more: false }).await
}

fn error(message: &str) -> String {
    json!({ "error": message }).to_string()
}

async fn refuse<E: HttpExchange>(exchange: &E, refused: &AuthError) -> Result<(), HostError> {
    let (status, message) = match refused {
        AuthError::Unauthorized(message) => (401, message),
        AuthError::Forbidden(message) => (403, message),
    };
    answer(exchange, status, error(message), Vec::new()).await
}

/// A whole JSON response.
async fn answer<E: HttpExchange>(
    exchange: &E,
    status: u16,
    json: String,
    mut headers: Headers,
) -> Result<(), HostError> {
    headers.push((b"content-type".to_vec(), b"application/json".to_vec()));
    headers.push((b"content-length".to_vec(), json.len().to_string().into_bytes()));
    send(exchange, ExchangeMessage::Start { status, headers }).await?;
    send(exchange, ExchangeMessage::Body { data: json.into_bytes(), more: false }).await
}

async fn send<E: HttpExchange>(exchange: &E, message: ExchangeMessage) -> Result<(), HostError> {
    exchange.send(message).await.map_err(host)
}

fn host(error: lk_agents_abi::worker::http::ExchangeError) -> HostError {
    HostError::new(error.to_string())
}

/// A request header by name, case-insensitively.
fn header(headers: &Headers, name: &str) -> Option<String> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name.as_bytes()))
        .and_then(|(_, value)| String::from_utf8(value.clone()).ok())
}
