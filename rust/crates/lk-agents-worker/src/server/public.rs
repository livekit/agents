//! The public listener: the agent's endpoints and HTTP routes, served by the worker itself where no
//! LiveKit Cloud node routes them (a self-hosted server, the console). Each connection becomes the
//! metadata and byte stream a node delivers, so the host's application server runs unchanged.

use crate::ipc::telemetry;
use std::sync::Arc;
use std::time::Duration;

use futures_util::{StreamExt, stream::FuturesUnordered};
use livekit_protocol::agent_http::{AgentEndpoint, AgentEndpointKind};
use lk_agents_abi::worker::http::{EndpointHandler, EndpointKind, EndpointRequest, HttpScheme};
use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::task::TokioSpawner;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;

use crate::http::HttpStream;

/// Matches the host application server's pending handler limit.
const MAX_CONNECTIONS: usize = 256;
const MAX_HEAD_BYTES: usize = 64 * 1024;
const MAX_HEADERS: usize = 128;
const HEAD_TIMEOUT: Duration = Duration::from_secs(30);

/// What the listener serves, and how it vouches for its callers.
pub struct PublicRoutes {
    /// `/agents/<agent>/<deployment>`, stripped before the host sees the path, as a node does.
    pub prefix: String,
    pub routes: Vec<AgentEndpoint>,
    /// The console's: nothing authenticates a caller, who holds every permission. Otherwise the
    /// endpoint verifies the caller's bearer token itself.
    pub unauthenticated: bool,
    pub proxy_headers: bool,
}

/// Serve `listener` until `stop`; dropping or stopping it cancels and joins its connections.
pub async fn serve_public(
    listener: TcpListener,
    routes: Arc<PublicRoutes>,
    handler: Arc<dyn EndpointHandler>,
    mut stop: watch::Receiver<bool>,
) -> Result<(), String> {
    if *stop.borrow_and_update() {
        return Ok(());
    }
    let spawner = TokioSpawner::current();
    let mut connections = FuturesUnordered::<OwnedTask>::new();
    let mut result = Ok(());
    loop {
        tokio::select! {
            biased;
            changed = stop.changed() => {
                if changed.is_err() || *stop.borrow_and_update() { break; }
            }
            _ = connections.next(), if !connections.is_empty() => {}
            accepted = listener.accept(), if connections.len() < MAX_CONNECTIONS => match accepted {
                Ok((socket, peer)) => connections.push(spawner.spawn(serve_connection(
                    socket,
                    peer.ip().to_string(),
                    routes.clone(),
                    handler.clone(),
                ))),
                Err(error) => { result = Err(error.to_string()); break; }
            },
        }
    }
    connections.iter().for_each(OwnedTask::cancel);
    while connections.next().await.is_some() {}
    result
}

async fn serve_connection(
    mut socket: TcpStream,
    peer: String,
    routes: Arc<PublicRoutes>,
    handler: Arc<dyn EndpointHandler>,
) {
    let exchange = match tokio::time::timeout(HEAD_TIMEOUT, read_head(&mut socket)).await {
        Ok(Ok(buf)) => accept(&buf, &peer, &routes),
        Ok(Err(status)) => Err(status),
        Err(_) => Err("408 Request Timeout"),
    };
    let (head, request) = match exchange {
        Ok(exchange) => exchange,
        Err(status) => {
            let reply =
                format!("HTTP/1.1 {status}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n");
            let _ = socket.write_all(reply.as_bytes()).await;
            return;
        }
    };
    let (writer, reader) = crate::transport::tcp_exchange(socket, head);
    let stream = Arc::new(HttpStream::new(reader, writer));
    if let Err(error) = handler.handle(&telemetry::Cx::NONE, stream.clone(), request).await {
        log::error!(target: crate::LOG, "endpoint host failed: {error}");
        stream.fail(livekit_protocol::agent_http::HttpStreamResetCode::HsrInternal);
    }
}

/// The bytes up to and past the end of the request head.
async fn read_head(socket: &mut TcpStream) -> Result<Vec<u8>, &'static str> {
    let mut buf = Vec::with_capacity(4096);
    loop {
        if buf.windows(4).any(|window| window == b"\r\n\r\n") {
            return Ok(buf);
        }
        if buf.len() >= MAX_HEAD_BYTES {
            return Err("431 Request Header Fields Too Large");
        }
        let mut chunk = [0u8; 4096];
        match socket.read(&mut chunk).await {
            Ok(0) | Err(_) => return Err("400 Bad Request"),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
}

/// The head the host reads (the path without the agent's prefix, one exchange per connection)
/// and the request's metadata; or the status the listener answers itself.
fn accept(
    buf: &[u8],
    peer: &str,
    routes: &PublicRoutes,
) -> Result<(Vec<u8>, EndpointRequest), &'static str> {
    let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut parsed = httparse::Request::new(&mut headers);
    let length = match parsed.parse(buf) {
        Ok(httparse::Status::Complete(length)) => length,
        Ok(httparse::Status::Partial) | Err(_) => return Err("400 Bad Request"),
    };
    let (Some(method), Some(target), Some(version)) = (parsed.method, parsed.path, parsed.version)
    else {
        return Err("400 Bad Request");
    };
    let (path, query) = target.split_at(target.find('?').unwrap_or(target.len()));
    let path = match path.strip_prefix(routes.prefix.as_str()) {
        Some("") => "/",
        Some(rest) if rest.starts_with('/') => rest,
        _ => return Err("404 Not Found"),
    };
    let matched: Vec<_> = routes.routes.iter().filter(|route| matches(&route.path, path)).collect();
    let Some(route) = matched.iter().find(|route| route.methods.iter().any(|m| m == method)) else {
        return Err(if matched.is_empty() { "404 Not Found" } else { "405 Method Not Allowed" });
    };
    let header = |name: &str| {
        parsed
            .headers
            .iter()
            .find(|header| header.name.eq_ignore_ascii_case(name))
            .and_then(|header| std::str::from_utf8(header.value).ok())
    };
    let upgrade = header("upgrade").is_some();
    let mut client_addr = peer.to_owned();
    let mut scheme = HttpScheme::Http;
    if routes.proxy_headers {
        if let Some(forwarded) = header("x-forwarded-for").and_then(|v| v.split(',').next()) {
            client_addr = forwarded.trim().to_owned();
        }
        if header("x-forwarded-proto").is_some_and(|proto| proto.trim() == "https") {
            scheme = HttpScheme::Https;
        }
    }
    let mut head = format!("{method} {path}{query} HTTP/1.{version}\r\n").into_bytes();
    for header in parsed.headers.iter() {
        // One exchange per connection, as on a node stream; an upgrade keeps its own.
        if !upgrade && header.name.eq_ignore_ascii_case("connection") {
            continue;
        }
        head.extend_from_slice(header.name.as_bytes());
        head.extend_from_slice(b": ");
        head.extend_from_slice(header.value);
        head.extend_from_slice(b"\r\n");
    }
    if !upgrade {
        head.extend_from_slice(b"connection: close\r\n");
    }
    head.extend_from_slice(b"\r\n");
    head.extend_from_slice(&buf[length..]);
    let kind = match AgentEndpointKind::try_from(route.kind) {
        Ok(AgentEndpointKind::AekText) => EndpointKind::Text,
        _ => EndpointKind::Http,
    };
    let request = EndpointRequest {
        request_id: lk_agents_net::strings::shortuuid("AER_"),
        authorized: routes.unauthenticated,
        unauthenticated: routes.unauthenticated,
        route: route.path.clone(),
        kind,
        client_addr,
        scheme,
        timeout_ms: 0,
    };
    Ok((head, request))
}

/// A manifest path against a request path: `{name}` is one segment, `{name:path}` the rest.
fn matches(pattern: &str, path: &str) -> bool {
    let mut pattern = pattern.trim_start_matches('/').split('/');
    let mut path = path.trim_start_matches('/').split('/');
    loop {
        match (pattern.next(), path.next()) {
            (None, None) => return true,
            (Some(p), _) if p.starts_with('{') && p.ends_with(":path}") => return true,
            (Some(p), Some(s)) if p.starts_with('{') && p.ends_with('}') && !s.is_empty() => {}
            (Some(p), Some(s)) if p == s => {}
            _ => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routes(proxy_headers: bool) -> PublicRoutes {
        PublicRoutes {
            prefix: "/agents/dentist/default".into(),
            routes: vec![
                AgentEndpoint {
                    path: "/backend/{path:path}".into(),
                    methods: vec!["GET".into(), "POST".into()],
                    kind: AgentEndpointKind::AekHttp as i32,
                    public: false,
                },
                AgentEndpoint {
                    path: "/items/{id}".into(),
                    methods: vec!["GET".into()],
                    kind: AgentEndpointKind::AekHttp as i32,
                    public: true,
                },
            ],
            unauthenticated: false,
            proxy_headers,
        }
    }

    #[test]
    fn the_prefix_is_stripped_and_the_exchange_closes_the_connection() {
        let (head, request) = accept(
            b"POST /agents/dentist/default/backend/?a=1 HTTP/1.1\r\nHost: x\r\nConnection: keep-alive\r\nContent-Length: 2\r\n\r\n{}",
            "10.0.0.1",
            &routes(false),
        )
        .unwrap();
        let head = String::from_utf8(head).unwrap();
        assert!(head.starts_with("POST /backend/?a=1 HTTP/1.1\r\n"), "{head}");
        assert!(!head.contains("keep-alive"));
        assert!(head.ends_with("connection: close\r\n\r\n{}"), "{head}");
        assert_eq!(request.route, "/backend/{path:path}");
        assert_eq!(request.client_addr, "10.0.0.1");
        assert!(!request.authorized);
    }

    #[test]
    fn unknown_paths_and_methods_are_answered_by_the_listener() {
        let routes = routes(false);
        let status = |raw: &str| accept(raw.as_bytes(), "::1", &routes).err();
        assert_eq!(status("GET /backend/ HTTP/1.1\r\nHost: x\r\n\r\n"), Some("404 Not Found"));
        assert_eq!(
            status("GET /agents/other/default/backend/ HTTP/1.1\r\nHost: x\r\n\r\n"),
            Some("404 Not Found")
        );
        assert_eq!(
            status("DELETE /agents/dentist/default/items/1 HTTP/1.1\r\nHost: x\r\n\r\n"),
            Some("405 Method Not Allowed")
        );
        assert_eq!(status("GET /agents/dentist/default/items/1 HTTP/1.1\r\nHost: x\r\n\r\n"), None);
    }

    #[test]
    fn forwarded_headers_count_only_when_trusted() {
        let raw = b"GET /agents/dentist/default/items/1 HTTP/1.1\r\nHost: x\r\nX-Forwarded-For: 203.0.113.9, 10.0.0.2\r\nX-Forwarded-Proto: https\r\n\r\n";
        let (_, trusted) = accept(raw, "10.0.0.2", &routes(true)).unwrap();
        assert_eq!(trusted.client_addr, "203.0.113.9");
        assert_eq!(trusted.scheme, HttpScheme::Https);
        let (_, untrusted) = accept(raw, "10.0.0.2", &routes(false)).unwrap();
        assert_eq!(untrusted.client_addr, "10.0.0.2");
        assert_eq!(untrusted.scheme, HttpScheme::Http);
    }

    #[test]
    fn manifest_paths_match_segments_and_the_rest() {
        assert!(matches("/backend/{path:path}", "/backend"));
        assert!(matches("/backend/{path:path}", "/backend/.well-known/agent-card.json"));
        assert!(matches("/items/{id}", "/items/7"));
        assert!(!matches("/items/{id}", "/items/"));
        assert!(!matches("/items/{id}", "/items/7/more"));
        assert!(!matches("/backend/{path:path}", "/backoffice"));
    }
}
