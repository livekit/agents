//! The public listener: the agent's endpoints and HTTP routes, served by the worker itself where no
//! LiveKit Cloud node routes them (a self-hosted server, the console). Each connection is served
//! as a node's stream is; the listener routes the parsed head itself, as a node does.

use std::sync::Arc;

use futures_util::{StreamExt, stream::FuturesUnordered};
use livekit_protocol::agent_http::{AgentEndpoint, AgentEndpointKind};
use lk_agents_abi::worker::http::{EndpointKind, EndpointRequest, HttpScheme, RequestHead};
use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::task::TokioSpawner;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;

use crate::http::{EndpointServer, HttpStream, Origin, decode_path};

/// Matches the endpoint server's pending handler limit.
const MAX_CONNECTIONS: usize = 256;

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
    server: Arc<EndpointServer>,
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
                    server.clone(),
                ))),
                Err(error) => { result = Err(error.to_string()); break; }
            },
        }
    }
    // End what still runs in a way each client can read, then cut what remains.
    server.stop().await;
    connections.iter().for_each(OwnedTask::cancel);
    while connections.next().await.is_some() {}
    result
}

async fn serve_connection(
    socket: TcpStream,
    peer: String,
    routes: Arc<PublicRoutes>,
    server: Arc<EndpointServer>,
) {
    let (writer, reader) = crate::transport::tcp_exchange(socket, Vec::new());
    let wire = HttpStream::new(reader, writer).into_wire();
    server.serve(wire, Origin::Listener { routes, peer }).await;
}

impl PublicRoutes {
    /// Route a parsed head: its path loses the agent's prefix, as a node strips it, and the
    /// request gets its metadata; or the status the listener answers itself.
    pub(crate) fn resolve(
        &self,
        head: &mut RequestHead,
        peer: &str,
    ) -> Result<EndpointRequest, u16> {
        let raw = std::str::from_utf8(&head.raw_path).map_err(|_| 400u16)?;
        let path = match raw.strip_prefix(self.prefix.as_str()) {
            Some("") => "/".to_owned(),
            Some(rest) if rest.starts_with('/') => rest.to_owned(),
            _ => return Err(404),
        };
        let matched: Vec<_> =
            self.routes.iter().filter(|route| matches(&route.path, &path)).collect();
        let Some(route) = matched.iter().find(|route| route.methods.contains(&head.method)) else {
            return Err(if matched.is_empty() { 404 } else { 405 });
        };
        let header = |name: &[u8]| {
            head.headers
                .iter()
                .find(|(key, _)| key == name)
                .and_then(|(_, value)| std::str::from_utf8(value).ok())
        };
        let mut client_addr = peer.to_owned();
        let mut scheme = HttpScheme::Http;
        if self.proxy_headers {
            if let Some(forwarded) = header(b"x-forwarded-for").and_then(|v| v.split(',').next()) {
                client_addr = forwarded.trim().to_owned();
            }
            if header(b"x-forwarded-proto").is_some_and(|proto| proto.trim() == "https") {
                scheme = HttpScheme::Https;
            }
        }
        let kind = match AgentEndpointKind::try_from(route.kind) {
            Ok(AgentEndpointKind::AekText) => EndpointKind::Text,
            _ => EndpointKind::Http,
        };
        let request = EndpointRequest {
            request_id: lk_agents_net::strings::shortuuid("AER_"),
            authorized: self.unauthenticated,
            unauthenticated: self.unauthenticated,
            route: route.path.clone(),
            kind,
            client_addr,
            scheme,
            timeout_ms: 0,
        };
        head.path = decode_path(&path);
        head.raw_path = path.into_bytes();
        Ok(request)
    }
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

    fn head(method: &str, path: &str, headers: &[(&str, &str)]) -> RequestHead {
        RequestHead {
            method: method.into(),
            path: decode_path(path),
            raw_path: path.as_bytes().to_vec(),
            headers: headers
                .iter()
                .map(|(k, v)| (k.as_bytes().to_vec(), v.as_bytes().to_vec()))
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn the_prefix_is_stripped_and_the_route_matched() {
        let mut parsed = head("POST", "/agents/dentist/default/backend/a%20b", &[]);
        let request = routes(false).resolve(&mut parsed, "10.0.0.1").unwrap();
        assert_eq!(parsed.raw_path, b"/backend/a%20b");
        assert_eq!(parsed.path, "/backend/a b");
        assert_eq!(request.route, "/backend/{path:path}");
        assert_eq!(request.client_addr, "10.0.0.1");
        assert!(!request.authorized);
    }

    #[test]
    fn unknown_paths_and_methods_are_answered_by_the_listener() {
        let routes = routes(false);
        let status =
            |method: &str, path: &str| routes.resolve(&mut head(method, path, &[]), "::1").err();
        assert_eq!(status("GET", "/backend/"), Some(404));
        assert_eq!(status("GET", "/agents/other/default/backend/"), Some(404));
        assert_eq!(status("GET", "/agents/dentist/defaultx/items/1"), Some(404));
        assert_eq!(status("DELETE", "/agents/dentist/default/items/1"), Some(405));
        assert_eq!(status("GET", "/agents/dentist/default/items/1"), None);
    }

    #[test]
    fn forwarded_headers_count_only_when_trusted() {
        let forwarded =
            [("x-forwarded-for", "203.0.113.9, 10.0.0.2"), ("x-forwarded-proto", "https")];
        let path = "/agents/dentist/default/items/1";
        let trusted = routes(true).resolve(&mut head("GET", path, &forwarded), "10.0.0.2").unwrap();
        assert_eq!(trusted.client_addr, "203.0.113.9");
        assert_eq!(trusted.scheme, HttpScheme::Https);
        let untrusted =
            routes(false).resolve(&mut head("GET", path, &forwarded), "10.0.0.2").unwrap();
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
