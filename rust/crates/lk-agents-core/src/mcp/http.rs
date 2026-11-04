//! MCP's HTTP protocol decisions over response status, headers and the message kind: the runtime
//! executes the exchange and reads bodies; these rules choose what each response means.

/// The header naming the streamable HTTP session.
pub const SESSION_HEADER: &str = "Mcp-Session-Id";

/// Headers the transport sets itself; a custom header may not replace them.
pub fn reserved_header(name: &str) -> bool {
    ["accept", "mcp-session-id", "last-event-id"]
        .iter()
        .any(|reserved| name.eq_ignore_ascii_case(reserved))
}

/// An authorization challenge the SDK surfaces to its caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Challenge {
    /// 401 with `WWW-Authenticate`.
    AuthRequired(String),
    /// 403 with `WWW-Authenticate`, and the scope it asks for.
    InsufficientScope { challenge: String, scope: Option<String> },
}

/// A failed status, as a challenge or a plain HTTP failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    Challenge(Challenge),
    Status,
}

/// A response's status: success, or why it failed.
pub fn check_status(status: u16, www_authenticate: Option<&str>) -> Result<(), Failure> {
    if let (401 | 403, Some(challenge)) = (status, www_authenticate) {
        let challenge = challenge.to_owned();
        return Err(Failure::Challenge(if status == 401 {
            Challenge::AuthRequired(challenge)
        } else {
            let scope = challenge_scope(&challenge);
            Challenge::InsufficientScope { challenge, scope }
        }));
    }
    if (200..300).contains(&status) { Ok(()) } else { Err(Failure::Status) }
}

/// The `scope=` parameter of a challenge, quoted or bare.
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

/// The JSON-RPC message a POST carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    /// `server/discover`: the SDK's probe for the protocol generation.
    Discover,
    Request,
    /// A notification, response or error: no reply is expected.
    NoReply,
}

/// The response headers of one POST.
#[derive(Debug, Clone, Copy)]
pub struct PostResponse<'a> {
    pub status: u16,
    pub has_session: bool,
    pub message: MessageKind,
    pub content_type: Option<&'a str>,
    pub content_length: Option<&'a str>,
    pub www_authenticate: Option<&'a str>,
}

/// What a POST response means, decided from its headers before any body is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostReply {
    Accepted,
    SessionExpired,
    Challenge(Challenge),
    /// A failed status. Read the body; [`Rejected`] says which error reply it may carry.
    Rejected(Rejected),
    /// The reply arrives as an SSE stream.
    Events,
    /// The body is one JSON-RPC message; an unparsable body is accepted when no reply is expected.
    Json {
        unparsed_accepted: bool,
    },
    UnexpectedContentType,
}

/// The error reply a rejected POST's body may carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejected {
    /// An old server rejected discovery before a session exists; the SDK then probes the legacy
    /// initialization protocol. The rejection answers the probe: the body's JSON-RPC error, else
    /// an invalid-request error naming the status and body.
    DiscoveryProbe,
    /// A JSON body holding a JSON-RPC error is the reply.
    JsonError,
    /// The body is only failure detail.
    Failure,
}

pub fn post_reply(response: PostResponse<'_>) -> PostReply {
    let PostResponse { status, has_session, message, content_type, .. } = response;
    if matches!(status, 202 | 204) {
        return PostReply::Accepted;
    }
    if status == 404 && has_session {
        return PostReply::SessionExpired;
    }
    let no_reply = message == MessageKind::NoReply;
    let success = (200..300).contains(&status);
    if success && no_reply && response.content_length == Some("0") {
        return PostReply::Accepted;
    }
    let json = content_type.is_some_and(|value| value.starts_with("application/json"));
    match check_status(status, response.www_authenticate) {
        Err(Failure::Challenge(challenge)) => PostReply::Challenge(challenge),
        Err(Failure::Status) => PostReply::Rejected(
            if !has_session
                && (400..500).contains(&status)
                && !matches!(status, 401 | 403)
                && message == MessageKind::Discover
            {
                Rejected::DiscoveryProbe
            } else if json {
                Rejected::JsonError
            } else {
                Rejected::Failure
            },
        ),
        Ok(()) if content_type.is_some_and(|value| value.starts_with("text/event-stream")) => {
            PostReply::Events
        }
        Ok(()) if json => PostReply::Json { unparsed_accepted: no_reply },
        Ok(()) => PostReply::UnexpectedContentType,
    }
}

/// What a GET opening the server's stream means.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamReply {
    Events,
    /// 405: the server offers no standalone SSE stream.
    Unsupported,
    Failed(Failure),
    UnexpectedContentType,
}

pub fn stream_reply(
    status: u16,
    content_type: Option<&str>,
    www_authenticate: Option<&str>,
) -> StreamReply {
    if status == 405 {
        return StreamReply::Unsupported;
    }
    if let Err(failure) = check_status(status, www_authenticate) {
        return StreamReply::Failed(failure);
    }
    match content_type {
        Some(value)
            if value.starts_with("text/event-stream") || value.starts_with("application/json") =>
        {
            StreamReply::Events
        }
        _ => StreamReply::UnexpectedContentType,
    }
}

/// A session DELETE: 405 means the server keeps no deletable session, which is not a failure.
pub fn delete_status(status: u16, www_authenticate: Option<&str>) -> Result<(), Failure> {
    if status == 405 { Ok(()) } else { check_status(status, www_authenticate) }
}
