use crate::errors::ProviderError;

/// Flat; [`ApiErrorKind::is_a`] knows the subclass edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApiErrorKind {
    Api,
    Status,
    Connection,
    Timeout,
    /// Status 499: we closed the request. Never retryable, never a mark against the provider
    /// (`llm.py:288`).
    Cancelled,
}

impl ApiErrorKind {
    /// The one place the 499 rule is written.
    pub fn for_status(status_code: i64) -> ApiErrorKind {
        if status_code == CANCELLED_STATUS { ApiErrorKind::Cancelled } else { ApiErrorKind::Status }
    }

    pub fn is_a(self, other: ApiErrorKind) -> bool {
        self == other
            || other == ApiErrorKind::Api
            || (other == ApiErrorKind::Connection && self == ApiErrorKind::Timeout)
            || (other == ApiErrorKind::Status && self == ApiErrorKind::Cancelled)
    }
}

pub const DEFAULT_CONNECTION_MESSAGE: &str = "Connection error.";

pub const DEFAULT_TIMEOUT_MESSAGE: &str = "Request timed out.";

/// The reference also lists 499; it is dropped here on purpose ([`ApiErrorKind::Cancelled`]).
pub const TRANSIENT_CLIENT_STATUSES: [i64; 2] = [408, 429];

/// The status of a request we closed ([`ApiErrorKind::Cancelled`]).
const CANCELLED_STATUS: i64 = 499;

/// The rule `APIStatusError.__init__` applies: is this status retryable? A non-transient 4xx is
/// forced non-retryable even when the caller passed `retryable=True` (#5831).
pub fn retryable_for_status(status_code: i64, retryable: Option<bool>) -> bool {
    let requested = retryable.unwrap_or(true);
    if (400..500).contains(&status_code) && !TRANSIENT_CLIENT_STATUSES.contains(&status_code) {
        return false;
    }
    requested
}

/// Native path only (IANA as of Python 3.14); Python uses its own `http.HTTPStatus`.
pub fn http_reason(status: i64) -> Option<&'static str> {
    PHRASES.binary_search_by_key(&status, |&(code, _)| code).ok().map(|i| PHRASES[i].1)
}

/// What `create_api_error_from_http` shows the user: message and reason when the message adds
/// detail, else just the reason (never `"Not Found (404 Not Found)"`).
pub fn http_error_message(message: &str, status: i64, reason: Option<&str>) -> String {
    let reason = match reason {
        Some(reason) => reason.to_owned(),
        None => format!("HTTP {status}"),
    };
    if !message.is_empty() && message != reason {
        format!("{message} ({status} {reason})")
    } else {
        format!("{reason} ({status})")
    }
}

/// The helpers the host's `APIError` classes apply.
#[crate::host(functions(pid = false))]
pub trait ApiErrorFunctions {
    /// The message of an `APIStatusError` built from an HTTP status: the message and the reason
    /// when the message adds detail, else just the reason.
    #[host(function, name = "http_error_display")]
    fn http_error_message(message: &str, status: i64, reason: Option<String>) -> String;

    /// `APIStatusError`'s `retryable`: a non-transient 4xx status, 499 among them, never is.
    #[host(function)]
    fn retryable_for_status(status_code: i64, retryable: Option<bool>) -> bool;
}

/// What a websocket close code says about the failure that closed the socket.
#[crate::host(record(module = "types", emit))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CloseKind {
    /// 1000: the peer ended the session on purpose.
    Normal,
    /// The peer refused what was sent or who sent it: a new connection fails the same way.
    Final,
    /// Going away, a server error, an abnormal close or no code: a new connection may succeed.
    Retryable,
}

impl CloseKind {
    /// The one place close codes are classified (RFC 6455 section 7.4, IANA's 3000 range, and a
    /// private 44xx code read as its HTTP status).
    pub fn of(code: Option<u16>) -> CloseKind {
        match code {
            Some(1000) => CloseKind::Normal,
            Some(1002 | 1003 | 1007 | 1008 | 1009 | 1010 | 1015 | 3000 | 3003) => CloseKind::Final,
            Some(code @ 4400..=4499) if !retryable_for_status(i64::from(code) - 4000, None) => {
                CloseKind::Final
            }
            _ => CloseKind::Retryable,
        }
    }
}

/// The host's websocket helpers.
#[crate::host(functions(pid = false))]
pub trait WebSocketFunctions {
    /// What a close code says about the failure: `http.WebSocket.close_error`'s retryability.
    #[host(function)]
    fn websocket_close_kind(code: Option<u16>) -> CloseKind;
}

/// The wait an `APITimeoutError` bounded, as `APIConnectOptions` names its limits. Absent when the
/// failing layer does not say (a foreign SDK timeout), never guessed. Its wire string is the
/// Python value of `APITimeoutError.timeout`.
#[crate::host(record(module = "types", emit))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Timeout {
    Pool,
    Connect,
    Read,
    Write,
    Total,
}

/// `body` stays on the Python object. `retryable` goes to the constructor: its `__init__` default
/// is `True`, so setting it afterwards is too late for a plugin reading it.
#[crate::host(error(
    by = kind,
    cause = cause,
    kinds(
        Api(class = "livekit.agents._bindings.APIError", kwargs(retryable)),
        Status(
            class = "livekit.agents._bindings.APIStatusError",
            kwargs(retryable, status_code, request_id)
        ),
        // The reference detects cancellation as `APIStatusError` with status 499.
        Cancelled(
            class = "livekit.agents._bindings.APIStatusError",
            kwargs(retryable, status_code, request_id)
        ),
        Connection(class = "livekit.agents._bindings.APIConnectionError", kwargs(retryable)),
        Timeout(class = "livekit.agents._bindings.APITimeoutError", kwargs(retryable, timeout)),
    )
))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ApiError {
    pub kind: ApiErrorKind,
    pub message: String,
    /// For [`ApiErrorKind::Status`] this went through [`retryable_for_status`].
    pub retryable: bool,
    pub status_code: Option<i64>,
    pub request_id: Option<String>,
    /// [`ApiErrorKind::Timeout`] only.
    pub timeout: Option<Timeout>,
    /// The failure this one was raised from: the host exception's `__cause__`.
    #[source]
    pub cause: Option<Box<ProviderError>>,
    /// What the provider itself said: its message and the wait it asked for.
    pub said: Option<Box<ProviderSaid>>,
}

/// What a provider said about its failure beyond the status, boxed so an error stays small.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderSaid {
    /// Its own message (`APIError.detail`): content, logged once when the failure is retried or
    /// given up, never part of the error's text.
    pub detail: Option<String>,
    /// How long it asked to wait before a retry (`Retry-After`).
    pub retry_after: Option<std::time::Duration>,
}

impl ProviderSaid {
    /// `None` when it said nothing.
    pub fn boxed(
        detail: Option<String>,
        retry_after: Option<std::time::Duration>,
    ) -> Option<Box<Self>> {
        (detail.is_some() || retry_after.is_some())
            .then(|| Box::new(ProviderSaid { detail, retry_after }))
    }
}

impl ApiError {
    /// `__init__` defaults to `retryable=True`.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            kind: ApiErrorKind::Api,
            message: message.into(),
            retryable: true,
            status_code: None,
            request_id: None,
            timeout: None,
            cause: None,
            said: None,
        }
    }

    /// `retryable = None` means the caller did not say; 499 comes out as `Cancelled`.
    pub fn status(
        message: impl Into<String>,
        status_code: i64,
        request_id: Option<String>,
        retryable: Option<bool>,
    ) -> Self {
        Self {
            kind: ApiErrorKind::for_status(status_code),
            message: message.into(),
            retryable: retryable_for_status(status_code, retryable),
            status_code: Some(status_code),
            request_id,
            timeout: None,
            cause: None,
            said: None,
        }
    }

    pub fn connection(message: Option<String>) -> Self {
        Self {
            kind: ApiErrorKind::Connection,
            message: message.unwrap_or_else(|| DEFAULT_CONNECTION_MESSAGE.to_owned()),
            retryable: true,
            status_code: None,
            request_id: None,
            timeout: None,
            cause: None,
            said: None,
        }
    }

    pub fn timeout(message: Option<String>) -> Self {
        Self {
            kind: ApiErrorKind::Timeout,
            message: message.unwrap_or_else(|| DEFAULT_TIMEOUT_MESSAGE.to_owned()),
            retryable: true,
            status_code: None,
            request_id: None,
            timeout: None,
            cause: None,
            said: None,
        }
    }

    pub fn from_http(message: &str, status: i64, request_id: Option<String>) -> Self {
        Self::status(
            http_error_message(message, status, http_reason(status)),
            status,
            request_id,
            None,
        )
    }

    /// An `APITimeoutError` for the wait `timeout` bounded.
    pub fn timed_out(timeout: Timeout) -> Self {
        Self {
            timeout: Some(timeout),
            ..Self::timeout(Some(format!("Request timed out ({} timeout).", timeout.as_str())))
        }
    }

    /// A request we closed: never retried, never a mark against the provider. The host sees
    /// `APIStatusError` with status 499.
    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::status(message, CANCELLED_STATUS, None, Some(false))
    }

    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    /// The wait the provider asked for before a retry.
    pub fn retry_after(&self) -> Option<std::time::Duration> {
        self.said.as_ref().and_then(|said| said.retry_after)
    }

    /// The one log of the provider's own message, under an `lk.pii.*` key: by the retry
    /// announcement or the give-up of the failure it came with.
    pub fn log_detail(&self) {
        if let Some(detail) = self.said.as_ref().and_then(|said| said.detail.as_ref()) {
            log::warn!(
                target: crate::LOG_TARGET,
                (crate::telemetry::spans::common::log_keys::PROVIDER_DETAIL) = detail.as_str();
                "{}: the provider's message is attached as {}",
                self.message,
                crate::telemetry::spans::common::log_keys::PROVIDER_DETAIL
            );
        }
    }
}

/// Sorted by code: [`http_reason`] binary-searches it.
const PHRASES: &[(i64, &str)] = &[
    (100, "Continue"),
    (101, "Switching Protocols"),
    (102, "Processing"),
    (103, "Early Hints"),
    (200, "OK"),
    (201, "Created"),
    (202, "Accepted"),
    (203, "Non-Authoritative Information"),
    (204, "No Content"),
    (205, "Reset Content"),
    (206, "Partial Content"),
    (207, "Multi-Status"),
    (208, "Already Reported"),
    (226, "IM Used"),
    (300, "Multiple Choices"),
    (301, "Moved Permanently"),
    (302, "Found"),
    (303, "See Other"),
    (304, "Not Modified"),
    (305, "Use Proxy"),
    (307, "Temporary Redirect"),
    (308, "Permanent Redirect"),
    (400, "Bad Request"),
    (401, "Unauthorized"),
    (402, "Payment Required"),
    (403, "Forbidden"),
    (404, "Not Found"),
    (405, "Method Not Allowed"),
    (406, "Not Acceptable"),
    (407, "Proxy Authentication Required"),
    (408, "Request Timeout"),
    (409, "Conflict"),
    (410, "Gone"),
    (411, "Length Required"),
    (412, "Precondition Failed"),
    (413, "Content Too Large"),
    (414, "URI Too Long"),
    (415, "Unsupported Media Type"),
    (416, "Range Not Satisfiable"),
    (417, "Expectation Failed"),
    (418, "I'm a Teapot"),
    (421, "Misdirected Request"),
    (422, "Unprocessable Content"),
    (423, "Locked"),
    (424, "Failed Dependency"),
    (425, "Too Early"),
    (426, "Upgrade Required"),
    (428, "Precondition Required"),
    (429, "Too Many Requests"),
    (431, "Request Header Fields Too Large"),
    (451, "Unavailable For Legal Reasons"),
    (500, "Internal Server Error"),
    (501, "Not Implemented"),
    (502, "Bad Gateway"),
    (503, "Service Unavailable"),
    (504, "Gateway Timeout"),
    (505, "HTTP Version Not Supported"),
    (506, "Variant Also Negotiates"),
    (507, "Insufficient Storage"),
    (508, "Loop Detected"),
    (510, "Not Extended"),
    (511, "Network Authentication Required"),
];

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: dropping the transient list (#5831, a 429 fatal), the pre-#5831 shape where a plugin's
    // `retryable=True` retries a 401, forcing `retryable` outside 4xx, or the 4xx range written
    // without an upper bound or with `<= 400`.
    #[test]
    fn retryability_follows_the_4xx_rule() {
        for (status, caller, want) in [
            (408, None, true),
            (429, None, true),
            (401, None, false),
            (404, None, false),
            (400, Some(true), false),
            (403, Some(true), false),
            (500, None, true),
            (503, None, true),
            (500, Some(false), false),
            (408, Some(false), false),
            (-1, None, true),
            (399, None, true),
            (400, None, false),
            (498, None, false),
        ] {
            assert_eq!(retryable_for_status(status, caller), want, "{status} {caller:?}");
        }
    }

    // Catches: 499 back in the transient set, `ApiError::status` minting `Status` for it, or the
    // caller's `retryable=True` winning over `Cancelled`.
    #[test]
    fn a_499_is_cancelled_and_never_retryable() {
        assert!(!retryable_for_status(499, None));
        assert!(!retryable_for_status(499, Some(true)));
        assert!(!TRANSIENT_CLIENT_STATUSES.contains(&CANCELLED_STATUS));

        assert_eq!(ApiErrorKind::for_status(499), ApiErrorKind::Cancelled);
        assert_eq!(ApiErrorKind::for_status(498), ApiErrorKind::Status);
        assert_eq!(ApiErrorKind::for_status(500), ApiErrorKind::Status);

        let err = ApiError::status("client closed", 499, None, Some(true));
        assert_eq!(err.kind, ApiErrorKind::Cancelled);
        assert!(!err.retryable);
        assert_eq!(err.status_code, Some(499));
        let err = ApiError::from_http("", 499, None);
        assert_eq!(err.kind, ApiErrorKind::Cancelled, "the http factory goes through `status`");
        assert!(!err.retryable);
    }

    // Catches: `Cancelled` falling outside `except APIStatusError`, or naming a nonexistent class.
    #[test]
    fn cancelled_is_a_status_error_and_an_api_error() {
        let c = ApiErrorKind::Cancelled;
        assert!(c.is_a(ApiErrorKind::Cancelled));
        assert!(c.is_a(ApiErrorKind::Status));
        assert!(c.is_a(ApiErrorKind::Api));
        assert!(!c.is_a(ApiErrorKind::Connection));
        assert!(!c.is_a(ApiErrorKind::Timeout));
        assert!(!ApiErrorKind::Status.is_a(ApiErrorKind::Cancelled), "a 500 is not a cancellation");
    }

    // Catches: dropping the `message != reason` arm ("Not Found (404 Not Found)").
    #[test]
    fn the_message_is_shown_only_when_it_adds_detail_beyond_the_reason() {
        assert_eq!(
            http_error_message("quota exceeded", 429, Some("Too Many Requests")),
            "quota exceeded (429 Too Many Requests)"
        );
        assert_eq!(http_error_message("Not Found", 404, Some("Not Found")), "Not Found (404)");
        assert_eq!(http_error_message("", 404, Some("Not Found")), "Not Found (404)");
    }

    // Catches: a mis-sorted phrase table (binary search fails silently).
    #[test]
    fn the_phrase_table_is_sorted_and_covers_the_transient_codes() {
        assert!(PHRASES.windows(2).all(|w| w[0].0 < w[1].0), "PHRASES must be sorted by code");
        assert_eq!(http_reason(429), Some("Too Many Requests"));
        assert_eq!(http_reason(499), None, "499 is nginx's, not IANA's");
    }

    // Catches: an auth or policy close retried forever (every code counted retryable), or a
    // going-away / server-error close made fatal.
    #[test]
    fn close_codes_split_into_normal_final_and_retryable() {
        for (code, want) in [
            (Some(1000), CloseKind::Normal),
            (Some(1001), CloseKind::Retryable),
            (Some(1006), CloseKind::Retryable),
            (Some(1011), CloseKind::Retryable),
            (Some(1013), CloseKind::Retryable),
            (None, CloseKind::Retryable),
            (Some(1008), CloseKind::Final),
            (Some(1003), CloseKind::Final),
            (Some(1007), CloseKind::Final),
            (Some(3000), CloseKind::Final),
            (Some(4401), CloseKind::Final),
            (Some(4403), CloseKind::Final),
            (Some(4429), CloseKind::Retryable),
            (Some(4000), CloseKind::Retryable),
        ] {
            assert_eq!(CloseKind::of(code), want, "{code:?}");
        }
    }

    // Catches: a `kind == Connection` test that misses `APITimeoutError`.
    #[test]
    fn a_timeout_is_a_connection_error_is_an_api_error() {
        assert!(ApiErrorKind::Timeout.is_a(ApiErrorKind::Connection));
        assert!(ApiErrorKind::Timeout.is_a(ApiErrorKind::Api));
        assert!(ApiErrorKind::Status.is_a(ApiErrorKind::Api));
        assert!(ApiErrorKind::Connection.is_a(ApiErrorKind::Connection));
        assert!(!ApiErrorKind::Connection.is_a(ApiErrorKind::Timeout));
        assert!(!ApiErrorKind::Status.is_a(ApiErrorKind::Connection));
    }
}
