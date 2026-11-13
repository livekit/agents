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

/// The status of a request we closed ([`ApiErrorKind::Cancelled`]).
const CANCELLED_STATUS: i64 = 499;

/// What failed, as the provider meant it. A plugin names the kind; only [`ErrorKind::retryable`]
/// turns it into a retry decision.
#[crate::host(record(module = "types", path = "livekit.agents.types", emit, extract))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// The credentials were refused (401, 403): a new attempt fails the same way.
    Auth,
    /// The account is out of quota or credit (402).
    Quota,
    /// The provider refused what was sent (another 4xx): its options, input or protocol.
    InvalidRequest,
    /// The provider asked to slow down (429).
    RateLimit,
    /// The provider failed (5xx).
    Server,
    /// The connection could not be made or was lost.
    Connection,
    /// A wait for the provider expired (408).
    Timeout,
    /// The provider broke its own contract: an answer or frame whose fields we read are missing
    /// or have the wrong type, or an answer that does not parse. The offending payload is the
    /// error's `detail`.
    Protocol,
}

impl ErrorKind {
    /// The one place a kind becomes a retry decision.
    pub fn retryable(self) -> bool {
        match self {
            ErrorKind::RateLimit
            | ErrorKind::Server
            | ErrorKind::Connection
            | ErrorKind::Timeout => true,
            ErrorKind::Auth
            | ErrorKind::Quota
            | ErrorKind::InvalidRequest
            | ErrorKind::Protocol => false,
        }
    }

    /// What an HTTP status says failed. A redirect, never followed, answers every attempt the
    /// same; any other status outside 3xx and 4xx is the provider's failure.
    pub fn for_status(status_code: i64) -> ErrorKind {
        match status_code {
            401 | 403 => ErrorKind::Auth,
            402 => ErrorKind::Quota,
            408 => ErrorKind::Timeout,
            429 => ErrorKind::RateLimit,
            300..=499 => ErrorKind::InvalidRequest,
            _ => ErrorKind::Server,
        }
    }

    /// What a websocket close code says failed; `None` for 1000, a close on purpose. RFC 6455
    /// section 7.4, IANA's 3000 range, and a private 44xx code read as its HTTP status.
    pub fn for_close(code: Option<u16>) -> Option<ErrorKind> {
        Some(match code {
            Some(1000) => return None,
            Some(1008 | 3000 | 3003) => ErrorKind::Auth,
            Some(1002 | 1003 | 1007 | 1009 | 1010 | 1015) => ErrorKind::InvalidRequest,
            Some(1011 | 1012 | 1014) => ErrorKind::Server,
            Some(1013) => ErrorKind::RateLimit,
            Some(3008) => ErrorKind::Timeout,
            Some(code @ 4400..=4499) => ErrorKind::for_status(i64::from(code) - 4000),
            _ => ErrorKind::Connection,
        })
    }
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

    /// `APIStatusError`'s kind: the one its caller named, else its status's.
    #[host(function)]
    fn error_kind_for_status(status_code: i64, kind: Option<ErrorKind>) -> ErrorKind;

    /// `APIError.retryable`: whether its kind is worth another attempt.
    #[host(function)]
    fn error_kind_retryable(kind: ErrorKind) -> bool;
}

/// The host's websocket helpers.
#[crate::host(functions(pid = false))]
pub trait WebSocketFunctions {
    /// What a close code says failed: `http.WebSocket.close_error`'s kind, `None` for a close on
    /// purpose.
    #[host(function)]
    fn websocket_close_kind(code: Option<u16>) -> Option<ErrorKind>;
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

/// `body` stays on the Python object. The host class derives `retryable` from `kind`; a native
/// error the core made final has that attribute cleared after construction.
#[crate::host(error(
    by = kind,
    cause = cause,
    attrs(retryable),
    kinds(
        Api(class = "livekit.agents._bindings.APIError", kwargs(kind = error_kind)),
        Status(
            class = "livekit.agents._bindings.APIStatusError",
            kwargs(kind = error_kind, status_code, request_id)
        ),
        // The reference detects cancellation as `APIStatusError` with status 499.
        Cancelled(
            class = "livekit.agents._bindings.APIStatusError",
            kwargs(kind = error_kind, status_code, request_id)
        ),
        Connection(class = "livekit.agents._bindings.APIConnectionError", kwargs(kind = error_kind)),
        Timeout(class = "livekit.agents._bindings.APITimeoutError", kwargs(timeout)),
    )
))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ApiError {
    /// The host class it raises as.
    pub kind: ApiErrorKind,
    /// What failed: `retryable` starts as its [`ErrorKind::retryable`].
    pub error_kind: ErrorKind,
    pub message: String,
    /// Whether the retry owner may try again: the kind's answer until the core made it final.
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
    fn of(kind: ApiErrorKind, error_kind: ErrorKind, message: String) -> Self {
        Self {
            kind,
            error_kind,
            message,
            retryable: error_kind.retryable(),
            status_code: None,
            request_id: None,
            timeout: None,
            cause: None,
            said: None,
        }
    }

    /// A native failure: `kind` names what failed and decides its retry.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self::of(ApiErrorKind::Api, kind, message.into())
    }

    /// The provider broke its contract; `payload` is what it sent, logged once as content.
    pub fn protocol(message: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            said: ProviderSaid::boxed(Some(payload.into()), None),
            ..Self::new(ErrorKind::Protocol, message)
        }
    }

    /// Of its status's kind; 499 comes out as `Cancelled`, never retried.
    pub fn status(
        message: impl Into<String>,
        status_code: i64,
        request_id: Option<String>,
    ) -> Self {
        let error = Self {
            status_code: Some(status_code),
            request_id,
            ..Self::of(
                ApiErrorKind::for_status(status_code),
                ErrorKind::for_status(status_code),
                message.into(),
            )
        };
        if error.kind == ApiErrorKind::Cancelled { error.into_final() } else { error }
    }

    pub fn connection(message: Option<String>) -> Self {
        let message = message.unwrap_or_else(|| DEFAULT_CONNECTION_MESSAGE.to_owned());
        Self::of(ApiErrorKind::Connection, ErrorKind::Connection, message)
    }

    pub fn timeout(message: Option<String>) -> Self {
        let message = message.unwrap_or_else(|| DEFAULT_TIMEOUT_MESSAGE.to_owned());
        Self::of(ApiErrorKind::Timeout, ErrorKind::Timeout, message)
    }

    pub fn from_http(message: &str, status: i64, request_id: Option<String>) -> Self {
        Self::status(http_error_message(message, status, http_reason(status)), status, request_id)
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
        Self::status(message, CANCELLED_STATUS, None)
    }

    /// The same failure as another kind, its retry decision that kind's.
    pub fn with_kind(self, error_kind: ErrorKind) -> Self {
        Self { error_kind, retryable: error_kind.retryable(), ..self }
    }

    /// The same failure, no longer to be retried: its retry owner gave up, or output it cannot
    /// replay was already delivered.
    pub fn into_final(self) -> Self {
        Self { retryable: false, ..self }
    }

    /// The inference gateway's `X-Should-Retry` verdict over the status's kind: `false` makes
    /// it final, `true` a provider failure worth another attempt.
    pub fn with_should_retry(self, should_retry: Option<bool>) -> Self {
        match should_retry {
            Some(false) => self.into_final(),
            Some(true) if !self.retryable => self.with_kind(ErrorKind::Server),
            _ => self,
        }
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

    // Catches: a final kind retried (auth, quota or a broken provider contract spent on the retry
    // budget) or a transient one made final (a 429 or a dropped connection failing the request).
    #[test]
    fn only_rate_limit_server_connection_and_timeout_retry() {
        for (kind, want) in [
            (ErrorKind::Auth, false),
            (ErrorKind::Quota, false),
            (ErrorKind::InvalidRequest, false),
            (ErrorKind::RateLimit, true),
            (ErrorKind::Server, true),
            (ErrorKind::Connection, true),
            (ErrorKind::Timeout, true),
            (ErrorKind::Protocol, false),
        ] {
            assert_eq!(kind.retryable(), want, "{kind:?}");
        }
    }

    // Catches: a 401 or 402 read as a generic client error, 408/429 made final, the 4xx range
    // written without an upper bound, a refused redirect retried, or a 5xx made final.
    #[test]
    fn a_status_names_its_kind() {
        for (status, want) in [
            (401, ErrorKind::Auth),
            (403, ErrorKind::Auth),
            (402, ErrorKind::Quota),
            (408, ErrorKind::Timeout),
            (429, ErrorKind::RateLimit),
            (400, ErrorKind::InvalidRequest),
            (404, ErrorKind::InvalidRequest),
            (499, ErrorKind::InvalidRequest),
            (500, ErrorKind::Server),
            (503, ErrorKind::Server),
            (302, ErrorKind::InvalidRequest),
            (299, ErrorKind::Server),
        ] {
            assert_eq!(ErrorKind::for_status(status), want, "{status}");
        }
    }

    // Catches: 499 retried, or `ApiError::status` minting `Status` for it.
    #[test]
    fn a_499_is_cancelled_and_never_retryable() {
        assert_eq!(ApiErrorKind::for_status(499), ApiErrorKind::Cancelled);
        assert_eq!(ApiErrorKind::for_status(498), ApiErrorKind::Status);
        assert_eq!(ApiErrorKind::for_status(500), ApiErrorKind::Status);

        let err = ApiError::status("client closed", 499, None);
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

    // Catches: an auth or policy close retried forever (every code read as a dropped
    // connection), a going-away or server-error close made final, or a 1000 read as a failure.
    #[test]
    fn a_close_code_names_its_kind() {
        for (code, want) in [
            (Some(1000), None),
            (Some(1001), Some(ErrorKind::Connection)),
            (Some(1006), Some(ErrorKind::Connection)),
            (None, Some(ErrorKind::Connection)),
            (Some(1011), Some(ErrorKind::Server)),
            (Some(1013), Some(ErrorKind::RateLimit)),
            (Some(1008), Some(ErrorKind::Auth)),
            (Some(3000), Some(ErrorKind::Auth)),
            (Some(1003), Some(ErrorKind::InvalidRequest)),
            (Some(1007), Some(ErrorKind::InvalidRequest)),
            (Some(4401), Some(ErrorKind::Auth)),
            (Some(4402), Some(ErrorKind::Quota)),
            (Some(4400), Some(ErrorKind::InvalidRequest)),
            (Some(4429), Some(ErrorKind::RateLimit)),
            (Some(4000), Some(ErrorKind::Connection)),
        ] {
            assert_eq!(ErrorKind::for_close(code), want, "{code:?}");
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
