//! `APIError` & co. as a plain Rust value, plus the rules its host classes apply.

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

pub const CANCELLED_STATUS: i64 = 499;

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
}

/// The wait an `APITimeoutError` bounded, as `APIConnectOptions` names its limits. Absent when the
/// failing layer does not say (a foreign SDK timeout), never guessed. Its wire string is the
/// Python value of `APITimeoutError.phase`.
#[crate::host(wire_enum)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeoutPhase {
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
    kinds(
        Api(class = "livekit.agents._exceptions.APIError", kwargs(retryable)),
        Status(
            class = "livekit.agents._exceptions.APIStatusError",
            kwargs(retryable, status_code, request_id)
        ),
        // The reference detects cancellation as `APIStatusError` with status 499.
        Cancelled(
            class = "livekit.agents._exceptions.APIStatusError",
            kwargs(retryable, status_code, request_id)
        ),
        Connection(class = "livekit.agents._exceptions.APIConnectionError", kwargs(retryable)),
        Timeout(class = "livekit.agents._exceptions.APITimeoutError", kwargs(retryable, phase)),
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
    pub phase: Option<TimeoutPhase>,
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
            phase: None,
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
            phase: None,
        }
    }

    pub fn connection(message: Option<String>) -> Self {
        Self {
            kind: ApiErrorKind::Connection,
            message: message.unwrap_or_else(|| DEFAULT_CONNECTION_MESSAGE.to_owned()),
            retryable: true,
            status_code: None,
            request_id: None,
            phase: None,
        }
    }

    pub fn timeout(message: Option<String>) -> Self {
        Self {
            kind: ApiErrorKind::Timeout,
            message: message.unwrap_or_else(|| DEFAULT_TIMEOUT_MESSAGE.to_owned()),
            retryable: true,
            status_code: None,
            request_id: None,
            phase: None,
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

    /// An `APITimeoutError` for the wait `phase` bounded.
    pub fn timed_out(phase: TimeoutPhase) -> Self {
        Self {
            phase: Some(phase),
            ..Self::timeout(Some(format!("Request timed out ({} timeout).", phase.as_str())))
        }
    }

    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
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
mod tests;
