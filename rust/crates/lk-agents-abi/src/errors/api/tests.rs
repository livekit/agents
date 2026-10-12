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

// Catches: an unregistered code rendered as "None" or "" instead of "HTTP <n>".
#[test]
fn an_unknown_status_falls_back_to_the_bare_code() {
    assert_eq!(http_error_message("", 599, None), "HTTP 599 (599)");
    assert_eq!(http_error_message("boom", 520, None), "boom (520 HTTP 520)");
    assert_eq!(http_reason(599), None);
}

// Catches: a mis-sorted phrase table (binary search fails silently).
#[test]
fn the_phrase_table_is_sorted_and_covers_the_transient_codes() {
    assert!(PHRASES.windows(2).all(|w| w[0].0 < w[1].0), "PHRASES must be sorted by code");
    assert_eq!(http_reason(429), Some("Too Many Requests"));
    assert_eq!(http_reason(499), None, "499 is nginx's, not IANA's");
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

// Catches: an `ApiError` constructor bypassing `retryable_for_status`.
#[test]
fn the_status_constructor_applies_the_4xx_rule() {
    assert!(!ApiError::status("nope", 401, None, Some(true)).retryable);
    assert!(ApiError::status("slow down", 429, None, None).retryable);
    assert_eq!(ApiError::status("nope", 401, None, None).kind, ApiErrorKind::Status);
}

// Catches: the factory returning a non-Status kind, or losing the composed message.
#[test]
fn from_http_is_always_a_status_error_and_carries_the_composed_message() {
    let err = ApiError::from_http("quota exceeded", 429, Some("req-1".into()));
    assert_eq!(err.kind, ApiErrorKind::Status);
    assert_eq!(err.message, "quota exceeded (429 Too Many Requests)");
    assert_eq!(err.status_code, Some(429));
    assert_eq!(err.request_id.as_deref(), Some("req-1"));
    assert!(err.retryable);
    assert_eq!(ApiError::from_http("", 204, None).kind, ApiErrorKind::Status);
}

