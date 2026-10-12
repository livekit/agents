use super::*;
use crate::errors::side_table::testkit::Table;
use crate::errors::ApiErrorKind;

fn host(retryable: bool, status_code: Option<i64>) -> ProviderError {
    let table = Table::holding([7]);
    ProviderError::Host {
        error: HostError::captured("MyPluginRateLimit", "quota exceeded", CapturedError::new(7, &table.as_registry())),
        api_kind: Some(ApiErrorKind::Status),
        retryable,
        status_code,
    }
}

// Catches: re-deriving retryability from `status_code` instead of trusting the host's flag.
#[test]
fn a_host_errors_retryable_flag_is_carried_not_recomputed() {
    assert!(host(true, Some(429)).retryable());
    assert!(!host(false, Some(429)).retryable(), "the host's own False must win");
    assert!(host(true, None).retryable(), "no status code is not a reason to give up");
}

// Catches: `Local(_) => false`, which never retries a native 503.
#[test]
fn a_native_error_reports_its_own_retryability() {
    assert!(ProviderError::Local(ApiError::status("busy", 503, None, None)).retryable());
    assert!(!ProviderError::Local(ApiError::status("bad key", 401, None, None)).retryable());
    assert!(ProviderError::Local(ApiError::timeout(None)).retryable());
}

// Catches: handing out a `CapturedError` for a native error.
#[test]
fn only_a_host_error_has_a_host_ref() {
    assert_eq!(host(true, Some(429)).host_ref().map(|r| r.id()), Some(7));
    assert_eq!(ProviderError::Local(ApiError::new("x")).host_ref(), None);
}

// Catches: losing the status code or the message on the way through.
#[test]
fn the_status_code_and_message_survive_both_variants() {
    assert_eq!(host(true, Some(429)).message(), "quota exceeded");
    assert_eq!(
        ProviderError::Local(ApiError::new("upstream said no")).message(),
        "upstream said no"
    );
    assert_eq!(host(true, Some(429)).status_code(), Some(429));
    assert_eq!(
        ProviderError::Local(ApiError::status("x", 500, None, None)).status_code(),
        Some(500)
    );
    assert_eq!(ProviderError::Local(ApiError::connection(None)).status_code(), None);
}
