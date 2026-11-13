use super::{ApiErrorKind, ErrorKind};

/// Several may be true.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TaxonomyMatches {
    pub timeout: bool,
    pub connection: bool,
    pub status: bool,
    pub api: bool,
}

impl ApiErrorKind {
    /// Most specific first; `None` means outside the taxonomy. A 499 status is `Cancelled`.
    pub fn from_matches(m: TaxonomyMatches, status_code: Option<i64>) -> Option<ApiErrorKind> {
        if m.timeout {
            Some(ApiErrorKind::Timeout)
        } else if m.connection {
            Some(ApiErrorKind::Connection)
        } else if m.status {
            Some(status_code.map_or(ApiErrorKind::Status, ApiErrorKind::for_status))
        } else if m.api {
            Some(ApiErrorKind::Api)
        } else {
            None
        }
    }
}

/// Only inside the taxonomy, only for a kind worth another attempt, and never when the object
/// was made final (`retryable` cleared). Without the classes the kind is trusted.
pub fn host_retryable(
    api_kind: Option<ApiErrorKind>,
    taxonomy_available: bool,
    kind: Option<ErrorKind>,
    declared: Option<bool>,
) -> bool {
    let in_taxonomy = api_kind.is_some() || !taxonomy_available;
    in_taxonomy && kind.is_some_and(ErrorKind::retryable) && declared.unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: testing a base class before its subclass (`Timeout` never answered).
    #[test]
    fn the_most_specific_class_wins() {
        let timeout = TaxonomyMatches { timeout: true, connection: true, api: true, status: false };
        assert_eq!(ApiErrorKind::from_matches(timeout, None), Some(ApiErrorKind::Timeout));
        let connection = TaxonomyMatches { connection: true, api: true, ..Default::default() };
        assert_eq!(ApiErrorKind::from_matches(connection, None), Some(ApiErrorKind::Connection));
        let status = TaxonomyMatches { status: true, api: true, ..Default::default() };
        assert_eq!(ApiErrorKind::from_matches(status, Some(500)), Some(ApiErrorKind::Status));
        assert_eq!(ApiErrorKind::from_matches(status, None), Some(ApiErrorKind::Status));
        assert_eq!(ApiErrorKind::from_matches(status, Some(499)), Some(ApiErrorKind::Cancelled));
        let api = TaxonomyMatches { api: true, ..Default::default() };
        assert_eq!(ApiErrorKind::from_matches(api, None), Some(ApiErrorKind::Api));
        assert_eq!(ApiErrorKind::from_matches(TaxonomyMatches::default(), Some(429)), None);
    }

    // Catches: retrying outside the taxonomy, a missing kind read as retryable, a final kind
    // retried, an error the core made final retried again, or never retrying when the classes
    // cannot be resolved.
    #[test]
    fn retryable_needs_the_taxonomy_and_a_transient_kind() {
        let status = Some(ApiErrorKind::Status);
        assert!(host_retryable(status, true, Some(ErrorKind::Server), None));
        assert!(!host_retryable(status, true, Some(ErrorKind::Auth), None));
        assert!(!host_retryable(status, true, None, None), "no kind is never retried");
        assert!(!host_retryable(status, true, Some(ErrorKind::Server), Some(false)));
        assert!(!host_retryable(None, true, Some(ErrorKind::Server), None), "outside the taxonomy");
        assert!(host_retryable(None, false, Some(ErrorKind::Server), None), "no classes: the kind");
        assert!(!host_retryable(None, false, None, None));
    }
}

/// Provider exception facts; missing or unreadable optional attributes stay absent.
#[crate::host(record(
    module = "host_errors",
    name = "ProviderErrorMetadata",
    host_type = "builtins.BaseException",
    boundary,
    extract
))]
#[derive(Default)]
pub struct ProviderErrorMetadata {
    #[host(lenient)]
    pub status_code: Option<i64>,
    /// `APIError.kind`: what failed.
    #[host(lenient)]
    pub kind: Option<ErrorKind>,
    /// `APIError.retryable`: cleared on an error the core made final.
    #[host(lenient)]
    pub retryable: Option<bool>,
    /// `APIError.detail`: the provider's own message.
    #[host(lenient)]
    pub detail: Option<String>,
    /// `APIStatusError.retry_after`, in seconds.
    #[host(lenient)]
    pub retry_after: Option<f64>,
}

#[crate::host(record(
    module = "host_errors",
    name = "RealtimeErrorMetadata",
    host_type = "builtins.BaseException",
    boundary,
    extract
))]
#[derive(Default)]
pub struct RealtimeErrorMetadata {
    #[host(own_value, extract_with = "py_values::exception_message")]
    pub message: String,
    #[host(lenient)]
    pub code: Option<String>,
}
