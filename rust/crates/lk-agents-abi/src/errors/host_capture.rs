//! The binding reads the facts of a captured host exception; the decision lives here.

use super::ApiErrorKind;

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

/// Only inside the taxonomy and only if `.retryable` says so. Without the classes the flag is
/// trusted.
pub fn host_retryable(
    api_kind: Option<ApiErrorKind>,
    taxonomy_available: bool,
    declared: Option<bool>,
) -> bool {
    let in_taxonomy = api_kind.is_some() || !taxonomy_available;
    in_taxonomy && declared.unwrap_or(false)
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


    // Catches: retrying outside the taxonomy, assuming retryable when unset, or never retrying
    // when the classes cannot be resolved.
    #[test]
    fn retryable_needs_the_taxonomy_and_the_objects_own_word() {
        assert!(host_retryable(Some(ApiErrorKind::Status), true, Some(true)));
        assert!(!host_retryable(Some(ApiErrorKind::Status), true, Some(false)));
        assert!(!host_retryable(Some(ApiErrorKind::Status), true, None), "no attribute is no");
        assert!(!host_retryable(None, true, Some(true)), "outside the taxonomy is never retried");
        assert!(host_retryable(None, false, Some(true)), "no classes to check: trust the object");
        assert!(!host_retryable(None, false, None));
    }
}
