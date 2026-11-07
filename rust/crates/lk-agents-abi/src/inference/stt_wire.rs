//! Numeric error codes emitted by the gateway's STT protocol.

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttErrorCode {
    #[host(rename = "2000")]
    UnsupportedMessageType,
    #[host(rename = "2001")]
    MissingMessageType,
    #[host(rename = "2002")]
    SessionNotCreated,
    #[host(rename = "2003")]
    SessionAlreadyCreated,
    #[host(rename = "2004")]
    InvalidData,
    #[host(rename = "2005")]
    QuotaExceeded,
    #[host(rename = "2006")]
    ConnectionError,
    #[host(rename = "2007")]
    InactivityTimeout,
    #[host(rename = "2008")]
    ServiceUnavailable,
}

impl SttErrorCode {
    pub fn retryable(self) -> bool {
        matches!(
            self,
            Self::QuotaExceeded
                | Self::ConnectionError
                | Self::InactivityTimeout
                | Self::ServiceUnavailable
        )
    }
}
