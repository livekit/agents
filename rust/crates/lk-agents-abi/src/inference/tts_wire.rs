//! The TTS gateway's error and fallback vocabularies.

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    #[host(rename = "UNSUPPORTED_MESSAGE_TYPE")]
    UnsupportedMessageType,
    #[host(rename = "MISSING_MESSAGE_TYPE")]
    MissingMessageType,
    #[host(rename = "SESSION_NOT_CREATED")]
    SessionNotCreated,
    #[host(rename = "SESSION_ALREADY_CREATED")]
    SessionAlreadyCreated,
    #[host(rename = "INVALID_DATA")]
    InvalidData,
    #[host(rename = "CONNECTION_ERROR")]
    ConnectionError,
    #[host(rename = "BAD_REQUEST")]
    BadRequest,
    #[host(rename = "NOT_FOUND")]
    NotFound,
    #[host(rename = "AUTHENTICATION_ERROR")]
    AuthenticationError,
    #[host(rename = "RATE_LIMITED")]
    RateLimited,
    #[host(rename = "SERVER_ERROR")]
    ServerError,
    #[host(rename = "INACTIVITY_TIMEOUT")]
    InactivityTimeout,
    #[host(rename = "REGION_RESTRICTED")]
    RegionRestricted,
    #[serde(other)]
    #[host(rename = "UNKNOWN")]
    Unknown,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackType {
    Fallback,
    SystemDefault,
    #[serde(other)]
    Unknown,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackCause {
    Timeout,
    ProviderError,
    QuotaExceeded,
    #[serde(other)]
    Unknown,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TimestampTransportStrategy {
    #[host(rename = "TIMESTAMP_TRANSPORT_STRATEGY_UNSPECIFIED")]
    #[serde(rename = "TIMESTAMP_TRANSPORT_STRATEGY_UNSPECIFIED")]
    Unspecified,
    #[host(rename = "SYNC")]
    Sync,
    #[host(rename = "ASYNC")]
    Async,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AutoModeStrategy {
    #[host(rename = "AUTO_MODE_STRATEGY_UNSPECIFIED")]
    #[serde(rename = "AUTO_MODE_STRATEGY_UNSPECIFIED")]
    Unspecified,
    #[host(rename = "CLIENT_SEGMENTED")]
    ClientSegmented,
    #[host(rename = "SENTENCE_BOUNDARY")]
    SentenceBoundary,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RimeSegment {
    #[serde(rename = "never")]
    Never,
    #[host(rename = "bySentence")]
    #[serde(rename = "bySentence")]
    BySentence,
    #[serde(rename = "immediate")]
    Immediate,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElevenlabsTextNormalization {
    On,
    Off,
    Auto,
}
