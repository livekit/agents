//! Public keypad reception and result contracts.

/// Why room digit reception stopped short.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[crate::host(error(kind = "livekit.agents.llm.tool_context.ToolError"))]
pub enum ReceiveDtmfError {
    #[error("num_digits must be greater than 0")]
    #[host(kind = "ValueError")]
    NoDigits,
    /// `DtmfEvent(digit)` refused the key, quoted as Python's `repr`.
    #[error("{0} is not a valid DtmfEvent")]
    #[host(kind = "ValueError")]
    UnknownKey(String),
    #[error("Digits input not fully received. Expect {0} digits, got {1}")]
    Incomplete(usize, usize),
    #[error("room event stream failed: {0}")]
    Room(String),
}

/// Runtime publication and collection of room keypad tones.
#[crate::host(functions)]
pub trait DtmfIo {
    #[host(function)]
    async fn send_dtmf_events(
        room: std::sync::Arc<dyn crate::voice::room_owner::SessionRoom>,
        events: Vec<String>,
    ) -> String;
    #[host(function)]
    async fn receive_dtmf(
        room: std::sync::Arc<dyn crate::voice::room_owner::SessionRoom>,
        num_digits: usize,
        timeout: f64,
        stop_event: String,
    ) -> Result<String, ReceiveDtmfError>;
}

#[crate::host(record(
    module = "workflows",
    path = "livekit.agents.beta.workflows",
    enum,
    enum_base = "StrEnum"
))]
pub enum DtmfEvent {
    #[host(rename = "1")]
    One,
    #[host(rename = "2")]
    Two,
    #[host(rename = "3")]
    Three,
    #[host(rename = "4")]
    Four,
    #[host(rename = "5")]
    Five,
    #[host(rename = "6")]
    Six,
    #[host(rename = "7")]
    Seven,
    #[host(rename = "8")]
    Eight,
    #[host(rename = "9")]
    Nine,
    #[host(rename = "0")]
    Zero,
    #[host(rename = "*")]
    Star,
    #[host(rename = "#")]
    Pound,
    #[host(rename = "A")]
    A,
    #[host(rename = "B")]
    B,
    #[host(rename = "C")]
    C,
    #[host(rename = "D")]
    D,
}
