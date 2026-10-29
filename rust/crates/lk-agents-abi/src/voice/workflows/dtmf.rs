//! Public keypad reception and result contracts.

/// A collected, space-separated keypad sequence.
#[crate::host(class(name = "GetDtmfResult", pid = false))]
pub trait DtmfResult: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(#[host(kw)] user_input: String) -> Self;
    #[host(getter)]
    fn user_input(&self) -> String;
    #[host(call, name = "__repr__")]
    fn host_repr(&self) -> String;
    #[host(call, name = "__eq__")]
    fn host_eq(&self, other: &Self) -> bool;
    #[host(call)]
    fn from_dtmf_inputs(events: Vec<String>) -> Result<Self, DtmfInputError>;
}

/// An invalid key in a supplied keypad sequence.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct DtmfInputError(pub String);

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
    #[host(class(name = "GetDtmfResult"))]
    type Result: DtmfResult;

    #[host(function)]
    async fn send_dtmf_events(
        room: crate::voice::room_owner::RoomSession,
        events: Vec<String>,
    ) -> String;
    #[host(function)]
    async fn receive_dtmf(
        room: crate::voice::room_owner::RoomSession,
        num_digits: usize,
        timeout: f64,
        stop_event: String,
    ) -> Result<Self::Result, ReceiveDtmfError>;
}
