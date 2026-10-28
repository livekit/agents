//! The console transport and its session audio legs. TCP execution belongs to the runtime.

use crate::errors::HostError;
use crate::voice::io::{AudioInput, AudioOutput};
use livekit_protocol::agent as proto;
use std::sync::Arc;

/// `SessionTransport` (`remote_session.py:78`), as `voice/remote_session.py`'s `_CoreTransport`
/// sees it. A host failure keeps the original exception.
pub trait SessionTransport: Send + Sync {
    /// `start()`; idempotent.
    fn start(&self) -> crate::BoxFuture<'_, Result<(), HostError>>;

    /// `send_message`. Failures are raised: only the caller knows whether anyone waits on it.
    fn send_message<'a>(
        &'a self,
        msg: &'a proto::AgentSessionMessage,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;

    /// `send_message_threadsafe`: enqueue and forget, dropped when closed. Not ordered against
    /// `send_message`.
    fn send_message_nowait(&self, msg: &proto::AgentSessionMessage);

    /// The next message (`__anext__`); `Ok(None)` is `StopAsyncIteration`, `Err` ends the loop.
    /// Only the receive loop reads.
    fn recv(&self) -> crate::BoxFuture<'_, Result<Option<proto::AgentSessionMessage>, HostError>>;

    /// `close()`; idempotent, never fails.
    fn close(&self) -> crate::BoxFuture<'_, ()>;
}

/// The console's local legs (`TcpAudioInput.push_frame`, `TcpAudioOutput.notify_playout_finished`),
/// on `SessionHost`.
pub trait ConsoleSink: Send + Sync {
    fn push_input_frame(&self, frame: &proto::agent_session_message::console_io::AudioFrame);

    fn playout_finished(&self);
}

/// The native console socket as the session's transport and audio legs.
#[crate::host(native(refused = "expected a native ConsoleIo"))]
pub trait ConsolePort: Send + Sync {
    fn as_transport(self: Arc<Self>) -> Arc<dyn SessionTransport>;
    fn as_console(self: Arc<Self>) -> Arc<dyn ConsoleSink>;
    fn as_audio_input(self: Arc<Self>) -> Arc<dyn AudioInput>;
    fn as_audio_output(self: Arc<Self>) -> Arc<dyn AudioOutput>;
}

/// `lk agent console`'s socket, opened when the session starts.
#[crate::host(class(name = "ConsoleIo", native = ConsolePort))]
pub trait ConsoleIo: ConsolePort + Sized + 'static {
    #[host(new)]
    fn new(addr: String) -> Result<Self, crate::errors::ProviderError>;
}

/// Starts a native voice session with its runtime-owned queue executor.
#[crate::host(functions)]
pub trait VoiceSessionExecution {
    #[host(function, name = "_core_voice_session_start")]
    async fn start(
        session: crate::HostRef,
        capture_run: bool,
    ) -> Result<(), crate::errors::ProviderError>;
}
