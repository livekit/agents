//! Generated host views of native synthesis streams.

use crate::SynthesizedAudio;
use crate::errors::{ProviderError, PullError};

/// One live synthesis.
#[crate::host(class(name = "InferenceTtsStream"))]
pub trait InferenceTtsStream: Send + Sync + Sized + 'static {
    #[host(call)]
    fn push(&self, text: String) -> Result<(), ProviderError>;
    #[host(call)]
    fn flush(&self) -> Result<(), ProviderError>;
    #[host(call)]
    fn end_input(&self) -> Result<(), ProviderError>;
    #[host(await)]
    async fn recv(&self) -> Result<Option<SynthesizedAudio>, PullError>;
    #[host(await)]
    async fn aclose(&self);
}

/// One synthesis, served by whichever entry the chain picks.
#[crate::host(class(name = "FallbackTtsStream"))]
pub trait FallbackTtsStream: Send + Sync + Sized + 'static {
    #[host(call)]
    fn push_text(&self, token: String) -> Result<(), ProviderError>;
    #[host(call)]
    fn flush(&self) -> Result<(), ProviderError>;
    #[host(call)]
    fn end_input(&self) -> Result<(), ProviderError>;
    #[host(await)]
    async fn recv(&self) -> Result<Option<SynthesizedAudio>, PullError>;
    #[host(await)]
    async fn aclose(&self);
}

/// The checks a host synthesis's end applies.
#[crate::host(functions(pid = false))]
pub trait SynthesisChecks {
    /// Why a synthesis of ``text`` that pushed ``duration`` seconds of audio (and, streamed,
    /// ``segments`` as ``(expected, actual)``) failed, or ``None``.
    #[host(function)]
    fn validate_synthesis(
        text: &str,
        duration: f64,
        segments: Option<(usize, usize)>,
    ) -> Option<String>;
}
