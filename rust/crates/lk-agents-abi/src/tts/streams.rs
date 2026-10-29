//! Generated host views of native synthesis streams.

use crate::SynthesizedAudio;
use crate::errors::{ProviderError, PullError};

/// One synthesis, `SynthesizeStream`'s shape: text in through `push_text`, `flush` and
/// `end_input`, audio out through `recv`. `aclose` cancels it; it does not end the input.
#[crate::host(class(name = "TtsStream"))]
pub trait TtsStream: Send + Sync + Sized + 'static {
    #[host(call)]
    fn push_text(&self, text: String) -> Result<(), ProviderError>;
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
