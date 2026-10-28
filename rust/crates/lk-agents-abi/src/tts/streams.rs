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
