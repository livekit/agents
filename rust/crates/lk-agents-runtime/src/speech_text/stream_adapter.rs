//! The pre-split host adapter with voice-owned provider text processing.

use std::sync::Arc;

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::tts::stream_adapter::{SentenceSplitter, StreamAdapter, StreamAdapterFunctions};
use lk_agents_abi::{ApiConnectOptions, Tts};

use super::stream::TtsStream;

pub struct AdapterStreams;

impl StreamAdapterFunctions for AdapterStreams {
    type Stream = TtsStream;

    async fn tts_adapter_stream(
        provider: Arc<dyn Tts>,
        conn_options: ApiConnectOptions,
    ) -> Result<Self::Stream, ProviderError> {
        conn_options.validate().map_err(|error| {
            ProviderError::Local(ApiError { retryable: false, ..ApiError::new(error.to_string()) })
        })?;
        let provider = provider.with_conn_options(&conn_options).unwrap_or(provider);
        let (sink, source) = StreamAdapter::new(
            provider,
            Arc::new(|_| Box::new(PreSplit)),
            lk_agents_voice::tts::markup::text_processor_factory(),
            crate::process::host_clock(),
        )
        .with_budget(conn_options)
        .stream()
        .await?;
        Ok(TtsStream::new(Some(sink), source))
    }
}

/// Each push is one sentence: the host ran its own tokenizer and pacer.
struct PreSplit;

impl SentenceSplitter for PreSplit {
    fn push(&mut self, text: &str) -> Vec<String> {
        vec![text.to_owned()]
    }

    fn flush(&mut self) -> Vec<String> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests;
