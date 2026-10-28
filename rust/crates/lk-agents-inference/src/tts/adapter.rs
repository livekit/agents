//! `tts.StreamAdapter`'s pre-split host entry: the host ran its own tokenizer and pacer, so each
//! push is one sentence, converted to the provider's markup and synthesized within the budget.

use std::marker::PhantomData;
use std::sync::Arc;

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::tts::stream_adapter::{SentenceSplitter, StreamAdapter, StreamAdapterFunctions};
use lk_agents_abi::{ApiConnectOptions, Tts};

use super::TtsHost;
use super::stream::TtsStream;

pub struct AdapterStreams<E>(PhantomData<fn() -> E>);

impl<E: TtsHost> StreamAdapterFunctions for AdapterStreams<E> {
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
            Arc::new(|provider| E::markup(provider)),
            E::clock(),
        )
        .with_budget(conn_options)
        .stream()
        .await?;
        Ok(TtsStream::new(Some(sink), source))
    }
}

/// Each push is already one sentence.
struct PreSplit;

impl SentenceSplitter for PreSplit {
    fn push(&mut self, text: &str) -> Vec<String> {
        vec![text.to_owned()]
    }

    fn flush(&mut self) -> Vec<String> {
        Vec::new()
    }
}
