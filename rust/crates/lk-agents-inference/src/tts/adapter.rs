//! `tts.StreamAdapter`'s pre-split host entry: the host ran its own tokenizer and pacer, so each
//! push is one sentence, converted to the provider's markup and synthesized within the budget.

use std::sync::Arc;

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::tts::stream_adapter::{SentenceSplitter, StreamAdapterFunctions};
use lk_agents_abi::{ApiConnectOptions, Tts};

use lk_agents_net::env::ProcessEnv;

use super::sentence_adapter::StreamAdapter;
use super::stream::TtsStream;

pub struct AdapterStreams;

impl StreamAdapterFunctions for AdapterStreams {
    type Stream = TtsStream;
    type Env = ProcessEnv;

    async fn tts_adapter_stream(
        env: &ProcessEnv,
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
            Arc::new(lk_agents_core::tts::markup::text_processor),
            env.clock.clone(),
        )
        .with_budget(conn_options)
        // The host's adapter lowers the markup before each pre-split sentence.
        .stream(false)
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
