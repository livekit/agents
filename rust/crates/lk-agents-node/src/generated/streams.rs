//! GENERATED FILE -- DO NOT EDIT. Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`); see `napi.rs` in `lk-agents-codegen`.

#![allow(clippy::all, dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]

use napi::Env;
use napi_derive::napi;

/// One live synthesis.
#[napi(js_name = "InferenceTtsStream")]
pub struct InferenceTtsStream {
    pub(crate) sink: std::sync::Arc<dyn lk_agents_abi::StreamSink<String>>,
    pub(crate) source: std::sync::Arc<tokio::sync::Mutex<Option<lk_agents_abi::TtsSource>>>,
}

impl InferenceTtsStream {
    pub fn new(sink: impl Into<std::sync::Arc<dyn lk_agents_abi::StreamSink<String>>>, source: lk_agents_abi::TtsSource) -> Self {
        InferenceTtsStream { sink: sink.into(), source: std::sync::Arc::new(tokio::sync::Mutex::new(Some(source))) }
    }
}

#[napi]
impl InferenceTtsStream {
    #[napi(js_name = "push")]
    pub fn push(&self, env: Env, text: String) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let text = text;
            self.sink.push(text).map_err(crate::host_error::provider_error_to_node)
        })
    }

    #[napi(js_name = "flush")]
    pub fn flush(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || self.sink.flush().map_err(crate::host_error::provider_error_to_node))
    }

    #[napi(js_name = "endInput")]
    pub fn end_input(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.sink.end_input();
            Ok(())
        })
    }

    /// The next item, or `null` at the end of the stream.
    #[napi(js_name = "recv", ts_return_type = "Promise<SynthesizedAudio | null>")]
    pub fn recv<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, Option<crate::generated::records::SynthesizedAudio>>> {
        let source = std::sync::Arc::clone(&self.source);
        env.spawn_future_with_callback(
            async move {
                let mut guard = source.lock().await;
                let Some(src) = guard.as_mut() else { return Ok(Ok(None)) };
                Ok(match src.next().await {
                    Ok(Some(v)) => Ok(Some(crate::generated::records::SynthesizedAudio::from(&v))),
                    Ok(None) => Ok(None),
                    Err(e) => Err(crate::host_error::provider_error_to_node(e)),
                })
            },
            |env, out| out.map_err(|e| e.into_napi(env)),
        )
    }

    #[napi(js_name = "aclose")]
    pub async fn aclose(&self) -> napi::Result<()> {
        self.sink.end_input();
        if let Some(src) = self.source.lock().await.take() {
            src.close(lk_agents_abi::Ts::from_nanos(u64::MAX)).await;
        }
        Ok(())
    }
}

/// One synthesis, served by whichever entry the chain picks.
#[napi(js_name = "FallbackTtsStream")]
pub struct FallbackTtsStream {
    pub(crate) sink: Option<std::sync::Arc<dyn lk_agents_abi::StreamSink<String>>>,
    pub(crate) source: std::sync::Arc<tokio::sync::Mutex<Option<lk_agents_abi::TtsSource>>>,
}

impl FallbackTtsStream {
    pub fn new(sink: Option<impl Into<std::sync::Arc<dyn lk_agents_abi::StreamSink<String>>>>, source: lk_agents_abi::TtsSource) -> Self {
        FallbackTtsStream { sink: sink.map(Into::into), source: std::sync::Arc::new(tokio::sync::Mutex::new(Some(source))) }
    }

    fn sink(&self) -> Result<&std::sync::Arc<dyn lk_agents_abi::StreamSink<String>>, crate::host_error::JsError> {
        self.sink.as_ref().ok_or_else(|| crate::host_error::JsError::new("RuntimeError", "this is a synthesize() stream; its text was given at the call"))
    }
}

#[napi]
impl FallbackTtsStream {
    #[napi(js_name = "pushText")]
    pub fn push_text(&self, env: Env, token: String) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let token = token;
            self.sink()?.push(token).map_err(crate::host_error::provider_error_to_node)
        })
    }

    #[napi(js_name = "flush")]
    pub fn flush(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || self.sink()?.flush().map_err(crate::host_error::provider_error_to_node))
    }

    #[napi(js_name = "endInput")]
    pub fn end_input(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.sink()?.end_input();
            Ok(())
        })
    }

    /// The next item, or `null` at the end of the stream.
    #[napi(js_name = "recv", ts_return_type = "Promise<SynthesizedAudio | null>")]
    pub fn recv<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, Option<crate::generated::records::SynthesizedAudio>>> {
        let source = std::sync::Arc::clone(&self.source);
        env.spawn_future_with_callback(
            async move {
                let mut guard = source.lock().await;
                let Some(src) = guard.as_mut() else { return Ok(Ok(None)) };
                Ok(match src.next().await {
                    Ok(Some(v)) => Ok(Some(crate::generated::records::SynthesizedAudio::from(&v))),
                    Ok(None) => Ok(None),
                    Err(e) => Err(crate::host_error::provider_error_to_node(e)),
                })
            },
            |env, out| out.map_err(|e| e.into_napi(env)),
        )
    }

    #[napi(js_name = "aclose")]
    pub async fn aclose(&self) -> napi::Result<()> {
        if let Some(src) = self.source.lock().await.take() {
            src.close(lk_agents_abi::Ts::from_nanos(u64::MAX)).await;
        }
        Ok(())
    }
}

/// One in-flight completion.
#[napi(js_name = "InferenceLlmStream")]
pub struct InferenceLlmStream {
    pub(crate) source: std::sync::Arc<tokio::sync::Mutex<Option<lk_agents_abi::LlmStream>>>,
}

impl InferenceLlmStream {
    pub fn new(source: lk_agents_abi::LlmStream) -> Self {
        InferenceLlmStream { source: std::sync::Arc::new(tokio::sync::Mutex::new(Some(source))) }
    }
}

#[napi]
impl InferenceLlmStream {
    /// The next item, or `null` at the end of the stream.
    #[napi(js_name = "recv", ts_return_type = "Promise<LlmChunk | null>")]
    pub fn recv<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, Option<crate::generated::records::LlmChunk>>> {
        let source = std::sync::Arc::clone(&self.source);
        env.spawn_future_with_callback(
            async move {
                let mut guard = source.lock().await;
                let Some(src) = guard.as_mut() else { return Ok(Ok(None)) };
                Ok(match src.next().await {
                    Ok(Some(v)) => Ok(Some(crate::generated::records::LlmChunk::from(&v))),
                    Ok(None) => Ok(None),
                    Err(e) => Err(crate::host_error::provider_error_to_node(e)),
                })
            },
            |env, out| out.map_err(|e| e.into_napi(env)),
        )
    }

    #[napi(js_name = "aclose")]
    pub async fn aclose(&self) -> napi::Result<()> {
        if let Some(src) = self.source.lock().await.take() {
            src.close(lk_agents_abi::Ts::from_nanos(u64::MAX)).await;
        }
        Ok(())
    }
}
