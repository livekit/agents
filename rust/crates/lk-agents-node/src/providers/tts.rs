//! `NodeTts` -- the JS twin of `PyTts`, and the same provider §1.0 named as the one to build
//! first, because it is the only one with both a simplex and a duplex path.

use std::sync::Arc;

use async_trait::async_trait;
use lk_agents_abi::{Clock, Tts, TtsCapabilities, TtsSink, TtsSource, no_duplex_path};
use lk_agents_abi::errors::ProviderError;

use crate::providers::host::{
    NodeHandle, NodeSink, NodeSource, TtsCallbacks, synthesized_audio_from_js,
};
use lk_agents_abi::utils::frame::FrameTable;

/// Any JavaScript object implementing the `TTS` contract, driven through the Rust trait.
///
/// `FallbackChain` holds this as an `Arc<dyn Tts>` and cannot tell it from a native provider or
/// from a `PyTts`. That is the §1.0 property the whole seam exists for, and it is the reason this
/// file is 120 lines rather than a second pipeline.
pub struct NodeTts {
    handle: Arc<NodeHandle>,
    label: String,
    /// **Read once, from the live object.** Per *instance*, never per class: twelve packages
    /// declare `streaming = false` outright, and six more decide it in the constructor from a
    /// flag. Reading a class-level property would send text to a `stream()` that throws.
    capabilities: TtsCapabilities,
    sample_rate: u32,
    num_channels: u16,
    /// The frame source. §8e's outbound half: the driver holds `current`, the shim reads it, and
    /// no call site here mentions a `FrameId` -- `NodeHandle::open` and the two stream halves take
    /// it, so forgetting it is not expressible.
    frames: Arc<FrameTable>,
    /// The core's clock; every stream's `close` deadline is read against it.
    clock: Arc<dyn Clock>,
}

impl NodeTts {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frames: Arc<FrameTable>,
        cb: TtsCallbacks,
        label: String,
        streaming: bool,
        aligned_transcript: bool,
        sample_rate: u32,
        num_channels: u16,
        clock: Arc<dyn Clock>,
    ) -> Self {
        NodeTts {
            handle: Arc::new(NodeHandle { cb: Arc::new(cb), stream: -1 }),
            label,
            capabilities: TtsCapabilities { streaming, aligned_transcript },
            sample_rate,
            num_channels,
            frames,
            clock,
        }
    }

    /// A handle onto one host stream, sharing the provider's callbacks.
    fn child(&self, stream: i64) -> Arc<NodeHandle> {
        Arc::new(NodeHandle { cb: self.handle.cb.clone(), stream })
    }
}

#[async_trait]
impl Tts for NodeTts {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn capabilities(&self) -> TtsCapabilities {
        self.capabilities
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn num_channels(&self) -> u16 {
        self.num_channels
    }

    async fn synthesize(&self, text: &str) -> Result<TtsSource, ProviderError> {
        let frame = self.frames.current();
        let out = self.handle.open(frame, "synthesize", Some(text.to_owned())).await?;
        if let Some(err) = out.error {
            return Err(err.into());
        }
        let stream = out.stream.ok_or_else(|| {
            ProviderError::from(lk_agents_abi::errors::ApiError::new(
                "the host returned neither a stream nor an error from synthesize()",
            ))
        })?;
        Ok(Box::new(NodeSource::new(
            self.child(stream),
            frame,
            synthesized_audio_from_js,
            Arc::clone(&self.clock),
        )) as TtsSource)
    }

    async fn stream(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
        // Per instance, at runtime. An instance with no duplex path answers `no_duplex_path`;
        // it is wrapped in `StreamAdapter` where it enters the engine, never here.
        if !self.capabilities.streaming {
            return Err(no_duplex_path(&self.label));
        }
        let frame = self.frames.current();
        let out = self.handle.open(frame, "stream", None).await?;
        // The two failures §1.0 refuses to conflate. `noDuplexPath` is the shim telling us the
        // base class raised, which is the same answer as `streaming: false`; anything else is a
        // real failure and must stay loud.
        if out.no_duplex_path {
            if let Some(err) = out.error {
                // Nothing downstream will re-raise it; give the entry back or the shim's error map
                // grows by one per provider that lies about `streaming`.
                crate::errors::release(&err.into());
            }
            return Err(no_duplex_path(&self.label));
        }
        if let Some(err) = out.error {
            return Err(err.into());
        }
        let Some(stream) = out.stream else {
            return Err(lk_agents_abi::errors::ApiError::new(
                "the host returned neither a stream nor an error from stream()",
            )
            .into());
        };

        // Two halves over **one** host stream: the sink pushes text in, the source pulls audio
        // out. Never one object with both -- §1.0's split, for the measured data-loss reason.
        let handle = self.child(stream);
        let sink = NodeSink::new(handle.clone(), frame);
        let source =
            NodeSource::new(handle, frame, synthesized_audio_from_js, Arc::clone(&self.clock));
        Ok((Box::new(sink) as TtsSink, Box::new(source) as TtsSource))
    }
}
