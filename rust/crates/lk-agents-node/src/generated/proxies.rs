//! GENERATED FILE -- DO NOT EDIT. Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`); see `napi.rs` in `lk-agents-codegen`.

#![allow(clippy::all, dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]


pub mod host_realtime_audio_node {
    #![allow(unused_imports)]
    use lk_agents_abi::AudioFrame;
    use lk_agents_voice::io::*;

    /// `lk_agents_voice::io::RealtimeAudioNode` over a JS object (`#[host(proxy)]` in `rust/crates/lk-agents-voice/src/io.rs`).
    pub struct HostRealtimeAudioNode {
        pub(crate) bridge: crate::bridge::NodeBridge,
    }

    impl HostRealtimeAudioNode {
        pub fn from_bridge(bridge: crate::bridge::NodeBridge) -> Self {
            HostRealtimeAudioNode { bridge }
        }
    }

    #[async_trait::async_trait]
    impl lk_agents_voice::io::RealtimeAudioNode for HostRealtimeAudioNode {
        async fn push(&self, frame: &lk_agents_abi::AudioFrame) {
            let _ = self.bridge.soon("push", vec![crate::bridge::JsArg::new(crate::providers::host::audio_frame_to_js(frame))]);
        }

        fn end(&self) {
            let _ = self.bridge.soon("end", vec![]);
        }
    }
}
pub use host_realtime_audio_node::HostRealtimeAudioNode;
