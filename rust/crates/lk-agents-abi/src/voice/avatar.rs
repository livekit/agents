//! What an avatar's audio receiver tells its host.

/// The `clear_buffer` event `DataStreamAudioReceiver` emits when the agent clears its audio
/// (`AudioReceiver.emit`).
#[crate::host(proxy(host = "livekit.agents.voice.avatar.DataStreamAudioReceiver"))]
pub trait ClearBufferListener: Send + Sync {
    #[host(notify, name = "_core_clear_buffer")]
    fn clear_buffer(&self);
}
