//! TTS PCM emission requests from `machine::AudioEmitter` to the inference emitter executor.

use lk_agents_abi::SynthesizedAudio;
use std::time::Duration;

/// Ordered audio delivery, flush scheduling, and diagnostics requested by the PCM owner.
/// The inference executor performs them after releasing the framing-state lock.
pub enum AudioEmitterEffect {
    EmitAudio(Box<SynthesizedAudio>),
    /// Schedule the held-tail flush after this relative delay; replaces the existing deadline.
    /// The executor supplies its own clock, so no absolute-clock agreement is required.
    ScheduleFlushAfter(Duration),
    CancelScheduledFlush,
    ReportDroppedBytes(usize),
    ReportProviderRequestIds(Vec<String>),
}
