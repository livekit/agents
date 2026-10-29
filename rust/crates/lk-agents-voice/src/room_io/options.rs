//! Resolve the legacy session scalars when no complete room configuration was supplied.

use super::{AudioInputConfig, AudioOutputConfig, RoomIoConfig, TextOutputConfig};
use lk_agents_abi::voice::rtc::TrackPublishOptions;

pub fn resolve_config(
    config: Option<RoomIoConfig>,
    participant_identity: Option<String>,
    input_sample_rate: u32,
    output_sample_rate: u32,
    auto_gain_control: bool,
    text_output: bool,
) -> RoomIoConfig {
    config.unwrap_or_else(|| RoomIoConfig {
        participant_identity: participant_identity.map(Into::into),
        audio_input: Some(AudioInputConfig {
            sample_rate: input_sample_rate,
            auto_gain_control,
            ..AudioInputConfig::default()
        }),
        audio_output: Some(AudioOutputConfig {
            publish: TrackPublishOptions { sample_rate: output_sample_rate, ..Default::default() },
        }),
        text_output: text_output.then(TextOutputConfig::default),
        ..RoomIoConfig::default()
    })
}
