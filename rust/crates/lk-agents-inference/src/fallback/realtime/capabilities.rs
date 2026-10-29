//! `llm/realtime_fallback_adapter.py`'s capability merge.

use lk_agents_abi::llm::realtime::RealtimeCapabilities;

/// Which hard capability disagreed, and the two values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityMismatch {
    pub name: &'static str,
    pub first: bool,
    pub other: bool,
}

impl std::fmt::Display for CapabilityMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "all realtime models must agree on `{}` to be used in a RealtimeModelFallbackAdapter, got {} and {}",
            self.name, self.first, self.other
        )
    }
}
impl std::error::Error for CapabilityMismatch {}

/// `_merge_capabilities` (`:70-86`). `audio_output` and `turn_detection` must agree (they shape
/// the pipeline at activity start); the rest are ANDed. The mismatch names the capability and
/// both values, as the reference's `ValueError` does.
pub fn merge_capabilities(
    models: &[RealtimeCapabilities],
) -> Result<RealtimeCapabilities, CapabilityMismatch> {
    let Some(first) = models.first().copied() else {
        return Ok(RealtimeCapabilities::default());
    };
    for c in &models[1..] {
        if c.audio_output != first.audio_output {
            return Err(CapabilityMismatch {
                name: "audio_output",
                first: first.audio_output,
                other: c.audio_output,
            });
        }
        if c.turn_detection != first.turn_detection {
            return Err(CapabilityMismatch {
                name: "turn_detection",
                first: first.turn_detection,
                other: c.turn_detection,
            });
        }
    }
    let all = |f: fn(&RealtimeCapabilities) -> bool| models.iter().all(f);
    Ok(RealtimeCapabilities {
        audio_output: first.audio_output,
        turn_detection: first.turn_detection,
        message_truncation: all(|c| c.message_truncation),
        user_transcription: all(|c| c.user_transcription),
        manual_function_calls: all(|c| c.manual_function_calls),
        auto_tool_reply_generation: all(|c| c.auto_tool_reply_generation),
        mutable_chat_context: all(|c| c.mutable_chat_context),
        mutable_instructions: all(|c| c.mutable_instructions),
        mutable_tools: all(|c| c.mutable_tools),
        per_response_tool_choice: all(|c| c.per_response_tool_choice),
        supports_say: all(|c| c.supports_say),
        can_disable_turn_detection: all(|c| c.can_disable_turn_detection),
        auto_turn_reply_generation: all(|c| c.auto_turn_reply_generation),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(audio_output: bool, supports_say: bool) -> RealtimeCapabilities {
        RealtimeCapabilities {
            audio_output,
            turn_detection: true,
            supports_say,
            ..Default::default()
        }
    }

    // Catches: ANDing a hard capability instead of rejecting a mismatch, a mismatch that does not
    // name the capability and both values, or a soft capability ORed.
    #[test]
    fn hard_capabilities_must_agree_and_soft_ones_are_anded() {
        let merged =
            merge_capabilities(&[caps(true, true), caps(true, false)]).expect("hard caps agree");
        assert!(merged.audio_output);
        assert!(!merged.supports_say, "soft caps are the conservative AND");

        let err = merge_capabilities(&[caps(true, true), caps(false, true)])
            .expect_err("audio_output differs");
        assert_eq!(
            err.to_string(),
            "all realtime models must agree on `audio_output` to be used in a RealtimeModelFallbackAdapter, got true and false"
        );

        let td = RealtimeCapabilities { turn_detection: false, ..caps(true, true) };
        let err = merge_capabilities(&[caps(true, true), td]).expect_err("turn_detection differs");
        assert_eq!(
            err.to_string(),
            "all realtime models must agree on `turn_detection` to be used in a RealtimeModelFallbackAdapter, got true and false"
        );
    }
}
