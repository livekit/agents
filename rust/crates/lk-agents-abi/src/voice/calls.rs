//! Call schemas: a renamed keyword is a compile error rather than a runtime `TypeError`. Never
//! constructed; the codegen reads them. `#[binding(py)]` fields are `()`: the call site has the value.

use lk_agents_host_derive::Host;

// --- the plugin seam -------------------------------------------------------------------------

/// The inner half of `{"type": "function", "function": {"name": ..}}` -- `llm.NamedToolChoice["function"]`.
/// Both emitted and read.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items)]
pub struct NamedToolChoiceFunction {
    pub name: String,
}

/// `ToolChoice`'s named form, as the two-level dict the providers read -- `llm.NamedToolChoice`.
/// On the read side `"type"` is not read: it has one legal value there.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items, tag = "type", tag_value = "function")]
pub struct NamedToolChoice {
    pub function: NamedToolChoiceFunction,
}

// --- the audio emitter's one constructor -----------------------------------------------------

/// The kwargs `tts.SynthesizedAudio(**kwargs)` is constructed from, as the emitter passes them.
// Four of the seven fields: `started_time` and `timed_transcript` ride in the frame's `userdata`.
// Every field has a default host-side, so a missing key here would be silent.
#[derive(Host)]
#[binding(emit)]
pub struct SynthesizedAudioKwargs {
    /// The `rtc.AudioFrame` the binding built, with `USERDATA_TIMED_TRANSCRIPT` already set on it.
    #[binding(py)]
    pub frame: (),
    pub request_id: String,
    pub segment_id: String,
    pub is_final: bool,
}
