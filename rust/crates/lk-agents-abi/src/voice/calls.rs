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
