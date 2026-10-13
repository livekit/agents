//! `llm/mcp.py`'s per-tool option records; the merge is `lk_agents_inference::mcp`.

use lk_agents_host_derive::Host;

// The generated module globs this one to resolve field types.
pub use crate::llm::provider_format::tools::{DuplicateMode, DuplicateScope, ToolFlag};

/// A fully resolved set of per-tool options: what `_resolve_tool_options` returns, never a
/// partial dict.
///
/// The emit target is what `resolve_mcp_tool_options` hands back to Python. `flags` crosses as its
/// `bits()` and the two enums as their declared `#[host(rename)]` wire names, so the four keys and
/// the six strings are spelled once, here.
///
/// [`Default`] is `_DEFAULT_TOOL_OPTIONS` (`mcp.py:70`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Host)]
#[binding(emit)]
pub struct McpToolOptions {
    /// Passed through to the `@function_tool` decorator, e.g. `ToolFlag::CANCELLABLE`.
    pub flags: ToolFlag,
    pub on_duplicate: DuplicateMode,
    pub duplicate_scope: DuplicateScope,
    /// Forward the tool's MCP progress notifications to `ctx.update()`. Also selects the
    /// non-blocking call path (`_tool_called_nonblocking`, which takes a `RunContext`).
    pub report_progress: bool,
}

/// Per-tool behavior for MCP tools exposed by an :class:`MCPToolset`.
//
// **The `///` above is the reference's docstring, verbatim** -- the generator renders it as the
// Python class docstring and nothing in the build compares docstrings, so this crate's own notes
// would become the public documentation of a public type without failing anything. Everything
// from here down is `//`, which the generator does not emit. `amd.rs` draws the same line.
//
// This is the caller's `MCPToolOptions(...)` literal: every key independently present or absent,
// which is what makes it the `total=False` twin. `McpToolOptions` above is the *resolved* set
// that `resolve_tool_options` returns, and the two must not be collapsed -- the whole point of
// the merge is that an absent key keeps its default rather than zeroing it.
//
// Every field carries an explicit `ty` because the Rust `Option` here means *absent* while the
// reference's annotations are the plain types; deriving them would publish `ToolFlag | None` for
// a key the reference declares as `ToolFlag`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "mcp",
    path = "livekit.agents.llm.mcp",
    name = "MCPToolOptions"
)]
// **The read target, added after `check-one-sided-declarations.py` grew a helper pass.** These four
// keys were spelled again in `lk-agents-py`'s `options_patch_from_py`, through a `get(k)` closure
// -- so they were invisible to that check until it learned to see helpers, and the resolved twin
// `McpToolOptions` above declared the same four names for the *write* direction. See N338.
//
// `from_items` and a `default` on every field, which is `total=False` said in the generator's
// vocabulary: absent takes `None`, and a present key that is `null` reaches `None` through the
// field's own `Option`. Same arrangement as `MetricsReport`.
#[binding(extract, from_items)]
pub struct McpToolOptionsPatch {
    /// Flags passed to the @function_tool decorator, e.g. ToolFlag.CANCELLABLE.
    #[python(ty = "ToolFlag", import = "..llm.tool_context:ToolFlag")]
    pub flags: Option<ToolFlag>,
    /// Behavior when a tool is called multiple times in the same context.
    #[python(ty = "DuplicateMode", import = "..llm.tool_context:DuplicateMode")]
    pub on_duplicate: Option<DuplicateMode>,
    /// What counts as a duplicate, e.g. ``"name_and_args"`` to compare arguments too.
    #[python(ty = "DuplicateScope", import = "..llm.tool_context:DuplicateScope")]
    pub duplicate_scope: Option<DuplicateScope>,
    /// Whether to forward the tool's progress notifications to ctx.update().
    #[python(ty = "bool")]
    pub report_progress: Option<bool>,
}
