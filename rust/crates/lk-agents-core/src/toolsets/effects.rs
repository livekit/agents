//! Work produced by [`super::machine::Lifetime`], executed by runtime toolset initialization
//! and cleanup. It represents shared setup/lease/release transitions, not individual tool calls.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolsetEffect {
    Setup(u64),
    CancelSetup,
    Grant(u64),
    Fail(u64),
    Released(u64),
    Close,
}
