//! Shared-core runtime work, grouped by the state owner that produces it.
//!
//! - [`AgentRunEffect`]: invocation decisions; runtime agent-run execution performs the work.
//!   Tool-call effects are converted into invocation variants, not nested payloads.
//! - [`TextSessionEffect`]: backend routing/continuation; its `Run` variant carries AgentRunEffect.
//! - [`ToolCallEffect`]: tool preparation, approval, admission, and execution requests.
//! - [`ToolsetEffect`]: shared toolset setup, grants, releases, and close.
//! - [`A2aTaskEffect`]: durable task progression; `Execute` carries a storage/handler Operation.
//!
//! These are the original domain types, not a root wrapper enum. Ordinary results and private
//! child state stay with their owners. Runtime owns all actual I/O and resource lifetimes.

pub use crate::a2a::task::effects::A2aTaskEffect;
pub use crate::agent_run::effects::AgentRunEffect;
pub use crate::text_session::effects::TextSessionEffect;
pub use crate::tool_calls::effects::ToolCallEffect;
pub use crate::toolsets::effects::ToolsetEffect;
