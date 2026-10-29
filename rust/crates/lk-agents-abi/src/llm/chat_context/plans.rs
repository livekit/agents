//! `ChatContext.copy`, `.truncate` and `.merge` as index plans over the host's own items, which
//! the host describes by their facts. Core plans them.

/// `(type, id, role, len(content), name, created_at, call_id)`.
pub type HostFacts = (String, String, Option<String>, usize, Option<String>, f64, Option<String>);

/// A fact the host reported that names no item kind, role or wall clock.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ChatFactsError(pub String);

#[crate::host(functions(pid = false))]
pub trait ChatContextPlans {
    /// `ChatContext.copy` as an index plan over the host's own items.
    #[host(function, name = "chat_ctx_copy_plan")]
    #[allow(clippy::too_many_arguments)]
    fn copy_plan(
        facts: Vec<HostFacts>,
        #[host(kw, default = false)] exclude_function_call: bool,
        #[host(default = false)] exclude_instructions: bool,
        #[host(default = false)] exclude_empty_message: bool,
        #[host(default = false)] exclude_handoff: bool,
        #[host(default = false)] exclude_config_update: bool,
        #[host(default = None)] tools: Option<Vec<String>>,
    ) -> Result<Vec<usize>, ChatFactsError>;

    /// `ChatContext.truncate` as an index plan; `None` when under the limit.
    #[host(function, name = "chat_ctx_truncate_plan")]
    fn truncate_plan(
        facts: Vec<HostFacts>,
        max_items: i64,
    ) -> Result<Option<Vec<usize>>, ChatFactsError>;

    /// `ChatContext.merge` as `(index, j)` insertions.
    #[host(function, name = "chat_ctx_merge_plan")]
    fn merge_plan(
        mine: Vec<HostFacts>,
        theirs: Vec<HostFacts>,
        #[host(kw, default = false)] exclude_function_call: bool,
        #[host(default = false)] exclude_instructions: bool,
        #[host(default = false)] exclude_config_update: bool,
    ) -> Result<Vec<(usize, usize)>, ChatFactsError>;

    /// `remove_instructions(chat_ctx)` as a plan: the positions of the rendered prompt's items,
    /// ascending, given the items' ids.
    #[host(function, name = "remove_instructions")]
    fn instructions_plan(item_ids: Vec<String>) -> Vec<usize>;
}
