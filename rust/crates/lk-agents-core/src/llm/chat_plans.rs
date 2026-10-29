//! `ChatContext.copy`, `.truncate` and `.merge` planned over the facts the host reports of its own
//! items.

use lk_agents_abi::llm::chat_context::{ChatContextPlans, ChatFactsError, HostFacts};
use lk_agents_abi::{CopyOptions, ItemFacts, MergeOptions, copy_plan, merge_plan, truncate_plan};

pub struct ChatPlans;

fn item_facts(facts: &[HostFacts]) -> Result<Vec<ItemFacts<'_>>, ChatFactsError> {
    facts
        .iter()
        .map(|(kind, id, role, content_len, name, created_at, call_id)| {
            Ok(ItemFacts {
                kind: kind.parse().map_err(ChatFactsError)?,
                role: role.as_deref().map(str::parse).transpose().map_err(ChatFactsError)?,
                content_len: *content_len,
                tool_name: name.as_deref(),
                call_id: call_id.as_deref(),
                id,
                created_at: lk_agents_abi::wall_from_secs(*created_at).ok_or_else(|| {
                    ChatFactsError(format!("created_at={created_at} is not a usable wall clock"))
                })?,
            })
        })
        .collect()
}

impl ChatContextPlans for ChatPlans {
    fn copy_plan(
        facts: Vec<HostFacts>,
        exclude_function_call: bool,
        exclude_instructions: bool,
        exclude_empty_message: bool,
        exclude_handoff: bool,
        exclude_config_update: bool,
        tools: Option<Vec<String>>,
    ) -> Result<Vec<usize>, ChatFactsError> {
        let opts = CopyOptions {
            exclude_function_call,
            exclude_instructions,
            exclude_empty_message,
            exclude_handoff,
            exclude_config_update,
            tools,
        };
        Ok(copy_plan(&item_facts(&facts)?, &opts))
    }

    fn truncate_plan(
        facts: Vec<HostFacts>,
        max_items: i64,
    ) -> Result<Option<Vec<usize>>, ChatFactsError> {
        Ok(truncate_plan(&item_facts(&facts)?, max_items))
    }

    fn merge_plan(
        mine: Vec<HostFacts>,
        theirs: Vec<HostFacts>,
        exclude_function_call: bool,
        exclude_instructions: bool,
        exclude_config_update: bool,
    ) -> Result<Vec<(usize, usize)>, ChatFactsError> {
        let opts =
            MergeOptions { exclude_function_call, exclude_instructions, exclude_config_update };
        Ok(merge_plan(&item_facts(&mine)?, &item_facts(&theirs)?, &opts))
    }
}
