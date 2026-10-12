//! `ChatContext._summarize` (`llm/chat_context.py:749`) and `to_xml`, split at the `llm.chat` await.
//! Summarises `text_content` (markup stripped). `to_xml` escapes nothing, deliberately.

use std::time::{Duration, SystemTime};

use lk_agents_abi::{ChatItem, Role};
use serde_json::Value;


/// [`to_xml`] over owned pairs; `attrs` is a list of pairs because `_summarize`'s two renderings
/// order `name` and `call_id` differently.
#[crate::host(export(pid = false, name = "to_xml"))]
pub fn to_xml_of(tag_name: &str, content: Option<&str>, attrs: Option<Vec<(String, String)>>) -> String {
    let attrs = attrs.unwrap_or_default();
    to_xml(tag_name, content, &attrs.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect::<Vec<_>>())
}

/// `chat_context.py:1100`. `if content:` is truthiness: an empty string renders self-closing.
pub fn to_xml(tag_name: &str, content: Option<&str>, attrs: &[(&str, &str)]) -> String {
    let attrs_str = if attrs.is_empty() {
        None
    } else {
        Some(
            attrs
                .iter()
                .map(|(k, v)| format!("{k}=\"{v}\""))
                .collect::<Vec<_>>()
                .join(" "),
        )
    };

    match content.filter(|c| !c.is_empty()) {
        Some(content) => {
            let open = match &attrs_str {
                Some(a) => format!("<{tag_name} {a}>"),
                None => format!("<{tag_name}>"),
            };
            format!("{open}\n{content}\n</{tag_name}>")
        }
        None => match &attrs_str {
            Some(a) => format!("<{tag_name} {a} />"),
            None => format!("<{tag_name} />"),
        },
    }
}

// The summarisation prompt (`chat_context.py:814`), copied byte for byte: the runs of spaces left
// by the reference's line continuations are part of the prompt.
const SUMMARY_SYSTEM_PROMPT: &str = r#"Compress older conversation history into a short, faithful summary.

The conversation is formatted as XML. Here is how to read it:
- <user>…</user>  — something the user said.
- <assistant>…</assistant>  — something the assistant said.
- <function_call name="…" call_id="…">…</function_call>  — the assistant invoked an action.
- <function_call_output name="…" call_id="…">…</function_call_output>  — the result of that                 action. May contain <error>…</error> if it failed.

Guidelines:
- Distill the *information learned* from function call outputs into the summary.                 Do not mention that a tool/function was called — just preserve the knowledge gained.
- Focus on: user goals, constraints, decisions, key facts, preferences, entities,                 and any pending or unresolved tasks.
- Omit greetings, filler, and chit-chat.
- Be concise."#;

const SUMMARY_USER_PREFIX: &str = "Conversation to summarize:\n\n";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummarizePlan {
    /// The tail: kept verbatim, after the summary.
    pub split_idx: usize,
    /// A prior summary is dropped too: the reference's comment says otherwise but its code does
    /// this, and the code is matched.
    pub preserved: Vec<usize>,
    pub system_prompt: &'static str,
    pub user_content: String,
    /// A microsecond below the first tail item (above the last head item when the tail is empty),
    /// so `find_insertion_index`'s `<=` never ties.
    pub created_at: SystemTime,
}

/// The `assistant` message body the summary is stored as.
#[crate::host(export(pid = false, name = "chat_history_summary_xml"))]
pub fn summary_content(summary: &str) -> String {
    to_xml("chat_history_summary", Some(summary), &[])
}

/// The attribute order differs between the two, as in the reference; harmonising it is a prompt
/// change.
fn call_item_xml(item: &ChatItem) -> Option<String> {
    match item {
        ChatItem::FunctionCall(f) => Some(to_xml(
            "function_call",
            Some(&f.arguments),
            &[("name", &f.name), ("call_id", &f.call_id)],
        )),
        ChatItem::FunctionCallOutput(f) => {
            // Wrapped again so the summariser can tell a failed tool from one that returned "error".
            let inner =
                if f.is_error { to_xml("error", Some(&f.output), &[]) } else { f.output.clone() };
            Some(to_xml(
                "function_call_output",
                Some(&inner),
                &[("call_id", &f.call_id), ("name", &f.name)],
            ))
        }
        _ => None,
    }
}

fn is_summary(item: &ChatItem) -> bool {
    match item {
        // `is True`: an identity test, so `1` is not a summary and neither is `"yes"`.
        ChatItem::Message(m) => m.extra.get("is_summary") == Some(&Value::Bool(true)),
        _ => false,
    }
}

fn is_summarizable(item: &ChatItem) -> bool {
    match item {
        ChatItem::Message(m) => matches!(m.role, Role::User | Role::Assistant),
        ChatItem::FunctionCall(_) | ChatItem::FunctionCallOutput(_) => true,
        _ => false,
    }
}

/// `_summarize`'s first half. `None` means do nothing (the reference's four `return self` paths);
/// `keep_last_turns = 0` summarises everything.
/// `_summarize`'s decisions up to the `llm.chat` await, for the host:
/// `(split_idx, preserved, system_prompt, user_content, created_at)`; `None` means do nothing.
#[crate::host(export(pid = false, name = "chat_ctx_summarize_plan"))]
pub fn summarize_plan_of(items: Vec<ChatItem>, keep_last_turns: usize) -> Option<(usize, Vec<usize>, &'static str, String, SystemTime)> {
    summarize_plan(&items, keep_last_turns).map(|p| (p.split_idx, p.preserved, p.system_prompt, p.user_content, p.created_at))
}

pub fn summarize_plan(items: &[ChatItem], keep_last_turns: usize) -> Option<SummarizePlan> {
    let msg_budget = keep_last_turns * 2;
    let mut split_idx = items.len();

    if msg_budget > 0 {
        let mut msg_count = 0;
        let mut filled = false;
        for i in (0..items.len()).rev() {
            if let ChatItem::Message(m) = &items[i] {
                if matches!(m.role, Role::User | Role::Assistant) {
                    msg_count += 1;
                    if msg_count >= msg_budget {
                        split_idx = i;
                        filled = true;
                        break;
                    }
                }
            }
        }
        // `for ... else`: shorter than the budget means nothing to summarise.
        if !filled {
            return None;
        }
    }

    if split_idx == 0 {
        return None;
    }

    let (head, tail) = items.split_at(split_idx);

    let mut contents: Vec<String> = Vec::new();
    for item in head {
        match item {
            ChatItem::Message(m) => {
                if !matches!(m.role, Role::User | Role::Assistant) {
                    continue;
                }
                if is_summary(item) {
                    continue; // no summaries of summaries
                }
                // `text_content`, not `raw_text_content`: expressive markup out.
                let text = m.text_content().unwrap_or_default();
                let text = text.trim();
                if text.is_empty() {
                    continue;
                }
                let tag = match m.role {
                    Role::User => "user",
                    _ => "assistant",
                };
                contents.push(to_xml(tag, Some(text), &[]));
            }
            _ => {
                if let Some(rendered) = call_item_xml(item) {
                    contents.push(rendered);
                }
            }
        }
    }

    if contents.is_empty() {
        return None;
    }
    let source_text = contents.join("\n");
    let source_text = source_text.trim();
    if source_text.is_empty() {
        return None;
    }

    let preserved: Vec<usize> =
        (0..split_idx).filter(|&i| !is_summarizable(&items[i])).collect();

    const NUDGE: Duration = Duration::from_micros(1);
    let created_at = match tail.first() {
        Some(first) => first.created_at().checked_sub(NUDGE).unwrap_or_else(|| first.created_at()),
        // `head` is non-empty here: `split_idx == 0` already returned.
        None => head[head.len() - 1].created_at() + NUDGE,
    };

    Some(SummarizePlan {
        split_idx,
        preserved,
        system_prompt: SUMMARY_SYSTEM_PROMPT,
        user_content: format!("{SUMMARY_USER_PREFIX}{source_text}"),
        created_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{call, msg, t};
    use lk_agents_abi::FunctionCallOutput;
    use serde_json::json;

    fn summary_msg(id: &str, text: &str, secs: u64) -> ChatItem {
        let mut item = msg(id, Role::Assistant, text, secs);
        if let ChatItem::Message(m) = &mut item {
            m.extra.insert("is_summary".into(), json!(true));
        }
        item
    }

    fn turns(n: u64) -> Vec<ChatItem> {
        (0..n)
            .flat_map(|i| {
                [
                    msg(&format!("u{i}"), Role::User, &format!("q{i}"), i * 2),
                    msg(&format!("a{i}"), Role::Assistant, &format!("r{i}"), i * 2 + 1),
                ]
            })
            .collect()
    }

    // Catches: summarising a conversation shorter than the budget (the `for ... else` path).
    #[test]
    fn a_conversation_shorter_than_the_budget_is_left_alone() {
        // One turn, budget four messages: the walk runs out and the `else` fires.
        assert!(summarize_plan(&turns(1), 2).is_none());
        // Two turns: split at zero, the second guard.
        assert!(summarize_plan(&turns(2), 2).is_none());
        assert!(summarize_plan(&turns(3), 2).is_some());
    }

    // Catches: counting items rather than user/assistant messages toward the budget.
    #[test]
    fn only_user_and_assistant_messages_consume_the_budget() {
        let mut items = turns(3);
        items.insert(1, call("f", "tool", 4));
        let plan = summarize_plan(&items, 2).unwrap();
        // Counting items would have stopped one message early, at index 4.
        assert_eq!(plan.split_idx, 3);
        assert_eq!(items[plan.split_idx].id(), "u1");
    }

    // Catches: reading `is_summary` with truthiness or `== 1`.
    #[test]
    fn a_prior_summary_is_never_summarised_again() {
        let mut items = turns(3);
        items[0] = summary_msg("s0", "earlier history", 0);
        let plan = summarize_plan(&items, 2).unwrap();
        assert!(!plan.user_content.contains("earlier history"));
        // ...and dropped from the rebuilt history, as the reference's code does.
        assert_eq!(plan.preserved, Vec::<usize>::new());
    }

    // Catches: dropping the structural head items along with the summarizable ones.
    #[test]
    fn system_messages_survive_the_rewrite_and_are_not_summarised() {
        let mut items = turns(3);
        items.insert(0, msg("sys", Role::System, "you are helpful", 0));
        let plan = summarize_plan(&items, 2).unwrap();
        assert_eq!(plan.preserved, vec![0]);
        assert!(!plan.user_content.contains("you are helpful"));
    }

    // Catches: a `created_at` tie with the first tail item (`find_insertion_index` uses `<=`).
    #[test]
    fn the_summary_is_stamped_strictly_before_the_tail() {
        let items = turns(3);
        let plan = summarize_plan(&items, 2).unwrap();
        assert!(plan.created_at < items[plan.split_idx].created_at());
    }

    // Catches: the empty-tail direction of the same nudge.
    #[test]
    fn with_no_tail_the_summary_is_stamped_after_the_last_head_item() {
        let items = turns(3);
        let plan = summarize_plan(&items, 0).unwrap();
        assert_eq!(plan.split_idx, items.len());
        assert!(plan.created_at > items[items.len() - 1].created_at());
    }

    // Catches: harmonising the two attribute orders, or dropping the `<error>` wrapper.
    #[test]
    fn call_and_output_render_with_their_own_attribute_orders() {
        assert_eq!(
            call_item_xml(&call("c0", "tool", 0)).unwrap(),
            "<function_call name=\"tool\" call_id=\"call_c0\">\n{}\n</function_call>"
        );

        let out = ChatItem::FunctionCallOutput(FunctionCallOutput {
            id: "o".into(),
            name: "tool".into(),
            call_id: "c0".into(),
            output: "boom".into(),
            is_error: true,
            created_at: t(0),
            reply_required: true,
        });
        assert_eq!(
            call_item_xml(&out).unwrap(),
            "<function_call_output call_id=\"c0\" name=\"tool\">\n<error>\nboom\n</error>\n</function_call_output>"
        );
    }

    // Catches: rendering empty content as `<tag></tag>`.
    #[test]
    fn empty_content_is_self_closing() {
        assert_eq!(to_xml("t", Some(""), &[]), "<t />");
        assert_eq!(to_xml("t", None, &[("k", "v")]), "<t k=\"v\" />");
    }

    // Catches: summarising expressive markup (assistant `text_content` strips `<expr/>`).
    #[test]
    fn assistant_markup_is_stripped_from_the_summarisation_input() {
        let mut items = turns(3);
        items[1] = msg("a0", Role::Assistant, "<expr type=\"mood\" label=\"happy\"/>hello", 1);
        let plan = summarize_plan(&items, 2).unwrap();
        assert!(!plan.user_content.contains("<expr"));
        assert!(plan.user_content.contains("hello"));
    }
}
