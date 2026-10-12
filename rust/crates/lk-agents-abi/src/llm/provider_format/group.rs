//! `_provider_format/utils.py`: the two rewrites every dialect runs before it formats. Neither clones
//! the context.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use lk_agents_abi::{ChatContext, ChatItem, ChatMessage, FunctionCall, FunctionCallOutput, Role};

use crate::llm::provider_format::FormatError;

/// `utils.py:9`. `{content}` is the only placeholder.
pub const DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE: &str = "<instructions>\n{content}\n</instructions>";

/// The first system/developer message is the preamble; later ones are wrapped in `template` (#5287:
/// otherwise Gemini, Anthropic and Bedrock hoist them). `{content}` only, not full `str.format`.
pub fn convert_mid_conversation_instructions<'a>(
    chat_ctx: &'a ChatContext,
    role: Role,
    template: &str,
) -> Vec<Cow<'a, ChatItem>> {
    let mut first_system_seen = false;
    let mut items: Vec<Cow<'a, ChatItem>> = Vec::with_capacity(chat_ctx.items.len());

    for item in &chat_ctx.items {
        let instruction = item.is_instruction();
        // A truthiness test, as the reference's `(text := item.raw_text_content)`: `Some("")`
        // leaves the item a system message and still marks the preamble seen.
        let text = if instruction && first_system_seen {
            item.as_message().and_then(ChatMessage::raw_text_content).filter(|t| !t.is_empty())
        } else {
            None
        };

        match text {
            Some(text) => {
                let original = item.as_message().expect("is_instruction implies a message");
                items.push(Cow::Owned(ChatItem::Message(ChatMessage::text(
                    original.id.clone(),
                    role,
                    template.replace("{content}", &text),
                    original.created_at,
                ))));
            }
            None => {
                if instruction {
                    first_system_seen = true;
                }
                items.push(Cow::Borrowed(item));
            }
        }
    }

    items
}

/// One assistant turn: at most one message, its parallel tool calls, and the matching results.
#[derive(Debug, Default, Clone)]
pub struct ChatItemGroup<'a> {
    pub message: Option<&'a ChatMessage>,
    pub tool_calls: Vec<&'a FunctionCall>,
    pub tool_outputs: Vec<&'a FunctionCallOutput>,
}

#[derive(Debug, Clone, Copy)]
pub enum GroupItem<'a> {
    Message(&'a ChatMessage),
    Call(&'a FunctionCall),
    Output(&'a FunctionCallOutput),
}

impl<'a> ChatItemGroup<'a> {
    /// `utils.py:161`: message, then calls, then outputs, as Anthropic, Bedrock and Gemini require.
    pub fn flatten(&self) -> impl Iterator<Item = GroupItem<'a>> + '_ {
        self.message
            .map(GroupItem::Message)
            .into_iter()
            .chain(self.tool_calls.iter().copied().map(GroupItem::Call))
            .chain(self.tool_outputs.iter().copied().map(GroupItem::Output))
    }

    /// Empty for an `agent_handoff`/`agent_config_update` key: the reference drops those items.
    pub fn is_empty(&self) -> bool {
        self.message.is_none() && self.tool_calls.is_empty() && self.tool_outputs.is_empty()
    }
}

/// The binding logs the reference's warnings (`utils.py:99`, `:140`, `:152`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dropped {
    pub reason: DropReason,
    pub call_id: String,
    pub tool_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropReason {
    /// "function output missing the corresponding function call, ignoring"
    OutputWithoutCall,
    /// "function call missing the corresponding function output, ignoring"
    CallWithoutOutput,
}

#[derive(Debug, Default, Clone)]
pub struct Grouping<'a> {
    pub groups: Vec<ChatItemGroup<'a>>,
    pub dropped: Vec<Dropped>,
}

/// `utils.py:54`. Messages and calls group by `group_id` when set (#4170), else by the `id` before
/// the first `/`; outputs attach by `call_id`.
pub fn group_tool_calls<'a>(
    items: impl IntoIterator<Item = &'a ChatItem>,
) -> Result<Grouping<'a>, FormatError> {
    let mut keys: Vec<ChatItemGroup<'a>> = Vec::new();
    // `item_groups` is an `OrderedDict`: re-assigning keeps position, and a non-groupable item
    // whose `id` collides replaces that group. An index over a `Vec` reproduces both.
    let mut index: HashMap<&'a str, usize> = HashMap::new();
    let mut tool_outputs: Vec<&'a FunctionCallOutput> = Vec::new();
    let mut dropped: Vec<Dropped> = Vec::new();

    for item in items {
        match item {
            ChatItem::Message(m) if m.role == Role::Assistant => {
                // The `/`-prefix key puts `item_7` with `item_7/0`, `item_7/1`.
                let group_id = m.id.split('/').next().unwrap_or(&m.id);
                let slot = upsert(&mut keys, &mut index, group_id);
                if slot.message.is_some() {
                    // `utils.py:117`'s assert; reachable when two messages share a `/`-prefix.
                    return Err(FormatError::MultipleMessagesInGroup {
                        group_id: group_id.to_string(),
                    });
                }
                slot.message = Some(m);
            }
            ChatItem::FunctionCall(f) => {
                let group_id = match &f.group_id {
                    Some(g) if !g.is_empty() => g.as_str(),
                    _ => f.id.split('/').next().unwrap_or(&f.id),
                };
                upsert(&mut keys, &mut index, group_id).tool_calls.push(f);
            }
            ChatItem::FunctionCallOutput(o) => tool_outputs.push(o),
            other => {
                // An assignment: a user message reusing a group's key wipes that group.
                let id = other.id();
                let slot = upsert(&mut keys, &mut index, id);
                *slot = ChatItemGroup::default();
                if let ChatItem::Message(m) = other {
                    slot.message = Some(m);
                }
            }
        }
    }

    // One flat `call_id -> group` map first, so with a shared `call_id` the last group wins.
    let mut call_id_to_group: HashMap<&str, usize> = HashMap::new();
    for (i, group) in keys.iter().enumerate() {
        for call in &group.tool_calls {
            call_id_to_group.insert(call.call_id.as_str(), i);
        }
    }
    for output in tool_outputs {
        match call_id_to_group.get(output.call_id.as_str()) {
            Some(&i) => keys[i].tool_outputs.push(output),
            None => dropped.push(Dropped {
                reason: DropReason::OutputWithoutCall,
                call_id: output.call_id.clone(),
                tool_name: output.name.clone(),
            }),
        }
    }

    for group in &mut keys {
        remove_invalid_tool_calls(group, &mut dropped);
    }

    Ok(Grouping { groups: keys, dropped })
}

fn upsert<'g, 'a>(
    keys: &'g mut Vec<ChatItemGroup<'a>>,
    index: &mut HashMap<&'a str, usize>,
    id: &'a str,
) -> &'g mut ChatItemGroup<'a> {
    let i = *index.entry(id).or_insert_with(|| {
        keys.push(ChatItemGroup::default());
        keys.len() - 1
    });
    &mut keys[i]
}

/// `utils.py:126`. Keep the equal-length early return: a balanced but mismatched group is left as
/// it is.
fn remove_invalid_tool_calls(group: &mut ChatItemGroup<'_>, dropped: &mut Vec<Dropped>) {
    if group.tool_calls.len() == group.tool_outputs.len() {
        return;
    }

    let call_ids: HashSet<&str> = group.tool_calls.iter().map(|c| c.call_id.as_str()).collect();
    let output_ids: HashSet<&str> = group.tool_outputs.iter().map(|o| o.call_id.as_str()).collect();

    group.tool_calls.retain(|c| {
        let keep = output_ids.contains(c.call_id.as_str());
        if !keep {
            dropped.push(Dropped {
                reason: DropReason::CallWithoutOutput,
                call_id: c.call_id.clone(),
                tool_name: c.name.clone(),
            });
        }
        keep
    });
    group.tool_outputs.retain(|o| {
        let keep = call_ids.contains(o.call_id.as_str());
        if !keep {
            dropped.push(Dropped {
                reason: DropReason::OutputWithoutCall,
                call_id: o.call_id.clone(),
                tool_name: o.name.clone(),
            });
        }
        keep
    });
}

#[cfg(test)]
mod tests {
    use lk_agents_abi::ChatContext;

    use super::*;
    use crate::llm::provider_format::testkit::{call, ctx, handoff, msg, msg_with, out, text_content};

    fn convert(c: &ChatContext) -> Vec<Cow<'_, ChatItem>> {
        convert_mid_conversation_instructions(c, Role::User, DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE)
    }

    fn roles_and_text(items: &[Cow<'_, ChatItem>]) -> Vec<(String, Option<String>)> {
        items
            .iter()
            .map(|i| match i.as_ref() {
                ChatItem::Message(m) => (m.role.as_str().to_string(), m.raw_text_content()),
                other => (other.kind().to_string(), None),
            })
            .collect()
    }

    // Catches: reading `(text := item.raw_text_content)` as a presence test; an empty later system
    // message would become a literal `<instructions>\n\n</instructions>` user turn.
    #[test]
    fn an_empty_later_system_message_is_left_alone_not_wrapped_in_the_template() {
        let c = ctx(vec![
            msg("s0", Role::System, &["preamble"]),
            msg("s1", Role::System, &[""]),
            msg("s2", Role::Developer, &[]),
        ]);
        assert_eq!(
            roles_and_text(&convert(&c)),
            [
                ("system".into(), Some("preamble".into())),
                ("system".into(), Some("".into())),
                ("developer".into(), None),
            ]
        );
    }

    // Catches: rewriting the first instruction too, or hoisting later ones out of position (#5287).
    #[test]
    fn the_first_instruction_is_the_preamble_and_later_ones_move_to_user_in_place() {
        let c = ctx(vec![
            msg("s0", Role::Developer, &["preamble"]),
            msg("u1", Role::User, &["hi"]),
            msg("s2", Role::System, &["mid"]),
        ]);
        assert_eq!(
            roles_and_text(&convert(&c)),
            [
                ("developer".into(), Some("preamble".into())),
                ("user".into(), Some("hi".into())),
                ("user".into(), Some("<instructions>\nmid\n</instructions>".into())),
            ]
        );
    }

    // Catches: setting `first_system_seen` only when the message had text.
    #[test]
    fn a_textless_first_instruction_still_consumes_the_preamble_slot() {
        let c =
            ctx(vec![msg("s0", Role::System, &[]), msg("s1", Role::System, &["real preamble"])]);
        assert_eq!(
            roles_and_text(&convert(&c)),
            [
                ("system".into(), None),
                ("user".into(), Some("<instructions>\nreal preamble\n</instructions>".into())),
            ]
        );
    }

    // Catches: re-substituting `{content}` inside the substituted text.
    #[test]
    fn the_template_substitutes_content_exactly_once() {
        let c = ctx(vec![msg("s0", Role::System, &["p"]), msg("s1", Role::System, &["{content}"])]);
        let items = convert(&c);
        assert_eq!(
            items[1].as_message().unwrap().raw_text_content().unwrap(),
            "<instructions>\n{content}\n</instructions>"
        );
    }

    fn group_ids<'a>(g: &Grouping<'a>) -> Vec<(Option<&'a str>, Vec<&'a str>, Vec<&'a str>)> {
        g.groups
            .iter()
            .map(|grp| {
                (
                    grp.message.map(|m| m.id.as_str()),
                    grp.tool_calls.iter().map(|c| c.call_id.as_str()).collect(),
                    grp.tool_outputs.iter().map(|o| o.call_id.as_str()).collect(),
                )
            })
            .collect()
    }

    // Catches: keying assistant messages by their full id instead of the `/`-prefix.
    #[test]
    fn a_message_and_its_slash_suffixed_calls_land_in_one_group() {
        let items = vec![
            msg("item_7", Role::Assistant, &["thinking"]),
            call("item_7/0", "c0", "a", "{}"),
            call("item_7/1", "c1", "b", "{}"),
            out("item_8", "c0", "a", "ok"),
            out("item_9", "c1", "b", "ok"),
        ];
        let g = group_tool_calls(&items).unwrap();
        assert_eq!(group_ids(&g), [(Some("item_7"), vec!["c0", "c1"], vec!["c0", "c1"])]);
    }

    // Catches: preferring the id prefix over `group_id`, or treating an empty `group_id` as set.
    #[test]
    fn group_id_beats_the_id_prefix_but_an_empty_group_id_does_not() {
        let mut a = call("x/0", "c0", "a", "{}");
        let mut b = call("y/0", "c1", "b", "{}");
        let mut c = call("z/0", "c2", "c", "{}");
        if let ChatItem::FunctionCall(f) = &mut a {
            f.group_id = Some("g".into());
        }
        if let ChatItem::FunctionCall(f) = &mut b {
            f.group_id = Some("g".into());
        }
        if let ChatItem::FunctionCall(f) = &mut c {
            f.group_id = Some(String::new());
        }
        // `d` shares the `z` prefix with `c`, so treating `""` as a group_id splits them.
        let d = call("z/1", "c3", "d", "{}");
        let items = vec![
            a,
            b,
            c,
            d,
            out("o0", "c0", "a", "k"),
            out("o1", "c1", "b", "k"),
            out("o2", "c2", "c", "k"),
            out("o3", "c3", "d", "k"),
        ];
        let g = group_tool_calls(&items).unwrap();
        assert_eq!(
            group_ids(&g),
            [
                (None, vec!["c0", "c1"], vec!["c0", "c1"]),
                (None, vec!["c2", "c3"], vec!["c2", "c3"]),
            ],
            "empty group_id must fall back to the id prefix"
        );
    }

    // Catches: `item_groups` as a HashMap, or appending on re-assignment.
    #[test]
    fn a_colliding_user_message_replaces_the_group_in_place() {
        let items = vec![
            msg("dup", Role::Assistant, &["first"]),
            msg("later", Role::User, &["b"]),
            msg("dup", Role::User, &["second"]),
        ];
        let g = group_tool_calls(&items).unwrap();
        let order: Vec<_> =
            g.groups.iter().map(|grp| grp.message.map(|m| m.raw_text_content().unwrap())).collect();
        assert_eq!(order, [Some("second".to_string()), Some("b".to_string())]);
    }

    // Catches: dropping the equal-length early return and always intersecting.
    #[test]
    fn a_balanced_group_with_mismatched_ids_is_left_untouched() {
        let items = vec![
            call("g", "c0", "a", "{}"),
            call("g/1", "c1", "b", "{}"),
            out("o0", "zzz", "a", "ok"),
            out("o1", "c0", "a", "ok"),
        ];
        // "zzz" is dropped and "c0" attaches: 2 calls, 1 output, so the filter runs.
        let g = group_tool_calls(&items).unwrap();
        assert_eq!(group_ids(&g), [(None, vec!["c0"], vec!["c0"])]);
        assert_eq!(g.dropped.len(), 2, "the orphan output and the unanswered call");

        // Now the balanced-but-mismatched case the early return protects.
        let items = vec![
            call("g", "c0", "a", "{}"),
            call("g/1", "c1", "b", "{}"),
            out("o0", "c0", "a", "ok"),
            out("o1", "c0", "a", "ok"),
        ];
        let g = group_tool_calls(&items).unwrap();
        assert_eq!(group_ids(&g), [(None, vec!["c0", "c1"], vec!["c0", "c0"])]);
        assert!(g.dropped.is_empty(), "the early return must skip the filter entirely");
    }

    // Catches: dropping an orphan silently, or reporting the wrong reason/id.
    #[test]
    fn orphans_are_dropped_and_reported_with_their_call_id_and_tool_name() {
        let items = vec![
            call("a", "answered", "lookup", "{}"),
            out("b", "answered", "lookup", "ok"),
            out("c", "ghost", "search", "ok"),
            call("d", "unanswered", "fetch", "{}"),
        ];
        let g = group_tool_calls(&items).unwrap();
        assert_eq!(
            g.dropped,
            [
                Dropped {
                    reason: DropReason::OutputWithoutCall,
                    call_id: "ghost".into(),
                    tool_name: "search".into(),
                },
                Dropped {
                    reason: DropReason::CallWithoutOutput,
                    call_id: "unanswered".into(),
                    tool_name: "fetch".into(),
                },
            ]
        );
    }

    // Catches: `flatten` yielding context order rather than message/calls/outputs.
    #[test]
    fn flatten_regroups_interleaved_calls_and_outputs() {
        let items = vec![
            msg("t", Role::Assistant, &["m"]),
            call("t/0", "c0", "a", "{}"),
            out("o0", "c0", "a", "ok"),
            call("t/1", "c1", "b", "{}"),
            out("o1", "c1", "b", "ok"),
        ];
        let g = group_tool_calls(&items).unwrap();
        let kinds: Vec<&str> = g.groups[0]
            .flatten()
            .map(|i| match i {
                GroupItem::Message(_) => "m",
                GroupItem::Call(_) => "c",
                GroupItem::Output(_) => "o",
            })
            .collect();
        assert_eq!(kinds, ["m", "c", "c", "o", "o"]);
    }

    // Catches: giving `agent_handoff`/`agent_config_update` a group member.
    #[test]
    fn a_handoff_produces_an_empty_group_and_reaches_no_provider() {
        let items = vec![handoff("h"), msg("m", Role::User, &["hi"])];
        let g = group_tool_calls(&items).unwrap();
        assert_eq!(g.groups.len(), 2);
        assert!(g.groups[0].is_empty(), "the handoff group must be empty, not absent");
        assert!(!g.groups[1].is_empty());
    }

    // Catches: silently keeping one of two colliding assistant messages instead of erroring.
    #[test]
    fn two_assistant_messages_in_one_group_is_an_error_not_a_silent_drop() {
        let items = vec![
            msg("shared", Role::Assistant, &["one"]),
            msg("shared/2", Role::Assistant, &["two"]),
        ];
        assert_eq!(
            group_tool_calls(&items).err(),
            Some(FormatError::MultipleMessagesInGroup { group_id: "shared".into() })
        );
    }

    // Catches: grouping user messages the way assistant ones are grouped.
    #[test]
    fn only_assistant_messages_join_a_tool_call_group() {
        let items =
            vec![msg("shared", Role::User, &["one"]), msg("shared/2", Role::User, &["two"])];
        let g = group_tool_calls(&items).unwrap();
        assert_eq!(g.groups.len(), 2);
    }

    // Catches: an image-only system message being rewritten, or not taking the preamble slot.
    #[test]
    fn an_image_only_instruction_is_not_rewritten() {
        let c = ctx(vec![
            msg("s0", Role::System, &["p"]),
            msg_with("s1", Role::System, vec![text_content("")]),
        ]);
        assert_eq!(roles_and_text(&convert(&c))[1].0, "system");
    }
}
