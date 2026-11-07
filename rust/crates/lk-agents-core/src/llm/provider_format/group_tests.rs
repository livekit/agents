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

// Catches: keeping a textless later instruction, which would shadow the preamble.
#[test]
fn an_empty_later_system_message_is_dropped_not_wrapped_in_the_template() {
    let c = ctx(vec![
        msg("s0", Role::System, &["preamble"]),
        msg("s1", Role::System, &[""]),
        msg("s2", Role::Developer, &[]),
    ]);
    assert_eq!(roles_and_text(&convert(&c)), [("system".into(), Some("preamble".into()))]);
}

// Catches: a first instruction after the conversation began kept as the preamble (the
// per-turn instructions `generate_reply` appends on the first turn); a handoff does not count.
#[test]
fn an_instruction_after_conversation_content_is_never_the_preamble() {
    let c = ctx(vec![msg("u0", Role::User, &["hi"]), msg("s1", Role::System, &["reply"])]);
    assert_eq!(
        roles_and_text(&convert(&c)),
        [
            ("user".into(), Some("hi".into())),
            ("user".into(), Some("<instructions>\nreply\n</instructions>".into())),
        ]
    );
    let c = ctx(vec![handoff("h"), msg("s1", Role::System, &["preamble"])]);
    assert_eq!(roles_and_text(&convert(&c))[1], ("system".into(), Some("preamble".into())));
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
    let c = ctx(vec![msg("s0", Role::System, &[]), msg("s1", Role::System, &["real preamble"])]);
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
        [(None, vec!["c0", "c1"], vec!["c0", "c1"]), (None, vec!["c2", "c3"], vec!["c2", "c3"]),],
        "empty group_id must fall back to the id prefix"
    );
}

// Catches: `item_groups` as a BTreeMap, or appending on re-assignment.
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
    let items =
        vec![msg("shared", Role::Assistant, &["one"]), msg("shared/2", Role::Assistant, &["two"])];
    assert_eq!(
        group_tool_calls(&items).err(),
        Some(FormatError::MultipleMessagesInGroup { group_id: "shared".into() })
    );
}

// Catches: grouping user messages the way assistant ones are grouped.
#[test]
fn only_assistant_messages_join_a_tool_call_group() {
    let items = vec![msg("shared", Role::User, &["one"]), msg("shared/2", Role::User, &["two"])];
    let g = group_tool_calls(&items).unwrap();
    assert_eq!(g.groups.len(), 2);
}

// Catches: a textless later instruction rewritten into an empty user turn.
#[test]
fn a_textless_later_instruction_is_dropped() {
    let c = ctx(vec![
        msg("s0", Role::System, &["p"]),
        msg_with("s1", Role::System, vec![text_content("")]),
    ]);
    assert_eq!(convert(&c).len(), 1);
}
