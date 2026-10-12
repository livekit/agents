//! `ChatContext` and its index plans: copy, truncate and merge, over owned items and host facts.

use super::*;
use crate::llm::chat_context::{ChatContent, ChatMessage, Role};
use crate::testkit::{call, config_update, empty_msg, handoff, ids, msg, out, t, text};

fn ctx(items: Vec<ChatItem>) -> ChatContext {
    ChatContext::new(items)
}

fn facts(c: &ChatContext) -> Vec<ItemFacts<'_>> {
    c.items.iter().map(ItemFacts::of).collect()
}

/// The ids `truncate(max_items)` leaves, through [`truncate_plan`].
fn truncated(c: &ChatContext, max_items: i64) -> Vec<&str> {
    match truncate_plan(&facts(c), max_items) {
        Some(plan) => plan.into_iter().map(|i| c.items[i].id()).collect(),
        None => ids(c),
    }
}

/// `parent.merge(child)` through [`merge_plan`], applied the way the host applies it.
fn merged(parent: &ChatContext, child: &ChatContext, opts: &MergeOptions) -> ChatContext {
    let mut out = parent.clone();
    for (idx, j) in merge_plan(&facts(parent), &facts(child), opts) {
        out.items.insert(idx, child.items[j].clone());
    }
    out
}

// Catches: `max_items` taken as unsigned.
#[test]
fn truncate_plan_slices_a_negative_limit_as_python_does() {
    let c = ctx(vec![
        msg("a", Role::User, "1", 1),
        msg("b", Role::User, "2", 2),
        msg("c", Role::User, "3", 3),
    ]);
    assert_eq!(truncate_plan(&facts(&c), -1), Some(vec![1, 2]));
    assert_eq!(truncate_plan(&facts(&c), -5), Some(vec![]));
    assert_eq!(truncate_plan(&facts(&c), 0), Some(vec![0, 1, 2]));
    assert_eq!(truncate_plan(&facts(&c), 3), None);
}

// Catches: plans keyed by id rather than position.
#[test]
fn plans_are_positional_so_duplicate_ids_keep_their_own_items() {
    let c = ctx(vec![
        msg("s", Role::System, "sys", 1),
        msg("x", Role::User, "first", 2),
        msg("x", Role::User, "second", 3),
    ]);
    assert_eq!(truncate_plan(&facts(&c), 1), Some(vec![0, 2]));
    let theirs = ctx(vec![msg("y", Role::User, "a", 4), msg("y", Role::User, "b", 5)]);
    assert_eq!(merge_plan(&facts(&c), &facts(&theirs), &MergeOptions::default()), [(3, 0)]);
}


// Catches: `<` instead of `<=`, appending when nothing precedes the stamp, or the reverse
// linear scan "optimised" into a binary search (visible on an out-of-order list).
#[test]
fn find_insertion_index_scans_from_the_end_and_puts_a_tie_after() {
    let c = ctx(vec![msg("a", Role::User, "1", 10), msg("b", Role::Assistant, "2", 20)]);
    assert_eq!(c.find_insertion_index(t(10)), 1);
    assert_eq!(c.find_insertion_index(t(20)), 2);
    assert_eq!(c.find_insertion_index(t(5)), 0);
    assert_eq!(ChatContext::default().find_insertion_index(t(5)), 0);
    let unordered = ctx(vec![
        msg("a", Role::User, "1", 10),
        msg("b", Role::User, "2", 30),
        msg("c", Role::User, "3", 15),
    ]);
    assert_eq!(unordered.find_insertion_index(t(20)), 3);
}



// Catches: computing every index up front, which reverses a batch.
#[test]
fn insert_all_places_each_item_against_the_list_as_it_grows() {
    let mut c = ctx(vec![msg("a", Role::User, "1", 10)]);
    c.insert_all(vec![out("o1", "f", 20), out("o2", "f", 20), out("o3", "f", 20)]);
    assert_eq!(ids(&c), ["a", "o1", "o2", "o3"]);
}


// Catches: keeping a window that starts on a function call or its output.
#[test]
fn truncate_drops_leading_function_calls_and_outputs() {
    let c = ctx(vec![
        msg("m0", Role::User, "old", 1),
        call("c1", "f", 2),
        out("o1", "f", 3),
        msg("m1", Role::Assistant, "new", 4),
    ]);
    assert_eq!(truncated(&c, 3), ["m1"]);
}

// Catches: #4845 -- the instruction lookup written as `role == System`.
#[test]
fn truncate_restores_a_developer_instruction_that_fell_out_of_the_window() {
    let c = ctx(vec![
        msg("dev", Role::Developer, "sys", 1),
        msg("m1", Role::User, "a", 2),
        msg("m2", Role::Assistant, "b", 3),
    ]);
    assert_eq!(truncated(&c, 2), ["dev", "m1", "m2"]);
}

// Catches: re-inserting the instruction when the window already has it.
#[test]
fn truncate_does_not_duplicate_an_instruction_still_inside_the_window() {
    let c = ctx(vec![
        msg("m0", Role::User, "old", 1),
        msg("sys", Role::System, "sys", 2),
        msg("m1", Role::User, "a", 3),
    ]);
    assert_eq!(truncated(&c, 2), ["sys", "m1"]);
}


// Catches: collapsing `NOT_GIVEN` and `[]` into one "no tools" case, or applying the
// tool-name filter to items that have no name.
#[test]
fn copy_distinguishes_an_absent_tool_list_from_an_empty_one() {
    let base = ctx(vec![
        msg("m", Role::User, "hi", 1),
        handoff("h", 2),
        config_update("u", 3),
        call("c", "lookup", 4),
    ]);
    let with = |tools: Option<Vec<String>>| ids(&base.copy(&CopyOptions { tools, ..Default::default() })).iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(with(None), ["m", "h", "u", "c"]);
    assert_eq!(with(Some(vec![])), ["m", "h", "u"]);
    assert_eq!(with(Some(vec!["lookup".into()])), ["m", "h", "u", "c"]);
}


// Catches: reading `exclude_empty_message` as "empty text".
#[test]
fn copy_exclude_empty_message_tests_the_content_list_not_the_text() {
    let base = ctx(vec![empty_msg("none", 1), msg("blank", Role::Assistant, "", 2)]);
    let kept = base.copy(&CopyOptions { exclude_empty_message: true, ..Default::default() });
    assert_eq!(ids(&kept), ["blank"]);
}

// Catches: de-duplicating by content instead of by id.
#[test]
fn merge_skips_ids_already_present() {
    let parent = ctx(vec![msg("a", Role::User, "hi", 1)]);
    let child = ctx(vec![msg("a", Role::User, "hi", 1), msg("b", Role::Assistant, "yo", 2)]);
    let once = merged(&parent, &child, &MergeOptions::default());
    assert_eq!(ids(&merged(&once, &child, &MergeOptions::default())), ["a", "b"]);
}

// Catches: appending the other context's items rather than placing them by `created_at`.
#[test]
fn merge_orders_by_created_at_rather_than_appending() {
    let parent = ctx(vec![msg("late", Role::Assistant, "z", 30)]);
    let child = ctx(vec![msg("mid", Role::User, "m", 20), msg("early", Role::User, "e", 10)]);
    assert_eq!(ids(&merged(&parent, &child, &MergeOptions::default())), ["early", "mid", "late"]);
}

// Catches: giving `merge` `copy`'s option set.
#[test]
fn merge_keeps_the_items_copy_can_drop() {
    let child = ctx(vec![handoff("h", 1), empty_msg("e", 2), config_update("u", 3)]);
    let opts = MergeOptions { exclude_function_call: true, exclude_instructions: true, exclude_config_update: true };
    assert_eq!(ids(&merged(&ChatContext::default(), &child, &opts)), ["h", "e"]);
}

// Catches: `push` quietly becoming a sorted insert.
#[test]
fn push_appends_and_insert_sorts() {
    let mut c = ChatContext::default();
    c.push(ChatMessage::text("late", Role::User, "1", t(30)));
    c.push(ChatMessage::text("early", Role::User, "2", t(10)));
    assert_eq!(ids(&c), ["late", "early"]);

    c.insert(ChatMessage::text("mid", Role::User, "3", t(20)));
    assert_eq!(ids(&c), ["late", "early", "mid"]);
}


// Catches: computing the window start with anything but a subtraction.
#[test]
fn truncate_keeps_exactly_the_last_n_items() {
    let five =
        || ctx((1..=5).map(|i| msg(&format!("m{i}"), Role::User, "x", i)).collect::<Vec<_>>());
    let c = five();
    assert_eq!(truncated(&c, 2), ["m4", "m5"], "5 - 2 = 3, not 5 / 2 = 2");
    assert_eq!(truncated(&c, 3), ["m3", "m4", "m5"], "5 - 3 = 2, not 5 / 3 = 1");
    assert_eq!(truncated(&c, 1), ["m5"]);
}

// Catches: a `copy` filter wired to the wrong predicate (`||` for `&&`).
#[test]
fn copy_applies_each_kind_filter_only_when_its_flag_is_set() {
    let base = ctx(vec![
        msg("sys", Role::System, "you are Ada", 1),
        msg("dev", Role::Developer, "debug", 2),
        msg("u", Role::User, "hi", 3),
        handoff("h", 4),
        config_update("cfg", 5),
    ]);

    assert_eq!(
        ids(&base.copy(&CopyOptions::default())),
        ["sys", "dev", "u", "h", "cfg"],
        "with no flags set every kind survives"
    );
    assert_eq!(
        ids(&base.copy(&CopyOptions { exclude_instructions: true, ..Default::default() })),
        ["u", "h", "cfg"],
        "both instruction roles go, and nothing else does"
    );
    assert_eq!(
        ids(&base.copy(&CopyOptions { exclude_handoff: true, ..Default::default() })),
        ["sys", "dev", "u", "cfg"]
    );
    assert_eq!(
        ids(&base.copy(&CopyOptions { exclude_config_update: true, ..Default::default() })),
        ["sys", "dev", "u", "h"]
    );
}

// Catches: `merge_excludes` joining its three clauses with `&&`.
#[test]
fn merge_drops_each_excluded_kind_on_its_own() {
    let child = ctx(vec![
        call("c", "book", 1),
        out("o", "book", 2),
        msg("sys", Role::System, "you are Ada", 3),
        config_update("cfg", 4),
        msg("u", Role::User, "hi", 5),
    ]);
    let kept = |opts: MergeOptions| {
        merged(&ChatContext::default(), &child, &opts).items.iter().map(|i| i.id().to_owned()).collect::<Vec<_>>()
    };

    assert_eq!(kept(MergeOptions::default()), ["c", "o", "sys", "cfg", "u"]);
    assert_eq!(
        kept(MergeOptions { exclude_function_call: true, ..Default::default() }),
        ["sys", "cfg", "u"]
    );
    assert_eq!(
        kept(MergeOptions { exclude_instructions: true, ..Default::default() }),
        ["c", "o", "cfg", "u"]
    );
    assert_eq!(
        kept(MergeOptions { exclude_config_update: true, ..Default::default() }),
        ["c", "o", "sys", "u"]
    );
}

// Catches: `messages()` including function calls or handoffs.
#[test]
fn messages_skips_every_non_message_kind() {
    let c = ctx(vec![
        msg("m", Role::User, "hi", 1),
        call("c", "f", 2),
        out("o", "f", 3),
        handoff("h", 4),
        config_update("u", 5),
    ]);
    assert_eq!(c.messages().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["m"]);
}

// Catches: comparing image content by anything other than its side-table id.
#[test]
fn media_content_survives_a_copy_and_compares_by_side_table_id() {
    struct Noop;
    impl crate::RefRegistry for Noop {
        fn release(&self, _id: u64) {}
    }
    let registry: std::sync::Arc<dyn crate::RefRegistry> = std::sync::Arc::new(Noop);

    let image = |id| {
        ChatContent::Image(crate::ImageContent {
            id: "img_1".into(),
            image: crate::ImageSource::Frame(crate::MediaRef::new(id, &registry)),
            inference_width: None,
            inference_height: None,
            inference_detail: crate::InferenceDetail::Auto,
            mime_type: None,
        })
    };
    let audio = ChatContent::Audio(crate::AudioContent {
        frames: crate::MediaRef::new(99, &registry),
        transcript: Some("hi".into()),
    });

    let a = ctx(vec![
        ChatMessage::new("m", Role::User, vec![text("look"), image(1), audio.clone()], t(1))
            .into(),
    ]);
    let copied = a.copy(&CopyOptions::default());
    assert_eq!(copied.items[0].as_message().unwrap().content.len(), 3);
    assert_eq!(a, copied);

    let b = ctx(vec![
        ChatMessage::new("m", Role::User, vec![text("look"), image(2), audio], t(1)).into(),
    ]);
    assert_ne!(a, b, "two different frames must not read as the same context");
}
