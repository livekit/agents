//! Unsolicited items keep their position across delayed deletion receipts and reconnects.

use super::*;

const DELETED_SUCCESSORS: usize = 300;
const UNSUBMITTED_SUFFIX: usize = 300;

fn pending_server_item(
    suffix: usize,
    server_item: ChatItem,
) -> (RealtimeMachine, Vec<Outgoing>, u64) {
    let mut m = machine();
    m.open(at(0));
    let initial = ChatContext::new(
        std::iter::once(message("a"))
            .chain((0..DELETED_SUCCESSORS).map(|id| message(&format!("deleted-{id}"))))
            .chain(std::iter::once(message("c")))
            .collect(),
    );
    let initial = m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: initial }));
    let initial = finish_context(&mut m, at(0), initial);
    acknowledge_page(&mut m, initial);
    finish_context(&mut m, at(0), Step::default());

    let target = ChatContext::new(
        [message("a"), message("b"), message("c")]
            .into_iter()
            .chain((0..suffix).map(|id| message(&format!("tail-{id}"))))
            .collect(),
    );
    let mut page = m.command(at(1), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target }));
    let mut submitted = Vec::new();
    loop {
        assert!(page.fatal.is_none());
        assert!(page.frames.len() <= 64);
        let started_creating = creates(&page).contains(&"b");
        submitted.extend(page.frames);
        if started_creating {
            break;
        }
        assert!(m.context_work_pending(), "the replacement never began its create phase");
        page = m.continue_context(at(1));
        retire_context(&mut m);
    }
    assert_eq!(
        submitted.iter().filter(|frame| matches!(frame, Outgoing::ItemDelete { .. })).count(),
        DELETED_SUCCESSORS
    );
    if suffix == 0 {
        submitted.extend(finish_context(&mut m, at(1), Step::default()).frames);
        assert!(!m.context_pages_pending(), "no prior preparation may mask the gap walk's yield");
    } else {
        assert!(
            submitted.iter().filter(|frame| matches!(frame, Outgoing::ItemCreate { .. })).count()
                < suffix + 1,
            "the schedule must retain an unsubmitted target suffix"
        );
    }

    let generation = m.connection_generation();
    let inserted = m.incoming_on(
        generation,
        at(2),
        Incoming::ItemAdded { previous_item_id: Some("a".into()), item: server_item },
    );
    assert!(
        m.chat_ctx().get_by_id("u").is_some(),
        "the acknowledged mirror must update immediately"
    );
    assert!(inserted.frames.is_empty(), "the deleted-successor walk must yield after one page");
    assert!(m.context_pages_pending(), "the successor gap must retain a continuation");
    assert!(m.tick(at(2)).frames.is_empty(), "control must remain serviceable between gap pages");
    (m, submitted, generation)
}

fn expected_ids(suffix: usize) -> Vec<String> {
    ["a", "b", "u", "c"]
        .into_iter()
        .map(str::to_owned)
        .chain((0..suffix).map(|id| format!("tail-{id}")))
        .collect()
}

fn acknowledge_frames(m: &mut RealtimeMachine, frames: Vec<Outgoing>) {
    for frame in frames {
        let incoming = match frame {
            Outgoing::ItemCreate { item, previous_item_id, .. } => Incoming::ItemAdded {
                previous_item_id: previous_item_id.filter(|id| id != "root"),
                item,
            },
            Outgoing::ItemDelete { item_id, .. } => Incoming::ItemDeleted { item_id },
            _ => continue,
        };
        m.incoming_on(m.connection_generation(), at(3), incoming);
        retire_context(m);
    }
}

#[test]
fn server_item_placement_yields_across_deleted_successors_and_keeps_local_neighbors() {
    let (mut m, mut submitted, _) = pending_server_item(0, message("u"));
    let rest = finish_context(&mut m, at(3), Step::default());
    submitted.extend(rest.frames);
    assert!(
        !submitted.iter().any(|frame| matches!(frame,
            Outgoing::ItemCreate { item, .. } if item.id() == "u"
        )),
        "an acknowledged unsolicited item must not be created again"
    );
    acknowledge_frames(&mut m, submitted);
    assert_eq!(
        m.chat_ctx().items.iter().map(|item| item.id().to_owned()).collect::<Vec<_>>(),
        expected_ids(0)
    );
}

#[test]
fn reconnect_during_server_item_placement_preserves_pending_item_and_unsubmitted_target() {
    let (mut m, _, previous_generation) = pending_server_item(UNSUBMITTED_SUFFIX, message("u"));
    let first = m.reconnect(at(3));
    assert!(creates(&first).is_empty(), "the old successor walk must finish before replay");
    assert!(m.context_work_pending());
    let second = m.reconnect(at(4));
    assert!(creates(&second).is_empty());
    m.incoming_on(
        previous_generation,
        at(4),
        Incoming::ItemAdded { previous_item_id: None, item: message("obsolete") },
    );
    assert!(
        m.chat_ctx().get_by_id("obsolete").is_none(),
        "an older socket changed the current mirror"
    );
    assert!(m.tick(at(4)).frames.is_empty());
    let replay = finish_context(&mut m, at(4), second);
    assert_eq!(
        creates(&replay),
        expected_ids(UNSUBMITTED_SUFFIX),
        "replay must preserve ordered, exactly-once identities"
    );
    acknowledge_page(&mut m, replay);
    assert_eq!(
        m.chat_ctx().items.iter().map(|item| item.id().to_owned()).collect::<Vec<_>>(),
        expected_ids(UNSUBMITTED_SUFFIX)
    );
}

#[test]
fn current_connection_items_survive_completion_of_an_older_server_item_gap() {
    let (mut m, _, _) = pending_server_item(UNSUBMITTED_SUFFIX, message("u"));
    let first = m.reconnect(at(3));
    assert!(creates(&first).is_empty());
    m.incoming_on(
        m.connection_generation(),
        at(3),
        Incoming::ItemAdded { previous_item_id: None, item: message("current") },
    );
    retire_context(&mut m);
    assert!(m.chat_ctx().get_by_id("current").is_some());
    let replay = finish_context(&mut m, at(3), Step::default());
    assert!(
        m.chat_ctx().get_by_id("current").is_some(),
        "resuming an old gap walk cleared the current acknowledged mirror"
    );
    assert_eq!(
        creates(&replay),
        expected_ids(UNSUBMITTED_SUFFIX),
        "an already acknowledged current item was recreated"
    );
    acknowledge_frames(&mut m, replay.frames);
    let mut expected = expected_ids(UNSUBMITTED_SUFFIX);
    expected.push("current".into());
    assert_eq!(
        m.chat_ctx().items.iter().map(|item| item.id().to_owned()).collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn a_tool_output_waits_for_its_acknowledged_call_during_server_item_placement() {
    let call = ChatItem::FunctionCall(lk_agents_abi::FunctionCall {
        id: "u".into(),
        call_id: "expert".into(),
        arguments: "{}".into(),
        name: "lookup".into(),
        created_at: SystemTime::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    });
    let (mut m, mut submitted, _) = pending_server_item(0, call);
    let output = |id: &str, call_id: &str| {
        ChatItem::FunctionCallOutput(lk_agents_abi::FunctionCallOutput::new(
            id,
            call_id,
            "resolved",
            false,
            SystemTime::UNIX_EPOCH,
        ))
    };
    let rejected = m.command(
        at(3),
        Command::PushItem(PushItem {
            item: Box::new(output("unknown-output", "unknown-call")),
            reply_required: false,
        }),
    );
    assert!(rejected.frames.is_empty());
    let queued = m.command(
        at(3),
        Command::PushItem(PushItem {
            item: Box::new(output("tool-result", "expert")),
            reply_required: false,
        }),
    );
    assert!(queued.frames.is_empty(), "the call placement must settle before its output is sent");
    let rest = finish_context(&mut m, at(3), queued);
    submitted.extend(rest.frames);
    assert_eq!(submitted.iter().filter(|frame| matches!(frame, Outgoing::ItemCreate { item: ChatItem::FunctionCallOutput(output), .. } if output.id == "tool-result" && output.call_id == "expert" && output.output == "resolved")).count(), 1);
    assert!(!submitted.iter().any(
        |frame| matches!(frame, Outgoing::ItemCreate { item, .. } if item.id() == "unknown-output")
    ));
}
