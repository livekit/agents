//! Context progress and ordering with cleanup admission withheld.
use super::*;

#[test]
fn retirement_only_never_defers_audio_replies_or_items() {
    let mut m = machine();
    m.open(at(0));
    let first =
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(32) }));
    let mut frames = first.frames;
    for _ in 0..32 {
        if !m.context_pages_pending() {
            break;
        }
        let page = m.continue_context(at(0));
        assert!(page.fatal.is_none());
        frames.extend(page.frames);
    }
    assert!(!m.context_pages_pending());
    assert!(m.has_context_retirement());
    for _ in 0..160 {
        let step = m.command(at(0), audio(2));
        assert!(!step.recycle && step.fatal.is_none());
        assert_eq!(step.frames.len(), 1, "cleanup must not defer audio");
    }
    let reply = m.command(
        at(0),
        Command::GenerateReply(GenerateReply {
            req_id: "during-cleanup".into(),
            instructions: None,
            tool_choice: None,
        }),
    );
    assert_eq!(reply.frames.len(), 1);
    let pushed = m.command(
        at(0),
        Command::PushItem(PushItem {
            item: Box::new(message("after-cleanup")),
            reply_required: false,
        }),
    );
    assert_eq!(creates(&pushed), ["after-cleanup"]);
    // Replay must preserve the pushed tail after an exact page-sized emission.
    retire_context(&mut m);
    let first = m.reconnect(at(1));
    let replay = finish_context(&mut m, at(1), first);
    assert!(creates(&replay).contains(&"after-cleanup"));
}

#[test]
fn ordinary_reconnect_waits_for_retirement_capacity() {
    let mut m = machine();
    m.open(at(0));
    let first =
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }));
    let initial = finish_context(&mut m, at(0), first);
    acknowledge_page(&mut m, initial);
    retire_context(&mut m);
    let first = m.reconnect(at(1));
    assert!(first.fatal.is_none());
    let mut submitted: Vec<String> = creates(&first).into_iter().map(str::to_owned).collect();
    for _ in 0..10_000 {
        if !m.context_continuation_ready() {
            break;
        }
        let page = m.continue_context(at(1));
        submitted.extend(creates(&page).into_iter().map(str::to_owned));
        assert!(page.fatal.is_none(), "ordinary continuation must backpressure");
        assert!(!m.closing);
        assert!(m.retired.len() <= 4);
        if !m.context_pages_pending() {
            break;
        }
    }
    assert!(m.has_context_retirement());
    assert!(m.context_pages_pending(), "replay must pause until cleanup can be admitted");
    assert!(!m.context_continuation_ready());
    retire_context(&mut m);
    let rest = finish_context(&mut m, at(1), Step::default());
    assert!(rest.fatal.is_none() && !m.closing);
    submitted.extend(creates(&rest).into_iter().map(str::to_owned));
    assert_eq!(submitted, (0..10_000).map(|id| id.to_string()).collect::<Vec<_>>());
    assert!(!m.context_pages_pending());
}

#[test]
fn content_only_changes_do_not_restart_snapshot_progress() {
    let mut m = machine();
    m.open(at(0));
    let first =
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(300) }));
    let initial = finish_context(&mut m, at(0), first);
    acknowledge_page(&mut m, initial);
    retire_context(&mut m);
    m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(301) }));
    let mut completed = false;
    for _ in 0..100 {
        m.incoming_on(
            m.connection_generation(),
            at(0),
            Incoming::TranscriptionCompleted(crate::realtime::protocol::TranscriptionCompleted {
                item_id: "0".into(),
                content_index: 0,
                transcript: "continued".into(),
                confidence: None,
                in_progress: false,
                event_id: "transcript".into(),
                usage: None,
            }),
        );
        let page = m.continue_context(at(0));
        assert!(page.fatal.is_none());
        retire_context(&mut m);
        if !m.context_pages_pending() {
            completed = true;
            break;
        }
    }
    assert!(completed, "steady content updates must not starve structural traversal");
}

#[test]
fn rejected_delete_preserves_order_after_another_deleted_predecessor() {
    let mut m = machine();
    m.open(at(0));
    acknowledge(&mut m, at(0), "keep", None);
    acknowledge(&mut m, at(0), "a", Some("keep"));
    acknowledge(&mut m, at(0), "b", Some("a"));
    acknowledge(&mut m, at(0), "last", Some("b"));
    let first = m.command(
        at(0),
        Command::UpdateChatCtx(UpdateChatCtx {
            chat_ctx: ChatContext::new(vec![message("keep"), message("last")]),
        }),
    );
    let edits = finish_context(&mut m, at(0), first);
    let event_id = edits
        .frames
        .iter()
        .find_map(|frame| match frame {
            Outgoing::ItemDelete { item_id, event_id } if item_id == "b" => Some(event_id.clone()),
            _ => None,
        })
        .unwrap();
    let rejection = m.incoming_on(
        m.connection_generation(),
        at(0),
        Incoming::Error(crate::realtime::protocol::ServerError {
            event_id: Some(event_id),
            code: Some("delete_rejected".into()),
            type_: None,
            message: "cannot delete b".into(),
        }),
    );
    finish_context(&mut m, at(0), rejection);
    let first = m.reconnect(at(1));
    let replay = finish_context(&mut m, at(1), first);
    assert_eq!(creates(&replay), ["keep", "b", "last"]);
}

#[test]
fn deferred_audio_and_truncation_share_the_remaining_page_allowance() {
    use lk_agents_abi::llm::realtime::Truncate;
    let mut m = machine();
    m.open(at(0));
    // Duplicate IDs are filtered during replay. The remaining62 validation units
    // emit nothing and leave one desired item before the deferred63audio + truncate.
    let first = m.command(
        at(0),
        Command::UpdateChatCtx(UpdateChatCtx {
            chat_ctx: ChatContext::new(vec![message("only"); 63]),
        }),
    );
    assert_eq!(creates(&first), ["only"]);
    for _ in 0..63 {
        assert!(m.command(at(0), audio(2)).frames.is_empty());
    }
    assert!(
        m.command(
            at(0),
            Command::Truncate(Truncate {
                message_id: "only".into(),
                has_audio: false,
                audio_end_ms: 0,
                audio_transcript: Some("cut".into()),
            })
        )
        .frames
        .is_empty()
    );
    let mut audio_frames = 0;
    let mut edited = false;
    for _ in 0..100 {
        let page = m.continue_context(at(0));
        assert!(page.fatal.is_none());
        assert!(
            page.frames.len() <= 64,
            "nested truncate exceeded its page: {}",
            page.frames.len()
        );
        for frame in page.frames {
            match frame {
                Outgoing::AudioAppend { .. } => audio_frames += 1,
                Outgoing::ItemCreate { item: ChatItem::Message(message), .. } => {
                    edited |=
                        message.content == vec![lk_agents_abi::ChatContent::Text("cut".into())];
                }
                _ => {}
            }
        }
        retire_context(&mut m);
        if !m.context_pages_pending() {
            break;
        }
    }
    assert_eq!(audio_frames, 63);
    assert!(edited);
    assert!(!m.context_pages_pending());
}

#[test]
fn repeated_reconnects_preserve_replay_with_initial_cleanup_withheld() {
    let mut m = machine();
    m.open(at(0));
    let mut initial =
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }));
    for _ in 0..1_000 {
        if !m.context_pages_pending() {
            break;
        }
        assert!(m.context_continuation_ready(), "initial submission unexpectedly parked");
        let page = m.continue_context(at(0));
        assert!(page.fatal.is_none());
        initial.merge(page);
    }
    assert!(!m.context_pages_pending());
    assert_eq!(creates(&initial).len(), 10_000);
    acknowledge_page(&mut m, initial);
    assert!(m.has_context_retirement());
    assert!(m.retired.len() <= 3);

    let first = m.reconnect(at(1));
    assert!(first.fatal.is_none(), "the first reconnect exhausted cleanup slots");
    assert!(first.frames.iter().all(|frame| !matches!(frame, Outgoing::ItemCreate { .. })));
    let queued = m.retired.len();
    let second = m.reconnect(at(2));
    assert!(second.fatal.is_none(), "an empty old connection consumed a fifth cleanup slot");
    assert!(!m.closing);
    assert_eq!(m.retired.len(), queued, "empty connection state must not add a cleanup batch");
    assert!(!m.context_continuation_ready(), "replay must await withheld disposal admission");

    retire_context(&mut m);
    let replay = finish_context(&mut m, at(2), second);
    assert!(replay.fatal.is_none() && !m.closing);
    assert_eq!(creates(&replay), (0..10_000).map(|id| id.to_string()).collect::<Vec<_>>());
}

fn completed_update_with_cleanup_withheld() -> RealtimeMachine {
    let mut m = machine();
    m.open(at(0));
    let mut submitted =
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }));
    while m.context_pages_pending() {
        assert!(m.context_continuation_ready());
        let page = m.continue_context(at(0));
        assert!(page.fatal.is_none());
        submitted.merge(page);
    }
    acknowledge_page(&mut m, submitted);
    assert_eq!(m.retired.len(), 2);
    m
}

fn expect_terminal_context_refusal(m: &mut RealtimeMachine, step: Step, req_id: &str) {
    let error = step.fatal.expect("external context bursts must refuse at the ownership cap");
    assert!(!error.recoverable);
    assert_eq!(error.message, "realtime context retirement capacity exceeded");
    assert_eq!(
        step.events
            .iter()
            .filter(|event| matches!(event,
            Event::GenerateReplyFailed(failed) if failed.req_id == req_id))
            .count(),
        1,
        "terminal refusal must settle the outstanding reply exactly once"
    );
    assert!(m.closing);
    assert!(m.has_context_retirement(), "terminal refusal must retain cleanup ownership");
    // Merely observing the closed machine or trying to continue it cannot dispose its owners.
    assert!(m.continue_context(at(1)).frames.is_empty());
    assert!(m.has_context_retirement());
    let mut admitted = Vec::new();
    while let Some(owner) = m.take_context_retirement() {
        admitted.push(owner);
    }
    assert!(!admitted.is_empty());
    assert!(!m.has_context_retirement());
    // In production these opaque owners are dropped only by the admitted cleanup worker.
    drop(admitted);
}

#[test]
fn external_update_burst_refuses_terminally_when_prepare_owners_exhaust_cleanup_capacity() {
    let mut m = completed_update_with_cleanup_withheld();
    m.command(
        at(0),
        Command::GenerateReply(GenerateReply {
            req_id: "prepare-burst".into(),
            instructions: None,
            tool_choice: None,
        }),
    );
    // U2 parks. U3 and U4 replace its entire 10k Prepare owner, filling slots 3 and 4.
    for expected in [2, 3, 4] {
        let step = m.command(
            at(0),
            Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }),
        );
        assert!(step.fatal.is_none());
        assert_eq!(m.retired.len(), expected);
        assert!(!m.context_continuation_ready());
    }
    let refused =
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }));
    expect_terminal_context_refusal(&mut m, refused, "prepare-burst");
}

#[test]
fn external_update_burst_refuses_terminally_when_overlay_pending_inputs_exhaust_cleanup_capacity() {
    let mut m = completed_update_with_cleanup_withheld();
    let reconnect = m.reconnect(at(1));
    assert!(reconnect.fatal.is_none());
    assert_eq!(m.retired.len(), 3);
    m.command(
        at(1),
        Command::GenerateReply(GenerateReply {
            req_id: "overlay-burst".into(),
            instructions: None,
            tool_choice: None,
        }),
    );
    // U2 becomes the pending input; U3 retires that input as the fourth large owner.
    for expected in [3, 4] {
        let step = m.command(
            at(1),
            Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }),
        );
        assert!(step.fatal.is_none());
        assert_eq!(m.retired.len(), expected);
        assert!(!m.context_continuation_ready());
    }
    let refused =
        m.command(at(1), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }));
    expect_terminal_context_refusal(&mut m, refused, "overlay-burst");
}
