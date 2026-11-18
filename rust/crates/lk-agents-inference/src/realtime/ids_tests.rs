use super::*;
use crate::realtime::{OPENAI, XAI, machine::Command, testfix};
use lk_agents_abi::{ChatContext, Ts, llm::realtime::UpdateChatCtx};

#[test]
fn aliases_are_reversible_even_when_a_legal_id_collides_with_an_alias() {
    let mut names = Names::default();
    let foreign = "xai/this-is-an-item-with-a-long-provider-specific-identity";
    let alias = names.outbound(foreign);
    assert!(valid(&alias));
    let other = names.outbound(&alias);
    assert!(valid(&other));
    assert_ne!(alias, other);
    assert_eq!(names.inbound(&alias), foreign);
    assert_eq!(names.inbound(&other), alias);
    assert_eq!(names.outbound(foreign), alias);
}

#[test]
fn native_provider_ids_are_retained_until_imported_into_another_session() {
    let id = "12345678-1234-1234-1234-123456789abc";
    let mut original = Names::default();
    assert_eq!(original.inbound(id), id);
    assert_eq!(original.outbound(id), id);
    let mut destination = Names::default();
    let alias = destination.outbound(id);
    assert!(valid(&alias));
    assert_eq!(destination.inbound(&alias), id);
}

/// The application carries the exact history across dialects, while acknowledgements and
/// late transcription events use each connection's wire identities.
#[test]
fn switching_dialects_replays_foreign_history_without_replacing_application_ids() {
    let logical = "12345678-1234-1234-1234-123456789abc";
    let history = ChatContext::new(vec![testfix::msg(logical, "book Tuesday")]);
    for dialect in [XAI, OPENAI] {
        let mut machine = testfix::machine(dialect);
        let open = machine.open(Ts::ZERO);
        machine.prepare_step(open);
        let update = machine
            .command(Ts::ZERO, Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: history.clone() }));
        let prepared = machine.prepare_step(update);
        let (item, previous_item_id) = prepared
            .frames
            .into_iter()
            .find_map(|f| match f {
                Outgoing::ItemCreate { item, previous_item_id, .. } => {
                    Some((item, previous_item_id))
                }
                _ => None,
            })
            .expect("history replay");
        let wire = item.id().to_owned();
        assert!(valid(&wire));
        assert_ne!(wire, logical);
        machine.incoming_on(
            machine.connection_generation(),
            Ts::ZERO,
            Incoming::ItemAdded { item, previous_item_id },
        );
        assert_eq!(machine.chat_ctx(), history);
        let mut late = Incoming::TranscriptionFailed { item_id: wire, content_index: 0 };
        let mut ids = ProviderIds::default();
        ids.items.outbound(logical);
        ids.incoming(&mut late);
        assert!(
            matches!(late, Incoming::TranscriptionFailed { item_id, .. } if item_id == logical)
        );
        let reconnect = machine.reconnect(Ts::ZERO);
        let replay = machine.prepare_step(reconnect);
        assert!(
            replay
                .frames
                .iter()
                .filter_map(|f| match f {
                    Outgoing::ItemCreate { item, .. } => Some(item.id()),
                    _ => None,
                })
                .all(valid)
        );
    }
}

#[test]
fn function_call_outputs_share_the_calls_alias_and_preserve_root_anchors() {
    let mut ids = ProviderIds::default();
    let call = "provider/call/12345678-1234-1234-1234-123456789abc";
    let mut item = ChatItem::FunctionCall(lk_agents_abi::FunctionCall {
        id: format!("item/{call}"),
        call_id: call.into(),
        name: "book".into(),
        arguments: "{}".into(),
        created_at: testfix::TIME,
        extra: Default::default(),
        group_id: None,
    });
    ids.item(&mut item, true);
    let ChatItem::FunctionCall(wire_call) = &item else { unreachable!() };
    let alias = wire_call.call_id.clone();
    assert!(valid(&alias));
    let mut output = Outgoing::ItemCreate {
        event_id: "update".into(),
        previous_item_id: Some("root".into()),
        item: ChatItem::FunctionCallOutput(lk_agents_abi::FunctionCallOutput::new(
            "output/long/provider-id",
            call,
            "done",
            false,
            testfix::TIME,
        )),
    };
    ids.outgoing(&mut output);
    let Outgoing::ItemCreate { item: ChatItem::FunctionCallOutput(o), previous_item_id, .. } =
        output
    else {
        unreachable!()
    };
    assert_eq!(o.call_id, alias);
    assert_eq!(previous_item_id.as_deref(), Some("root"));
    ids.item(&mut item, false);
    assert!(matches!(item, ChatItem::FunctionCall(c) if c.call_id == call));
}
