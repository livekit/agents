//! An `llm_node` that answers text itself made no model call, so no `LLMMetrics` is reported.
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{NodeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis};
use lk_agents_voice::events::SessionEvent;

// Catches: the core metering the node's output rather than the model call: a node returning a
// `str` got an `LLMMetrics` the reference never emits (it meters the `LLMStream`, and such a node
// opens none).
#[test]
fn a_node_answering_text_itself_is_not_metered() {
    let nodes = NodeSetup { llm: true, tts: false, stt: false, llm_text: Some("A fixed line.") };
    let setup = Setup { tts: false, nodes: Some(nodes), ..Setup::default() };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(500) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let events = run.record.host_view().events();
    let spoken = events.iter().any(|(_, ev)| {
        matches!(ev, SessionEvent::ConversationItemAdded { item, .. }
            if matches!(item.as_ref(), ChatItem::Message(m) if m.role == Role::Assistant))
    });
    assert!(spoken, "the node's text is the reply");
    let metered = events.iter().any(|(_, ev)| {
        matches!(ev, SessionEvent::MetricsCollected { metrics, .. }
            if matches!(metrics.as_ref(), AgentMetrics::Llm(_)))
    });
    assert!(!metered, "no model call, no LLMMetrics");
}
