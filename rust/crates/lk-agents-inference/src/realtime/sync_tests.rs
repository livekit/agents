//! The realtime core keeps the server's conversation in step with the session's: across
//! reconnects, context edits and cancelled responses, it sends only frames the server takes, and
//! what it tells the driver belongs to the response it names. Each case runs the core over the
//! simulated server ([`super::sim`]).
use std::time::SystemTime;

use lk_agents_abi::{
    ChatContext, ChatItem, ChatMessage, FunctionCallOutput, GenerateReply, PushItem, Role,
    ToolSpec, UpdateChatCtx, UpdateTools,
    llm::realtime::{CommitAudio, Event, Interrupt},
    metrics::base::AgentMetrics,
};

use super::server::Exo;
use super::sim::{Command, Setup, Sim};

fn reply(req_id: &str) -> Command {
    Command::GenerateReply(GenerateReply {
        req_id: req_id.into(),
        instructions: None,
        tool_choice: None,
    })
}

fn message(id: &str, role: Role, text: &str) -> ChatItem {
    ChatItem::Message(ChatMessage::text(id, role, text, SystemTime::UNIX_EPOCH))
}

/// The developer's context: these items, in order.
fn context(items: Vec<ChatItem>) -> Command {
    Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: ChatContext::new(items) })
}

/// A system prompt saying `text` and the person's question, under the same ids every time.
fn edit(text: &str) -> Command {
    context(vec![
        message("ctx_system", Role::System, text),
        message("ctx_user", Role::User, "where is my order"),
    ])
}

/// The response that answered `req_id`.
fn answered(sim: &Sim, req_id: &str) -> String {
    sim.events()
        .iter()
        .find_map(|(_, e)| match e {
            Event::GenerationCreated(c) if c.req_id.as_deref() == Some(req_id) => {
                Some(c.response_id.clone())
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("{req_id} was never answered"))
}

// Catches: the session taking the cancelled response's tail for the reply asked in the same
// breath. The real server frees the conversation at a cancel and creates the new reply before the
// cancelled one's last audio delta and `response.done` (`probe-cancel_then_create`). Its late
// deltas were told as the new reply's, and its late `response.done` ended the new reply's
// generation, whose audio was then dropped: the agent fell silent. Over several draws, so the
// server creates the new reply both before and after the cancelled one ends.
#[test]
fn a_cancelled_response_tail_leaves_the_next_reply_alone() {
    let mut overlapped = false;
    for seed in 0..8 {
        let mut sim = Sim::new(Setup { seed, ..Setup::default() });
        sim.send(reply("first"));
        sim.advance(1_000);
        let first = answered(&sim, "first");
        sim.send(Command::Interrupt(Interrupt { response_id: Some(first.clone()) }));
        sim.send(reply("second"));
        sim.advance(20_000);
        overlapped |= sim.overlapped();
        let second = answered(&sim, "second");
        let messages: Vec<&str> = sim
            .events()
            .iter()
            .filter_map(|(_, e)| match e {
                Event::MessageStarted(m) if m.response_id.as_deref() == Some(&second) => {
                    Some(m.message_id.as_str())
                }
                _ => None,
            })
            .collect();
        let audio: Vec<(&str, Option<&str>)> = sim
            .events()
            .iter()
            .filter_map(|(_, e)| match e {
                Event::AudioDelta(a) => Some((a.message_id.as_str(), a.response_id.as_deref())),
                _ => None,
            })
            .collect();
        assert!(
            audio.iter().any(|(m, _)| messages.contains(m)),
            "seed {seed}: the new reply's audio is told"
        );
        assert!(
            audio.iter().all(|(m, r)| messages.contains(m) == (*r == Some(&second))),
            "seed {seed}: each delta is told as its own reply's: {audio:?}"
        );
        let ends: Vec<_> = sim
            .events()
            .iter()
            .filter_map(|(_, e)| match e {
                Event::GenerationDone(d) if d.response_id == second => Some(d.status),
                _ => None,
            })
            .collect();
        assert_eq!(ends.len(), 1, "seed {seed}: the new reply ends once: {ends:?}");
    }
    assert!(overlapped, "no draw created the new reply before the cancelled one ended");
}

/// The server drops the socket; it takes connections again 100 ms later.
fn drop_socket(sim: &mut Sim) {
    sim.drop_socket();
    sim.advance(100);
    sim.restore();
}

// Catches: a context update right after a reconnect replaying the conversation the reconnect had
// just restored: every item went twice, and the server refused each second one
// (`item_create_duplicate_item_id`).
#[test]
fn a_reconnect_restores_the_conversation_once() {
    let mut sim = Sim::new(Setup::default());
    sim.send(edit("You are helpful."));
    sim.send(reply("first"));
    sim.advance(10_000);
    drop_socket(&mut sim);
    sim.send(edit("You are helpful."));
    sim.advance(5_000);
    assert!(sim.refused().is_empty(), "{:?}", sim.refused());
}

// Catches: the session counting the audio it sent before a reconnect as still in the server's
// buffer. The new connection starts empty, so the commit of the person's turn went out over
// nothing and the server refused it (`input_audio_buffer_commit_empty`).
#[test]
fn a_commit_after_a_reconnect_names_audio_the_server_has() {
    let mut sim = Sim::new(Setup::default());
    sim.speak(300);
    sim.advance(300);
    drop_socket(&mut sim);
    sim.advance(5_000);
    sim.send(Command::CommitAudio(CommitAudio {}));
    sim.advance(1_000);
    assert!(sim.refused().is_empty(), "{:?}", sim.refused());
}

// Catches: a reconnect that crossed a context update the lost connection had not confirmed
// replaying the confirmed conversation and then resending that update's frames: what the update
// deletes and creates again under the same id was created twice, and what was sent right after
// the replay was diffed against nothing (`item_create_duplicate_item_id`). The new connection
// gets the conversation the server was being brought to, once.
#[test]
fn a_reconnect_across_an_unconfirmed_edit_creates_each_item_once() {
    let mut sim = Sim::new(Setup::default());
    sim.send(reply("first"));
    sim.advance(5_000);
    sim.send(edit("Hello."));
    sim.advance(1_000);
    // A second edit, its items' ids the first's, and the socket goes before it is confirmed.
    sim.send(edit("Hello again."));
    sim.drop_socket();
    sim.send(edit("Hello once more."));
    sim.advance(100);
    sim.restore();
    sim.advance(5_000);
    assert!(sim.refused().is_empty(), "{:?}", sim.refused());
}

// Catches: the output of a tool whose call the developer's context update removed while it ran,
// pushed to the server anyway: it names a call the server no longer holds
// (`invalid_tool_call_id`). It is not sent.
#[test]
fn an_output_for_a_removed_call_is_not_sent() {
    let mut sim = Sim::new(Setup { tool_calls: Some(100), ..Setup::default() });
    let lookup = ToolSpec {
        provider_tool: None,
        name: "lookup".into(),
        description: "Look an order up.".into(),
        json_schema: serde_json::json!({"type": "object", "properties": {}}),
        flags: Default::default(),
    };
    sim.send(Command::UpdateTools(UpdateTools { tools: vec![lookup] }));
    sim.send(reply("first"));
    sim.advance(3_000);
    let call = sim
        .events()
        .iter()
        .find_map(|(_, e)| match e {
            Event::FunctionCall(f) => Some(f.call.call_id.clone()),
            _ => None,
        })
        .expect("the model called its tool");
    // The developer replaces the context, call included, while the tool runs.
    sim.send(edit("Hello."));
    sim.advance(1_000);
    let output = FunctionCallOutput::new("output", call, "shipped", false, SystemTime::UNIX_EPOCH);
    let item = Box::new(ChatItem::FunctionCallOutput(output));
    sim.send(Command::PushItem(PushItem { item, reply_required: true }));
    sim.advance(5_000);
    assert!(sim.refused().is_empty(), "{:?}", sim.refused());
}

// Catches: a context update sent while the previous one was unconfirmed, diffed against what the
// server had confirmed rather than what it was being brought to: it deleted again the items the
// first one had deleted (`item_delete_invalid_item_id`).
#[test]
fn an_edit_over_an_unconfirmed_edit_deletes_once() {
    let mut sim = Sim::new(Setup::default());
    sim.send(reply("first"));
    sim.advance(5_000);
    sim.send(edit("Hello."));
    sim.send(context(vec![message("ctx_again", Role::User, "Again.")]));
    sim.advance(2_000);
    assert!(sim.refused().is_empty(), "{:?}", sim.refused());
}

// Catches: the session stamping its `RealtimeModelMetrics` (each response's, and the
// connection's) with the seconds its clock had run (a `RealClock` counts from the model's
// creation): a response reported at `timestamp=5.1` on a call in 2026. The reference stamps them
// with `time.time()`.
#[test]
fn realtime_metrics_are_stamped_on_the_wall_clock() {
    let mut sim = Sim::new(Setup::default());
    sim.send(reply("first"));
    sim.advance(20);
    assert!(sim.exo(Exo::FinishNow) || sim.events().len() > 1);
    sim.advance(3_000);
    let stamps: Vec<f64> = sim
        .events()
        .iter()
        .filter_map(|(_, e)| match e {
            Event::Metrics(m) => match &*m.metrics {
                AgentMetrics::RealtimeModel(m) => Some(m.timestamp),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert!(stamps.len() >= 2, "the connection's and the response's metrics: {stamps:?}");
    assert!(stamps.iter().all(|t| *t > 1.7e9), "{stamps:?}");
}
