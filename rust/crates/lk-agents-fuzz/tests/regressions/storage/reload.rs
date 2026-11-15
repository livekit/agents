//! A turn's reload of the partition another writer moved: the rows the call changed since its
//! last commit stay as the call has them.

use lk_agents_abi::ChatItem;
use lk_agents_abi::llm::tool_executor::WAITING_PREFIX;
use lk_agents_fuzz::drive::{Run, Setup, StorageSetup, simulate};
use lk_agents_fuzz::input::{
    AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
};
use lk_agents_voice::events::SessionEvent;

/// The person's turn: `words`, heard and ended.
fn says(utterance: u32, words: &str) -> [Input; 7] {
    [
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(300) },
        Input::SttFinal {
            utterance: Ref::new(utterance),
            text: UserText(words.into()),
            speaker: None,
        },
        Input::SttEnd { utterance: Ref::new(utterance) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(3_000) },
    ]
}

// Catches (seed 1682): the store compared what it last committed of a row with the row read back,
// though the stored form keeps less of an item (a user message's STT metadata): every row read as
// another writer's edit, so a reload after a text turn put back the question the person had just
// answered, and their next words answered it again while the question asked since waited.
#[test]
fn an_answered_question_stays_answered_across_a_reload() {
    let (mut reloaded, mut exercised) = (0, 0);
    for seed in 0..8 {
        let setup = Setup {
            vad: true,
            stt: true,
            turn: TurnMode::Stt,
            tools: vec!["lookup".into()],
            storage: Some(StorageSetup::default()),
            ..Setup::default()
        };
        let ask = |chat: u32, tool: u32, question: &str| {
            [
                Input::LlmToolCall {
                    chat: Ref::new(chat),
                    name: "lookup".into(),
                    args: ToolArgs("{}".into()),
                },
                Input::LlmEnd { chat: Ref::new(chat) },
                Input::Advance { ms: Millis(100) },
                Input::ToolAnswered {
                    tool: Ref::new(tool),
                    answer: ToolAnswer::AskVerbatim,
                    text: AgentText(question.into()),
                },
                Input::Advance { ms: Millis(3_000) },
            ]
        };
        let inputs: Vec<Input> = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
        ]
        .into_iter()
        .chain(ask(0, 0, "Which day?"))
        .chain(says(0, "Monday"))
        .chain(ask(1, 1, "Which city?"))
        .chain(says(1, "Paris"))
        .chain(says(2, "Thanks"))
        .collect();
        let run = simulate(async {
            let mut run = Run::start(&setup, seed, Default::default()).await;
            for input in &inputs {
                crate::playout::apply_with_playout(&mut run, input).await;
            }
            run
        });
        // The history as the host holds it: told item by item, replaced whole by a reload.
        let mut history: Vec<ChatItem> = Vec::new();
        let mut answered = false;
        for (_, event) in run.record.host_view().events() {
            match event {
                SessionEvent::HistoryReloaded { items } => {
                    reloaded += usize::from(answered);
                    history = items.clone();
                }
                SessionEvent::ConversationItemAdded { item, .. } => {
                    match history.iter_mut().find(|i| i.id() == item.id()) {
                        Some(told) => *told = (**item).clone(),
                        None => history.push((**item).clone()),
                    }
                }
                _ => {}
            }
            answered |= history.iter().any(
                |item| matches!(item, ChatItem::FunctionCallOutput(o) if o.output == "Monday"),
            );
        }
        let outputs: Vec<&str> = history
            .iter()
            .filter_map(|item| match item {
                ChatItem::FunctionCallOutput(o) => Some(o.output.as_str()),
                _ => None,
            })
            .collect();
        // Both questions put: the store's latencies decide whether the scripted replies line up.
        let asked = run.record.host_view().events().into_iter().filter(|(_, event)| {
            matches!(event, SessionEvent::ConversationItemAdded { item, .. }
                if matches!(item.as_ref(), ChatItem::FunctionCallOutput(o)
                    if o.output.starts_with(WAITING_PREFIX)))
        });
        if asked.count() < 2 {
            continue;
        }
        exercised += 1;
        assert_eq!(outputs, ["Monday", "Paris"], "seed {seed}: a question was put back");
    }
    assert!(exercised >= 4, "both questions were put in only {exercised} of 8 runs");
    assert!(reloaded > 0, "no reload came after the first answer");
}
