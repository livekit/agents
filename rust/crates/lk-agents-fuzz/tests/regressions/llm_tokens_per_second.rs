//! A reply's LLM rate is its completion tokens over the streaming window.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::fakes::COMPLETION_TOKENS;
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
use lk_agents_voice::events::SessionEvent;

// Catches: `llm_node_tps` counted chunks over the time since the request started, so it read
// the model's time to first token as slowness, and a provider sending one chunk per word or per
// sentence moved it tenfold. The reference divides `usage.completion_tokens` by the time from the
// first to the last text chunk (`generation.py`).
#[test]
fn the_rate_is_completion_tokens_over_the_streaming_window() {
    let setup = Setup { tts: false, ..Setup::default() };
    let text = |s: &str| Input::LlmText { chat: Ref::new(0), text: AgentText(s.into()) };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        // A slow first token: not part of the rate.
        Input::Advance { ms: Millis(800) },
        text("Your order shipped"),
        Input::Advance { ms: Millis(500) },
        text(" yesterday."),
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(500) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let rates: Vec<_> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, ev)| match ev {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::Message(m) if m.role == Role::Assistant => Some(m.metrics.llm_node_tps),
                _ => None,
            },
            _ => None,
        })
        .collect();
    // Each chunk's wall stamp is a microsecond past the last (wall time strictly increases), and
    // the model streams a word a chunk, so the window reads a hair over 0.5 s.
    let expected = f64::from(COMPLETION_TOKENS) / 0.5;
    assert!(
        matches!(rates.as_slice(), [Some(rate)] if (rate - expected).abs() < 1e-3),
        "{rates:?}, expected {expected}"
    );
}
