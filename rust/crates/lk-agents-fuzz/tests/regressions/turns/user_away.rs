//! The user's away state: when silence marks the user away, and what holds it off.

use lk_agents_abi::Role;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
use lk_agents_voice::events::SessionEvent;

/// The user is not marked away while a tool runs or its follow-up reply plays.
/// https://github.com/livekit/agents/issues/6904 https://github.com/livekit/agents/issues/8317
#[test]
fn a_busy_tool_cannot_mark_the_user_away() {
    let run = simulate(async {
        let mut run =
            Run::start(&Setup { tts: false, ..Setup::default() }, 0, Default::default()).await;
        for input in [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20_000) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Value,
                text: AgentText("done".into()),
            },
            Input::Advance { ms: Millis(100) },
            Input::LlmText { chat: Ref::new(1), text: AgentText("All done.".into()) },
            Input::LlmEnd { chat: Ref::new(1) },
            Input::Advance { ms: Millis(500) },
            Input::Advance { ms: Millis(15_100) },
        ] {
            run.apply(&input).await;
        }
        run
    });
    let away: Vec<_> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(at, event)| match event {
            SessionEvent::UserStateChanged {
                new: lk_agents_abi::voice::events::UserState::Away,
                ..
            } => Some(at.at_ms),
            _ => None,
        })
        .collect();
    assert_eq!(away.len(), 1, "{away:?}");
    assert!(away[0] >= 35_200, "away fired during tool execution or the follow-up: {away:?}");
    assert_eq!(
        run.record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, event)| match event {
                SessionEvent::ConversationItemAdded { item, .. } => item
                    .as_message()
                    .filter(|m| m.role == Role::Assistant)
                    .and_then(|m| m.raw_text_content()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        ["All done."]
    );
}

/// User activity restarts the away window, and no away state follows the close.
/// https://github.com/livekit/agents/issues/6030 https://github.com/livekit/agents/issues/2364
#[test]
fn activity_rearms_away_detection_without_post_close_events() {
    use lk_agents_abi::voice::events::UserState;
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in [
            Input::Advance { ms: Millis(15_100) },
            Input::ClaimUserTurn {},
            Input::Advance { ms: Millis(20_000) },
            Input::ReleaseUserTurn {},
            Input::Advance { ms: Millis(14_900) },
            Input::Advance { ms: Millis(200) },
            Input::Close { drain: false },
            Input::Advance { ms: Millis(20_000) },
        ] {
            run.apply(&input).await;
        }
        run
    });
    let states: Vec<_> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(at, event)| match event {
            SessionEvent::UserStateChanged { new, .. } => Some((at.at_ms, *new)),
            _ => None,
        })
        .collect();
    assert_eq!(
        states.iter().map(|(_, state)| *state).collect::<Vec<_>>(),
        [UserState::Away, UserState::Speaking, UserState::Listening, UserState::Away]
    );
    assert!(states[3].0 >= 50_100, "activity must restart the full away window: {states:?}");
}
