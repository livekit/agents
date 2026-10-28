//! A sink paused for a speech that ends while the false interruption resolves is released.
use lk_agents_fuzz::drive::{Case, Policy, RoomLegs, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput, TurnMode};

// Catches: the false-interruption timer took the pause of a speech already done (its task had
// marked the handle done, its step not yet handled) and resumed nothing, so the speech's own
// release found no pause left and the room's audio output stayed paused: every later speech
// played into it until the close (survey seed 31909 under the task bridge, speech_settles).
#[test]
fn the_next_speech_plays() {
    let setup = Setup {
        vad: true,
        turn: TurnMode::Manual,
        room_legs: Some(RoomLegs { pre_connect: None }),
        ..Setup::default()
    };
    let say = |input| Input::Say {
        text: AgentText("One moment.".into()),
        input,
        allow: None,
        to_ctx: true,
    };
    let inputs = vec![
        Input::Advance { ms: Millis(100) },
        say(SayInput::Audio),
        Input::Advance { ms: Millis(20) },
        // The person starts while the speech has no audio yet: the sink is paused.
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::Advance { ms: Millis(400) },
        // In one instant: the speech ends with nothing played, and the person stops.
        Input::FeedEnd { stream: Ref::new(0) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(100) },
        say(SayInput::Words),
        Input::Advance { ms: Millis(3_000) },
    ];
    for policy in [Policy::BasicPos, Policy::TaskPos] {
        let stuck: Vec<u64> = (0..64)
            .filter(|&schedule| {
                let case = Case {
                    setup: setup.clone(),
                    inputs: inputs.clone(),
                    schedule,
                    random: Vec::new(),
                    policy,
                };
                run(case).failures.iter().any(|f| f.contract == "speech_settles")
            })
            .collect();
        assert!(stuck.is_empty(), "{policy:?}: schedules leaving a speech unplayed: {stuck:?}");
    }
}
