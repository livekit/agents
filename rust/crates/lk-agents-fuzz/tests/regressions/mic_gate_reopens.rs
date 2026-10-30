//! The microphone gate the agent's speech closed reopens before the next frame.
use lk_agents_fuzz::drive::{Case, Policy, Setup, run};
use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, SayInput};

// Catches: the gate reopening through the owner's queue, behind frames already queued, so frames
// pushed as it reopened were silenced (`forwarded_while_enabled`, campaign seed 27243; a45633b37).
// The gate's effects settle in their batch, before another frame is forwarded. The race is the
// scheduler's, so 32 schedules of each queue policy run.
#[test]
fn the_microphone_is_heard_once_the_gate_reopens() {
    let mic = |chunks| Input::MicFrames { mic: Ref::new(0), chunks: Count(chunks) };
    let inputs = vec![
        Input::Say {
            text: AgentText("Your balance is due on Friday, and the next one in March.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(20) },
        // The agent starts speaking: the recogniser's gate closes until the echo canceller warms.
        Input::PlayoutStarted { segment: Ref::new(0) },
        mic(10),
        Input::Advance { ms: Millis(100) },
        mic(10),
        Input::Advance { ms: Millis(2_900) },
        // The gate reopens, and the person's frames arrive in the same instant.
        mic(10),
        Input::Advance { ms: Millis(100) },
        mic(10),
        Input::Advance { ms: Millis(100) },
    ];
    for policy in [Policy::Arrival, Policy::BasicPos, Policy::TaskBridge] {
        let silenced: Vec<u64> = (0..32)
            .filter(|&schedule| {
                let case = Case {
                    setup: Setup { stt: true, interruption: true, ..Setup::default() },
                    inputs: inputs.clone(),
                    schedule,
                    random: Vec::new(),
                    policy,
                };
                run(case).failures.iter().any(|f| f.contract == "forwarded_while_enabled")
            })
            .collect();
        assert!(
            silenced.is_empty(),
            "{policy:?}: schedules silencing the microphone: {silenced:?}"
        );
    }
}
