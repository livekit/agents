//! A final that cuts a paused speech asks for no preemptive generation: the speech was live.
use lk_agents_fuzz::drive::{RoomLegs, Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Confidence, Count, Input, Millis, Ref, SayInput, UserText};
use lk_agents_voice::{events::SessionEvent, trace::Step};

// Catches: the final's interruption of the paused speech landed before its preemptive generation
// was decided, so a preemptive reply was planned (an extra LLM request) for a turn the agent was
// still talking over. The reference's `on_final_transcript` cuts the pause in a task, after
// `on_preemptive_generation` refused for the live speech; found by
// `scripts/parity.py vad_interruption`.
#[test]
fn a_final_over_a_paused_speech_plans_no_preemptive_reply() {
    let setup = Setup {
        vad: true,
        stt: true,
        preemptive: true,
        pausable_output: true,
        room_legs: Some(RoomLegs { pre_connect: None }),
        ..Setup::default()
    };
    let mut inputs = vec![
        Input::Advance { ms: Millis(100) },
        Input::Say {
            text: AgentText("A long answer.".into()),
            input: SayInput::Audio,
            allow: None,
            to_ctx: true,
        },
    ];
    // Four seconds of audio, fed as it plays: past the echo canceller's warm-up.
    for _ in 0..8 {
        inputs.push(Input::FeedAudio { stream: Ref::new(0), chunks: Count(50) });
        inputs.push(Input::Advance { ms: Millis(450) });
    }
    inputs.extend([
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::VadInference {
            vad: Ref::new(0),
            probability: Confidence(90),
            speaking: true,
            duration: Millis(800),
        },
        Input::Advance { ms: Millis(50) },
        Input::SttStart { stt: Ref::new(0) },
        Input::SttFinal {
            utterance: Ref::new(0),
            text: UserText("wait stop".into()),
            speaker: None,
        },
        Input::Advance { ms: Millis(50) },
    ]);
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let created = run
        .record
        .journal()
        .iter()
        .filter(|(_, step)| matches!(step, Step::Emitted(SessionEvent::SpeechCreated { .. })))
        .count();
    assert_eq!(created, 1, "only the say: the user is still speaking, no reply is planned yet");
}
