//! Under STT turn detection, the STT's end of speech arms the resume of a speech an interim paused.
use lk_agents_fuzz::drive::{RoomLegs, Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, SayInput, TurnMode, UserText};
use lk_agents_voice::{events::SessionEvent, trace::Step};

// Catches: only a VAD's end of speech armed the false-interruption timer, so with an STT and no
// VAD an interim that paused the agent left it paused for good (and the session's close waited on
// it). The reference's STT `END_OF_SPEECH` is `on_end_of_speech(None)` in "stt" mode, which starts
// `_start_false_interruption_timer`; found by `scripts/parity.py false_interruption`.
#[test]
fn an_stt_end_of_speech_resumes_the_paused_speech() {
    let setup = Setup {
        stt: true,
        pausable_output: true,
        turn: TurnMode::Stt,
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
        Input::SttStart { stt: Ref::new(0) },
        Input::SttInterim { utterance: Ref::new(0), text: UserText("uh".into()), speaker: None },
        Input::SttEnd { utterance: Ref::new(0) },
        Input::Advance { ms: Millis(5_000) },
    ]);
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let resumed: Vec<bool> = run
        .record
        .journal()
        .iter()
        .filter_map(|(_, step)| match step {
            Step::Emitted(SessionEvent::AgentFalseInterruption { resumed }) => Some(*resumed),
            _ => None,
        })
        .collect();
    assert_eq!(resumed, [true], "one false interruption, resumed");
}
