//! The calls the session makes on the agent's behalf are metered like its replies: the VAD's
//! inferences, and AMD's classification.
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_fuzz::drive::{AmdAnswer, AmdSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{
    AmdInput, Confidence, Input, Millis, Ref, ToolArgs, TurnMode, UserText,
};
use lk_agents_voice::events::SessionEvent;

fn metrics(setup: &Setup, inputs: &[Input]) -> Vec<AgentMetrics> {
    let run = simulate(async {
        let mut run = Run::start(setup, 0, Default::default()).await;
        for input in inputs {
            run.apply(input).await;
        }
        run
    });
    run.record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, ev)| match ev {
            SessionEvent::MetricsCollected { metrics, .. } => Some(metrics.as_ref().clone()),
            _ => None,
        })
        .collect()
}

// Catches: no `VADMetrics` at all. The reference's `VADStream._metrics_monitor_task` reports one
// per `1 / update_interval` inferences (a second of audio), as `parity.py update_endpointing`
// shows.
#[test]
fn a_second_of_vad_inferences_is_one_record() {
    let setup = Setup { vad: true, ..Setup::default() };
    let inference = Input::VadInference {
        vad: Ref::new(0),
        probability: Confidence(10),
        speaking: false,
        duration: Millis(32),
    };
    let mut inputs = vec![inference; 40];
    inputs.push(Input::Advance { ms: Millis(20) });
    let vad: Vec<_> = metrics(&setup, &inputs)
        .into_iter()
        .filter_map(|m| match m {
            AgentMetrics::Vad(v) => Some(v.inference_count),
            _ => None,
        })
        .collect();
    // 1 / 0.032 s = 31.25: the 32nd inference reports, the next 8 wait for theirs.
    assert_eq!(vad, vec![32]);
}

// Catches: AMD's classification left unmetered. The reference's classifier calls the session's
// LLM, whose `LLMStream` reports an `LLMMetrics` like any reply's (`parity.py amd_machine`).
#[test]
fn an_amd_classification_is_metered() {
    let setup = Setup {
        vad: true,
        stt: true,
        turn: TurnMode::Stt,
        amd: Some(AmdSetup { answer: AmdAnswer::Track, ..AmdSetup::default() }),
        ..Setup::default()
    };
    let inputs = [
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(300) },
        Input::SttFinal {
            utterance: Ref::new(0),
            text: UserText("hi you have reached the voicemail of theo".into()),
            speaker: None,
        },
        Input::SttEnd { utterance: Ref::new(0) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(20) },
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "save_prediction".into(),
            args: ToolArgs(r#"{"label":"machine-vm"}"#.into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(3000) },
        Input::Amd { input: AmdInput::Execute },
        Input::Advance { ms: Millis(20) },
    ];
    let llm = metrics(&setup, &inputs).iter().filter(|m| matches!(m, AgentMetrics::Llm(_))).count();
    assert_eq!(llm, 1);
}
