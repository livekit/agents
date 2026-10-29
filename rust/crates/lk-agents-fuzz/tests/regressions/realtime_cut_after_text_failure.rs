//! A realtime reply cut while its transcript failed still truncates what the server holds.
use lk_agents_abi::RealtimeCommandKind;
use lk_agents_fuzz::drive::{NodeSetup, RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis};
use lk_agents_fuzz::record::Kind;
use lk_agents_voice::trace::{RealtimeStep, Step};

// Catches: the realtime step returning its text output's failure before telling the server what
// was heard, so after a cut no `conversation.item.truncate` went out and the server kept audio
// the person never heard (`realtime_server_holds_what_was_heard`, campaign seed 32802).
#[test]
fn a_cut_reply_is_truncated_even_when_its_transcript_failed() {
    let setup = Setup {
        realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
        nodes: Some(NodeSetup {
            llm: false,
            tts: false,
            stt: false,
            llm_text: None,
            transcription: true,
            realtime_audio: false,
        }),
        text_output_raises: true,
        ..Setup::default()
    };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        run.apply(&Input::GenerateReply { user_input: None, instructions: None, allow: None })
            .await;
        for _ in 0..100 {
            run.apply(&Input::Advance { ms: Millis(20) }).await;
            if !run.record.live().of(Kind::Segment).is_empty() {
                break;
            }
        }
        assert!(!run.record.live().of(Kind::Segment).is_empty(), "the reply never played");
        run.apply(&Input::Interrupt { force: false, wait: false }).await;
        run.apply(&Input::Advance { ms: Millis(2_000) }).await;
        run
    });
    let truncated = run.record.journal().iter().any(|(_, step)| {
        matches!(
            step,
            Step::Realtime(RealtimeStep::Sent { kind: RealtimeCommandKind::Truncate, .. })
        )
    });
    assert!(truncated, "the server must be told what the person heard of the cut reply");
}
