use lk_agents_fuzz::{
    drive::{RealtimeSetup, Setup},
    fakes::RealtimeDelays,
    search,
};

/// Catches protocol bookkeeping failures under voice scheduling that an interface-level fake
/// used to hide: context writes awaiting acknowledgement race tools, cancellation and reconnect.
#[test]
fn seeded_voice_runs_reconcile_against_the_independent_realtime_server() {
    let mut generated = false;
    for seed in 0..16 {
        let setup = Setup {
            stt: false,
            vad: false,
            realtime: Some(RealtimeSetup {
                seed,
                provider: if seed % 4 == 1 {
                    lk_agents_abi::llm::realtime::RealtimeProvider::Xai
                } else {
                    lk_agents_abi::llm::realtime::RealtimeProvider::Openai
                },
                turn_detection: seed % 2 == 0,
                semantic: seed % 4 == 0,
                text_only: seed % 3 == 0 && seed % 4 != 1,
                fallback: seed % 2 == 1,
                tool_calls: Some(50),
                delays: RealtimeDelays {
                    context_ack: Some(250),
                    first_transcription: (seed % 3 == 0).then_some(25_000),
                    transcription: Some(100),
                    transcription_partial: (seed % 4 == 0).then_some(10),
                    ..Default::default()
                },
                ..Default::default()
            }),
            ..Default::default()
        };
        let outcome = search::explore(setup, seed, 500);
        assert!(!outcome.stopped_early, "seed {seed}: {:?}", outcome.ending);
        generated |= outcome.reached.iter().any(|step| step == "Realtime::GenerationStatus");
        assert!(outcome.failures.is_empty(), "seed {seed}: {:#?}", outcome.failures);
    }
    assert!(generated, "the campaign must exercise realtime generations");
}
