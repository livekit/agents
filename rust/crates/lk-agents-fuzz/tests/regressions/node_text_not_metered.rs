//! An `llm_node` that answers text itself made no model call, so no `LLMMetrics` is reported.
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{NodeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis};
use lk_agents_voice::events::SessionEvent;

// Catches: the core metering the node's output rather than the model call: a node returning a
// `str` got an `LLMMetrics` the reference never emits (it meters the `LLMStream`, and such a node
// opens none).
#[test]
fn a_node_answering_text_itself_is_not_metered() {
    let nodes = NodeSetup {
        llm: true,
        tts: false,
        stt: false,
        llm_text: Some("A fixed line."),
        transcription: false,
        realtime_audio: false,
    };
    let setup = Setup { tts: false, nodes: Some(nodes), ..Setup::default() };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(500) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let events = run.record.host_view().events();
    let spoken = events.iter().any(|(_, ev)| {
        matches!(ev, SessionEvent::ConversationItemAdded { item, .. }
            if matches!(item.as_ref(), ChatItem::Message(m) if m.role == Role::Assistant))
    });
    assert!(spoken, "the node's text is the reply");
    let metered = events.iter().any(|(_, ev)| {
        matches!(ev, SessionEvent::MetricsCollected { metrics, .. }
            if matches!(metrics.as_ref(), AgentMetrics::Llm(_)))
    });
    assert!(!metered, "no model call, no LLMMetrics");
}

// A synthesized segment must feed and flush the same transcription node whose completion it
// awaits. Opening a fresh node for each playback phase stranded the original drain (seed 354).
#[test]
fn a_synthesized_segment_settles_its_transcription_node_before_close() {
    use lk_agents_fuzz::drive::{Case, Outcome, finish};
    use lk_agents_fuzz::input::AgentText;
    use lk_agents_fuzz::record::NodeUse;
    use lk_agents_voice::trace::{GenerationStep, SpeechStep, Step};

    let setup = Setup {
        preemptive: true,
        nodes: Some(NodeSetup {
            llm: false,
            tts: true,
            stt: false,
            llm_text: Some("Your order has shipped."),
            transcription: false,
            realtime_audio: false,
        }),
        ..Setup::default()
    };
    let (synthesized_before_authorization, transcriptions, done_before_close, outcome) =
        simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            // Like seed 354, the new agent enables its transcription node at handoff.
            run.apply(&Input::Handoff {
                agent: "lookup_order".into(),
                instructions: AgentText("Report the shipment.".into()),
                tools: vec![],
                turn: None,
            })
            .await;
            run.apply(&Input::Advance { ms: Millis(100) }).await;
            for _ in 0..4 {
                for input in run.releases() {
                    run.apply(&input).await;
                }
                run.apply(&run.release_advance()).await;
            }
            run.apply(&Input::PauseAuthorization {}).await;
            run.apply(&Input::GenerateReply { user_input: None, instructions: None, allow: None })
                .await;
            run.apply(&Input::Advance { ms: Millis(100) }).await;
            let synthesized_before_authorization = run.record.journal().iter().any(|(_, step)| {
                matches!(step, Step::Generation(GenerationStep::TtsRequested { .. }))
            }) && !run
                .record
                .journal()
                .iter()
                .any(|(_, step)| matches!(step, Step::Speech(SpeechStep::Authorized { .. })));
            run.apply(&Input::ResumeAuthorization {}).await;
            // Answer every external dependency, without closing or interrupting the speech.
            for _ in 0..20 {
                for input in run.releases() {
                    run.apply(&input).await;
                }
                run.apply(&run.release_advance()).await;
            }
            let transcriptions = run
                .record
                .inference_view()
                .nodes()
                .filter(|(_, _, node)| *node == NodeUse::Transcription)
                .count();
            let speeches = run.record.host_view().speeches();
            let done_before_close = !speeches.is_empty() && speeches.iter().all(|s| s.is_done());
            let case = Case { setup, ..Case::default() };
            let outcome = finish(run, Outcome { case, ..Outcome::default() }).await;
            (synthesized_before_authorization, transcriptions, done_before_close, outcome)
        });
    assert!(synthesized_before_authorization, "the reply must synthesize into deferred text");
    assert_eq!(transcriptions, 1, "one transcription invocation must actually process the segment");
    assert!(done_before_close, "the node drain must finish before Close: {:?}", outcome.failures);
    assert!(outcome.reached.iter().any(|step| step == "Playout::PlaybackFinished"));
    assert!(!outcome.stopped_early, "{outcome:?}");
    assert!(outcome.failures.is_empty(), "{:?}", outcome.failures);
}
