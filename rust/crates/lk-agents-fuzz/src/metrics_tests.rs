//! Metrics regression tests.

mod llm_tokens_per_second {
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
                    ChatItem::Message(m) if m.role == Role::Assistant => {
                        Some(m.metrics.llm_node_tps)
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect();
        // Each chunk's wall stamp is a microsecond past the last (wall time strictly increases), and
        // the model streams a word a chunk, so the window reads a hair over 0.5 s.
        let expected = COMPLETION_TOKENS as f64 / 0.5;
        assert!(
            matches!(rates.as_slice(), [Some(rate)] if (rate - expected).abs() < 1e-3),
            "{rates:?}, expected {expected}"
        );
    }
}

mod model_metrics_reported {
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
        let llm =
            metrics(&setup, &inputs).iter().filter(|m| matches!(m, AgentMetrics::Llm(_))).count();
        assert_eq!(llm, 1);
    }
}

mod node_text_not_metered {
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
                run.apply(&Input::GenerateReply {
                    user_input: None,
                    instructions: None,
                    allow: None,
                })
                .await;
                run.apply(&Input::Advance { ms: Millis(100) }).await;
                let synthesized_before_authorization =
                    run.record.journal().iter().any(|(_, step)| {
                        matches!(step, Step::Generation(GenerationStep::TtsRequested { .. }))
                    }) && !run.record.journal().iter().any(|(_, step)| {
                        matches!(step, Step::Speech(SpeechStep::Authorized { .. }))
                    });
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
                let done_before_close =
                    !speeches.is_empty() && speeches.iter().all(|s| s.is_done());
                let case = Case { setup, ..Case::default() };
                let outcome = finish(run, Outcome { case, ..Outcome::default() }).await;
                (synthesized_before_authorization, transcriptions, done_before_close, outcome)
            });
        assert!(synthesized_before_authorization, "the reply must synthesize into deferred text");
        assert_eq!(
            transcriptions, 1,
            "one transcription invocation must actually process the segment"
        );
        assert!(
            done_before_close,
            "the node drain must finish before Close: {:?}",
            outcome.failures
        );
        assert!(outcome.reached.iter().any(|step| step == "Playout::PlaybackFinished"));
        assert!(!outcome.stopped_early, "{outcome:?}");
        assert!(outcome.failures.is_empty(), "{:?}", outcome.failures);
    }
}

mod preemptive_reply_latency {
    //! An adopted preemptive reply measures its end-to-end latency from the turn it answers.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, TurnMode, UserText};
    use lk_agents_voice::events::SessionEvent;

    // Catches: a preemptive reply started before its user turn was committed, so it never learned
    // the turn's metrics and its message had no `e2e_latency`; with preemptive generation on (the
    // default) no reply on a call had one. The reference writes the turn's metrics onto the
    // preemptive generation's user message when it adopts it.
    #[test]
    fn an_adopted_preemptive_reply_has_its_latency() {
        let setup = Setup {
            stt: true,
            vad: true,
            preemptive: true,
            turn: TurnMode::Stt,
            ..Setup::default()
        };
        let words = || UserText("what's my balance".into());
        let inputs = [
            Input::Advance { ms: Millis(20) },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(600) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(300) },
            Input::SttPreflight { utterance: Ref::new(0), text: words(), speaker: None },
            Input::Advance { ms: Millis(50) },
            Input::SttFinal { utterance: Ref::new(0), text: words(), speaker: None },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(1000) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("It's forty dollars.".into()) },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::TtsRelease { tts: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let replies: Vec<_> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::Message(m) if m.role == Role::Assistant => Some(m.metrics.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(replies.len(), 1, "one reply");
        assert!(replies[0].e2e_latency.is_some(), "{:?}", replies[0]);
    }
}

mod turn_metrics_name_models {
    //! A turn's metrics name the models that produced it.
    use lk_agents_abi::{ChatItem, MetricsReport, Role};
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, TurnMode, UserText};
    use lk_agents_voice::events::SessionEvent;

    // Catches: the user and assistant messages committed without `stt_metadata`, `llm_metadata` and
    // `tts_metadata`. The reference stamps each with the provider's `metrics_metadata`
    // (`_init_metrics_from_end_of_turn`, `_pipeline_reply_task_impl`); on a real call every chat item,
    // and the turn metrics exported from them, had no model name.
    #[test]
    fn a_spoken_turn_and_its_reply_name_their_models() {
        let setup = Setup { vad: true, stt: true, turn: TurnMode::Stt, ..Setup::default() };
        let inputs = [
            Input::Advance { ms: Millis(20) },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(300) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("how much is it".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(2_000) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("It's forty dollars.".into()) },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let of = |role: Role| -> Vec<MetricsReport> {
            run.record
                .host_view()
                .events()
                .into_iter()
                .filter_map(|(_, ev)| match ev {
                    SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                        ChatItem::Message(m) if m.role == role => Some(m.metrics.clone()),
                        _ => None,
                    },
                    _ => None,
                })
                .collect()
        };
        let (users, replies) = (of(Role::User), of(Role::Assistant));
        let ([user], [reply]) = (users.as_slice(), replies.as_slice()) else {
            panic!("one turn and one reply: {users:?} {replies:?}");
        };
        assert!(user.stt_metadata.is_some(), "{user:?}");
        assert!(reply.llm_metadata.is_some() && reply.tts_metadata.is_some(), "{reply:?}");
    }
}

mod user_turn_metrics {
    //! A spoken user turn is committed with its end-of-turn metrics.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref, TurnMode, UserText};
    use lk_agents_voice::events::SessionEvent;

    // Catches: the user message built with an empty `MetricsReport`, though the turn's anchors and
    // delays were computed for the EOU metrics. The reference's `_init_metrics_from_end_of_turn`
    // stamps them on the message, and the reply's `e2e_latency` is measured from its
    // `stopped_speaking_at`: with them gone no reply on a call had an end-to-end latency, and the
    // session report showed no turn timings.
    #[test]
    fn a_spoken_turn_carries_its_metrics() {
        let setup = Setup { vad: true, stt: true, turn: TurnMode::Stt, ..Setup::default() };
        let inputs = [
            Input::Advance { ms: Millis(20) },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(300) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("hello there".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(2000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let users: Vec<_> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::Message(m) if m.role == Role::User => Some(m.metrics.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(users.len(), 1, "one user turn");
        let m = &users[0];
        assert!(m.started_speaking_at.is_some() && m.stopped_speaking_at.is_some(), "{m:?}");
        assert!(m.end_of_turn_delay.is_some() && m.transcription_delay.is_some(), "{m:?}");
    }
}
