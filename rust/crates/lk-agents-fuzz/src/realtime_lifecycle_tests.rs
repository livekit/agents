//! Realtime lifecycle regression tests.

mod realtime_close_interrupts {
    //! A close tells the realtime socket to stop even when no generation is open.
    use lk_agents_abi::RealtimeCommandKind;
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};
    use lk_agents_voice::trace::{RealtimeStep, Step};

    // Catches: the session-level interrupt (which a close without drain runs) telling the socket only
    // about generations it had open. The reference calls `rt_session.interrupt()` unconditionally, so
    // a reply that had ended, or a response the server started and had not announced, is still told.
    #[test]
    fn a_close_interrupts_the_socket_with_no_generation_open() {
        let setup = Setup {
            realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
            ..Setup::default()
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::GenerationDone },
            Input::Advance { ms: Millis(2_000) },
            Input::Close { drain: false },
            Input::Advance { ms: Millis(20) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let interrupts = run
            .record
            .journal()
            .iter()
            .filter(|(_, step)| {
                matches!(
                    step,
                    Step::Realtime(RealtimeStep::Sent { kind: RealtimeCommandKind::Interrupt, .. })
                )
            })
            .count();
        assert_eq!(interrupts, 1);
    }
}

mod realtime_cut_after_text_failure {
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
}

mod realtime_items_filled {
    //! A realtime server's empty records of its items stay out of history; the words that fill
    //! them reach it, once, under the server's ids.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_core::tts::markup::ChatMessageExt;
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{Count, Input, Millis, Ref, RtEvent};
    use lk_agents_voice::events::SessionEvent;

    // Catches: `remote_item_added` committed into history as it came, empty, and the transcript and
    // the reply that fill those items never told to the host: the user's final transcript upserted
    // the empty record under its id (a replacement, never announced) and the reply was committed
    // under an id of its own, beside its empty placeholder. On a real call history held an empty
    // message per turn and no user words. The reference keeps the server's items as placeholders in
    // the agent's context and commits the transcript and the message under the server's ids.
    #[test]
    fn server_items_reach_history_filled() {
        let setup = Setup {
            realtime: Some(RealtimeSetup {
                turn_detection: false,
                tool_calls: Some(0),
                ..RealtimeSetup::default()
            }),
            ..Setup::default()
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::GenerationDone },
            Input::Advance { ms: Millis(500) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            // The person speaks: a commit of an empty buffer is an error on a real server.
            Input::MicFrames { mic: Ref::new(0), chunks: Count(30) },
            Input::Advance { ms: Millis(300) },
            Input::CommitTurn { skip_reply: false },
            Input::Advance { ms: Millis(1_000) },
            Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::GenerationDone },
            Input::Advance { ms: Millis(500) },
            Input::PlayoutFinished { segment: Ref::new(1) },
            Input::Advance { ms: Millis(500) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let messages: Vec<(Role, String, String)> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::Message(m) => {
                        Some((m.role, m.id.clone(), m.text_content().unwrap_or_default()))
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert!(messages.iter().all(|(_, _, text)| !text.is_empty()), "{messages:#?}");
        let user: Vec<_> = messages.iter().filter(|(role, ..)| *role == Role::User).collect();
        assert_eq!(user.len(), 1, "{messages:#?}");
        assert!(user[0].1.starts_with("item_user_"), "{messages:#?}");
        let replies: Vec<_> =
            messages.iter().filter(|(role, ..)| *role == Role::Assistant).collect();
        assert_eq!(replies.len(), 2, "{messages:#?}");
        assert!(replies.iter().all(|(_, id, _)| id.starts_with("item_msg_")), "{messages:#?}");
    }
}

mod realtime_late_answer_interrupted {
    //! A reply the server creates as the client's request bound runs out is interrupted, not left
    //! holding the server.
    use lk_agents_abi::RealtimeCommandKind;
    use lk_agents_fuzz::drive::{Outcome, RealtimeSetup, Run, Setup, finish, simulate};
    use lk_agents_fuzz::fakes::RealtimeDelays;
    use lk_agents_fuzz::input::{Input, Millis};
    use lk_agents_voice::agent_session::effects::Timer;
    use lk_agents_voice::trace::{CoreStep, RealtimeStep, Step, TimerName};

    // Catches: the timeout's cancel crossing the server's creation on the wire. The late generation
    // was discarded without an interrupt and held the server, which refused the next
    // `response.create` (survey seeds 401 and 50; eca089161).
    #[test]
    fn a_reply_created_as_its_request_timed_out_does_not_hold_the_server() {
        // The server answers exactly at the request's 10 s bound.
        let delays = RealtimeDelays { created: Some(10_000), ..RealtimeDelays::default() };
        let setup = Setup {
            realtime: Some(RealtimeSetup {
                turn_detection: false,
                delays,
                ..RealtimeSetup::default()
            }),
            ..Setup::default()
        };
        let reply = || Input::GenerateReply { user_input: None, instructions: None, allow: None };
        let inputs = [reply(), Input::Advance { ms: Millis(10_100) }, reply()];
        let (outcome, journal) = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            let journal: Vec<_> = run.record.journal().to_vec();
            (finish(run, Outcome::default()).await, journal)
        });
        let (fired, at) = journal
            .iter()
            .enumerate()
            .find_map(|(i, (at, s))| {
                matches!(
                    s,
                    Step::Core(CoreStep::TimerFired(
                        TimerName::Session(Timer::RealtimeRequest(_)),
                        _
                    ))
                )
                .then_some((i, at.at_ms))
            })
            .expect("the scenario: the request times out");
        let sent = |kind: RealtimeCommandKind| {
            journal[fired..].iter().position(|(when, s)| {
                when.at_ms == at
                    && matches!(s, Step::Realtime(RealtimeStep::Sent { kind: k, .. }) if *k == kind)
            })
        };
        let cancel =
            sent(RealtimeCommandKind::CancelReply).expect("the scenario: the timeout cancels");
        let interrupt =
            sent(RealtimeCommandKind::Interrupt).expect("the late reply is interrupted");
        assert!(cancel < interrupt, "the reply landed after the cancel crossed it");
        assert!(
            !outcome.failures.iter().any(|f| f.contract == "realtime_server_accepts_requests"),
            "{:#?}",
            outcome.failures
        );
    }
}

mod realtime_waiter_journalled {
    //! A realtime request settled before the server answered ends its handle through the journal.
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};
    use lk_agents_fuzz::record::Recording;
    use lk_agents_voice::trace::{InterruptCause, InterruptStep, Step};

    fn realtime(inputs: &[Input]) -> Recording {
        let setup = Setup {
            realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
            ..Setup::default()
        };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in inputs {
                run.apply(input).await;
            }
            run
        });
        run.record
    }

    fn interrupted(r: &Recording) -> Vec<(u64, InterruptCause)> {
        r.journal()
            .iter()
            .filter_map(|(_, s)| match s {
                Step::Interrupt(InterruptStep::Interrupted { speech, cause, .. }) => {
                    Some((speech.0, *cause))
                }
                _ => None,
            })
            .collect()
    }

    // Catches: a `generate_reply()` still waiting on the server when an interruption cancels the
    // server's turn, answered with a handle that reports itself interrupted while the journal never
    // said so (`interrupted_handle_is_journalled`, survey seed 14).
    #[test]
    fn an_interrupted_request_is_journalled_with_its_cause() {
        let r = realtime(&[
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Interrupt { force: true, wait: false },
            Input::Advance { ms: Millis(20) },
        ]);
        let speeches = r.host_view().speeches();
        let [speech] = &speeches[..] else { panic!("one handle, got {}", speeches.len()) };
        assert!(speech.is_interrupted());
        assert_eq!(interrupted(&r), [(speech.id, InterruptCause::Host)]);
    }

    // Catches: a request made after the socket ended answered with an interrupted handle no
    // interruption journalled. The reference's pending `generate_reply()` fails with a
    // `RealtimeError` there, which `_realtime_reply_task` logs, as it does a timeout. A real session
    // reconnects; a server that never takes the connection again ends it with the session's error
    // close, and the request with it, journalled.
    #[test]
    fn a_request_after_the_socket_ended_fails() {
        let r = realtime(&[
            Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::Disconnected },
            Input::Advance { ms: Millis(20) },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(30_000) },
        ]);
        let speeches = r.host_view().speeches();
        let [speech] = &speeches[..] else { panic!("one handle, got {}", speeches.len()) };
        assert!(speech.is_interrupted());
        assert_eq!(interrupted(&r), [(speech.id, InterruptCause::Closing)]);
    }
}
