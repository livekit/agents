//! Audio regression tests.

mod background_audio {
    //! `BackgroundAudioPlayer` over FakeRoom, in what the DST's player (an ambient stream and the
    //! thinking sound) does not reach: refused plays and a close cancelled while a stream retires.
    use lk_agents_abi::AudioFrame;
    use lk_agents_abi::errors::ProviderError;
    use lk_agents_abi::voice::background_audio::BackgroundAudioPlayer as _;
    use lk_agents_abi::voice::rtc::LocalParticipant as _;
    use lk_agents_fuzz::fakes::room::{FakeRoom, FakeRtc};
    use lk_agents_runtime::net::task::TokioSpawner;
    use lk_agents_runtime::room_io::FixedRoom;
    use lk_agents_runtime::rtc::events::RoomEventBus;
    use lk_agents_runtime::voice::background_audio::{
        BackgroundAudio, BackgroundAudioPlayer, FrameStream, LoopingStreamError, PlaybackError,
        Sound, SoundSource,
    };
    use lk_agents_runtime::voice::session::Timebase;
    use std::future::Future as _;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    fn file(path: &str, probability: f64) -> Sound {
        Sound {
            source: SoundSource::File(path.to_owned()),
            volume: 1.0,
            probability,
            fade_in: 0.0,
            fade_out: 0.0,
        }
    }

    fn player(time: &Timebase) -> BackgroundAudio {
        BackgroundAudio(Arc::new(BackgroundAudioPlayer::new(
            48_000,
            1,
            4_800,
            200,
            time.clock(),
            Arc::new(TokioSpawner::current()),
        )))
    }

    /// A player started on a fake room with no ambient sound, and the room.
    async fn started(name: &str, ambient: Vec<Sound>) -> (BackgroundAudio, Arc<FakeRoom>) {
        let (rtc, clock) = FakeRtc::with_virtual_clock(RoomEventBus::port);
        let room = rtc.room(name);
        std::mem::forget(rtc);
        let fixed =
            FixedRoom { rtc: Some(room.clone()), time: Timebase::from_virtual(clock), pool: None };
        let player = player(&fixed.time);
        player.start(Arc::new(fixed), 0.5, ambient, true, None).await.unwrap();
        (player, room)
    }

    /// A host stream of short frames.
    struct Frames(Mutex<u32>);

    impl FrameStream for Frames {
        fn next(
            &self,
            _cx: &lk_agents_abi::telemetry::Cx,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<AudioFrame>, ProviderError>> {
            let mut left = self.0.lock().unwrap();
            let frame = (*left > 0).then(|| {
                *left -= 1;
                AudioFrame::new(1000i16.to_le_bytes().repeat(480), 48_000, 1, 480)
            });
            Box::pin(std::future::ready(Ok(frame)))
        }
        fn cancel_pending(&self) {}
        fn close(&self, _cx: &lk_agents_abi::telemetry::Cx) -> lk_agents_abi::BoxFuture<'_, ()> {
            Box::pin(std::future::ready(()))
        }
    }

    // Catches: a sound played before `start()` or after `aclose()` queued where nothing mixes it, and
    // a looping iterator accepted (it can only play once).
    #[tokio::test(start_paused = true)]
    async fn play_refuses_what_cannot_play() {
        let idle = player(&Timebase::from_virtual(Arc::new(lk_agents_abi::VirtualClock::new())));
        assert!(matches!(
            idle.0.play(vec![file("a.ogg", 1.0)], true, false, false, 0.5),
            Err(PlaybackError::NotStarted)
        ));

        let (player, _room) = started("background", Vec::new()).await;
        let stream =
            Sound { source: SoundSource::Stream(Arc::new(Frames(Mutex::new(1)))), ..file("", 1.0) };
        let looping = player.0.play(vec![stream], true, true, false, 0.5);
        assert!(matches!(looping, Err(PlaybackError::LoopingStream(_))));
        assert!(LoopingStreamError.to_string().contains("loop=False"));
        // The draw picked silence: the handle is born done.
        let silent = player.0.play(vec![file("a.ogg", 0.0)], true, false, false, 0.5).unwrap();
        assert!(silent.done());

        player.0.close().await;
        assert!(player.0.closed());
        assert!(matches!(
            player.0.play(vec![file("a.ogg", 1.0)], false, false, false, 0.5),
            Err(PlaybackError::Closed)
        ));
    }

    /// A stream whose close waits for the test, counting entries and cancellations.
    #[derive(Default)]
    struct DelayedClose {
        entered: tokio::sync::Notify,
        release: tokio::sync::Notify,
        finished: AtomicBool,
        calls: AtomicUsize,
        cancelled: AtomicUsize,
    }

    impl FrameStream for DelayedClose {
        fn next(
            &self,
            _cx: &lk_agents_abi::telemetry::Cx,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<AudioFrame>, ProviderError>> {
            Box::pin(async { Ok(None) })
        }

        fn cancel_pending(&self) {}

        fn close(&self, _cx: &lk_agents_abi::telemetry::Cx) -> lk_agents_abi::BoxFuture<'_, ()> {
            Box::pin(async move {
                struct Closing<'a>(&'a DelayedClose, bool);
                impl Drop for Closing<'_> {
                    fn drop(&mut self) {
                        if !self.1 {
                            self.0.cancelled.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                }
                self.calls.fetch_add(1, Ordering::SeqCst);
                let mut closing = Closing(self, false);
                self.entered.notify_one();
                self.release.notified().await;
                closing.1 = true;
                self.finished.store(true, Ordering::SeqCst);
            })
        }
    }

    #[tokio::test(start_paused = true)]
    async fn stopping_during_stream_retirement_keeps_cleanup_owned_after_a_waiter_is_cancelled() {
        let stream = Arc::new(DelayedClose::default());
        let sound = Sound { source: SoundSource::Stream(stream.clone()), ..file("", 1.0) };
        let (player, room) = started("background-retirement-cancellation", vec![sound]).await;
        stream.entered.notified().await;
        let mut closing = Box::pin(player.0.close());
        std::future::poll_fn(|cx| {
            assert!(closing.as_mut().poll(cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        drop(closing);
        // Retirement stays owned after the close waiter is cancelled.
        stream.release.notify_one();
        tokio::task::yield_now().await;
        player.0.close().await;
        let finished = stream.finished.load(Ordering::SeqCst);
        let cancelled = stream.cancelled.load(Ordering::SeqCst);
        let publications = room.local_participant().track_publications().unwrap();
        // Leave no published track behind if the regression assertion fails.
        for publication in &publications {
            room.local_participant()
                .unpublish_track(&lk_agents_abi::telemetry::Cx::NONE, publication.sid())
                .await
                .unwrap();
        }
        assert!(finished, "stopping the pump abandoned an in-flight stream close");
        assert_eq!(cancelled, 0, "retirement cleanup was cancelled by the pump's stop");
        assert_eq!(stream.calls.load(Ordering::SeqCst), 1, "cleanup was restarted");
        assert!(publications.is_empty(), "cancelling a close waiter orphaned the output track");
    }
}

mod failed_synthesis_item {
    //! A reply whose voice failed mid-step commits what was heard of it, marked cut short.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_core::tts::markup::ChatMessageExt;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
    use lk_agents_voice::{events::SessionEvent, trace::Step};

    // Catches: a TTS failing after its first chunk played committed the reply as complete. The
    // reference commits the synchroniser's words up to the played position and marks the message
    // interrupted, so the LLM knows it was cut short (`_pipeline_reply_task_impl`,
    // `interrupted=speech_handle.interrupted or any(out.tts_failed ...)`, agent_activity.py:4176).
    #[test]
    fn a_failed_synthesis_commits_what_was_heard_as_interrupted() {
        let text = |t: &str| AgentText(t.into());
        let setup = Setup { sync_transcription: true, ..Setup::default() };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            let inputs = [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(20) },
                Input::LlmText { chat: Ref::new(0), text: text("Hello there.") },
                Input::Advance { ms: Millis(20) },
                Input::TtsHold { tts: Ref::new(0) },
                Input::LlmText { chat: Ref::new(0), text: text(" How are you?") },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(20) },
                Input::TtsFail { tts: Ref::new(0) },
                Input::Advance { ms: Millis(20) },
                Input::PlayoutFinished { segment: Ref::new(0) },
                Input::Advance { ms: Millis(100) },
            ];
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let messages: Vec<_> = run
            .record
            .journal()
            .iter()
            .filter_map(|(_, step)| match step {
                Step::Emitted(SessionEvent::ConversationItemAdded { item, .. }) => match &**item {
                    ChatItem::Message(msg) if msg.role == Role::Assistant => Some(msg.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        let [msg] = &messages[..] else { panic!("one reply, got {messages:?}") };
        assert!(msg.interrupted, "the reply is marked cut short");
        // The synchroniser heard the played segment through: its words are all of it.
        assert_eq!(msg.text_content().as_deref(), Some("Hello there. How are you?"));
    }
}

mod failed_synthesis_plays_out {
    //! Audio synthesised before the voice failed plays out before the speech ends.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};

    // Catches: a TTS failing after its first chunk left that chunk playing unwaited, so the speech
    // completed before any of its audio was heard (`synthesized_speech_is_played`, survey seed 122).
    // The reference's `forward_generation` waits for the playout whether or not the TTS failed.
    #[test]
    fn a_failed_synthesis_waits_for_what_it_made() {
        let text = |t: &str| AgentText(t.into());
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            let opening = [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(20) },
                Input::LlmText { chat: Ref::new(0), text: text("Hello there.") },
                Input::Advance { ms: Millis(20) },
                Input::TtsHold { tts: Ref::new(0) },
                Input::LlmText { chat: Ref::new(0), text: text(" How are you?") },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(20) },
                Input::TtsFail { tts: Ref::new(0) },
                Input::Advance { ms: Millis(100) },
            ];
            for input in &opening {
                run.apply(input).await;
            }
            assert!(run.record.applied().iter().all(|a| a.accepted), "every input landed");
            let speeches = run.record.host_view().speeches();
            let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
            assert!(!speech.is_done(), "the speech ended before its audio played");
            for input in
                [Input::PlayoutFinished { segment: Ref::new(0) }, Input::Advance { ms: Millis(20) }]
            {
                run.apply(&input).await;
            }
            run
        });
        let speeches = run.record.host_view().speeches();
        assert!(speeches[0].completed_normally(), "the playout released the speech");
    }
}

mod filler_refused_on_realtime {
    //! A filler's say on a realtime model that cannot say is refused, as the reference's `say`.
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
    use lk_agents_voice::trace::{FillerEnd, GenerationStep, Step};

    // Catches: the filler's text played as a pipeline speech with no voice on a realtime session
    // whose model has no `say` (the DST agent's parity finding). The reference's `session.say` raises
    // there ("without a TTS model or a RealtimeSession that supports say()"), which ends the filler
    // with nothing spoken.
    #[test]
    fn a_filler_on_a_realtime_model_without_say_is_refused() {
        let setup = Setup {
            realtime: Some(RealtimeSetup {
                turn_detection: false,
                tool_calls: Some(100),
                ..RealtimeSetup::default()
            }),
            tools: vec!["lookup".into()],
            // No voice: with one, the TTS speaks the filler.
            tts: false,
            ..Setup::default()
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(10_000) },
            Input::FillerStart {
                tool: Ref::new(0),
                speech: Ref::new(0),
                delay: Millis(0),
                interval: None,
                text: Some(AgentText("One moment.".into())),
            },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                crate::playout::apply_with_playout(&mut run, input).await;
            }
            run
        });
        let journal = run.record.journal();
        let fired = journal
            .iter()
            .any(|(_, step)| matches!(step, Step::Generation(GenerationStep::FillerFired { .. })));
        let refused = journal.iter().any(|(_, step)| {
            matches!(
                step,
                Step::Generation(GenerationStep::FillerStopped { cause: FillerEnd::Refused, .. })
            )
        });
        assert!(!fired && refused, "fired {fired}, refused {refused}");
    }
}

mod filler_source {
    //! A filler that ends while the host is still sourcing its words finishes that hook too.
    use lk_agents_fuzz::drive::{Case, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};

    // Catches: `end_filler` dropping the pending `FillerSource` hook without journalling
    // `HookFinished`. The filler fires inside the `function_tools_executed` bound, the speech ends
    // when that bound passes, and the host's answer then names a hook the driver no longer knows: the
    // hook stayed open in the journal and the release phase never went quiet (soak seed 2).
    #[test]
    fn a_filler_ended_mid_source_finishes_its_hook() {
        let inputs = vec![
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            // No text: the host is asked for the filler's words when it fires.
            Input::FillerStart {
                tool: Ref::new(0),
                speech: Ref::new(0),
                delay: Millis(10),
                interval: None,
                text: None,
            },
            // `wait_for_idle()` in the body: the tool ends, and the speech with it, with no reply.
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::AfterIdle,
                text: AgentText(String::new()),
            },
            Input::Advance { ms: Millis(50) },
        ];
        let setup = Setup { hook_bound_ms: Some(20), ..Setup::default() };
        let outcome = run(Case { setup, inputs, ..Case::default() });
        assert!(outcome.ok(), "{:#?}", outcome.failures);
    }
}

mod filler_source_no_host {
    //! A filler whose callable has no host to run it ends at its first fire.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref, ToolArgs};
    use lk_agents_voice::trace::{FillerEnd, GenerationStep, Step};

    // Catches: a `FillerSource` hook nobody can answer taken as a declined fire. With no delay and no
    // interval the scope cooled down, dwelt and fired again in the same instant, forever: the driver
    // looped without virtual time moving (survey seed 4372 at 5120 seeds).
    #[test]
    fn a_filler_callable_without_a_host_ends_the_filler() {
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::FillerStart {
                tool: Ref::new(0),
                speech: Ref::new(0),
                delay: Millis(0),
                interval: Some(Millis(0)),
                text: None,
            },
            Input::Advance { ms: Millis(50) },
        ];
        // No `hook_bound_ms`: the session has no host to call hooks on.
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        assert_eq!(run.record.livelock(), None);
        let ends: Vec<_> = run
            .record
            .journal()
            .iter()
            .filter_map(|(_, s)| match s {
                Step::Generation(GenerationStep::FillerFired { .. }) => Some(None),
                Step::Generation(GenerationStep::FillerStopped { cause, .. }) => Some(Some(*cause)),
                _ => None,
            })
            .collect();
        assert_eq!(ends, [Some(FillerEnd::Refused)], "no fire, the filler ends");
    }
}

mod filler_source_raises {
    //! A host filler callable that raises ends the filler, as the reference's loop dies on it.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, HookAnswer, Input, Millis, Ref, ToolArgs};
    use lk_agents_voice::agent_session::HookKind;
    use lk_agents_voice::trace::{FillerEnd, GenerationStep, HandoffStep, Step};

    /// The `FillerSource` hooks' positions among the hooks issued, as a `Ref<Hook>` counts them.
    fn sources(run: &Run) -> Vec<usize> {
        run.record
            .journal()
            .iter()
            .filter_map(|(_, s)| match s {
                Step::Handoff(HandoffStep::HookIssued { kind, .. }) => Some(*kind),
                _ => None,
            })
            .enumerate()
            .filter_map(|(nth, kind)| (kind == HookKind::FillerSource).then_some(nth))
            .collect()
    }

    // Catches: a `Failed` answer to `FillerSource` skipping the fire and rescheduling it. In the
    // reference `src(step)` raising ends `_FillerScheduler._loop`, so no further filler plays.
    #[test]
    fn a_raising_filler_callable_ends_the_filler() {
        let opening = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::FillerStart {
                tool: Ref::new(0),
                speech: Ref::new(0),
                delay: Millis(10),
                interval: Some(Millis(30)),
                text: None,
            },
            Input::Advance { ms: Millis(20) },
        ];
        let setup = Setup { hook_bound_ms: Some(1_000), ..Setup::default() };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &opening {
                run.apply(input).await;
            }
            let [nth] = sources(&run)[..] else { panic!("one filler source asked") };
            let hook = Ref::new(nth as u32);
            let text = AgentText(String::new());
            run.apply(&Input::HookDone { hook, answer: HookAnswer::Failed, text }).await;
            run.apply(&Input::Advance { ms: Millis(200) }).await;
            run
        });
        assert_eq!(sources(&run).len(), 1, "the callable is not asked again");
        let ends: Vec<_> = run
            .record
            .journal()
            .iter()
            .filter_map(|(_, s)| match s {
                Step::Generation(GenerationStep::FillerFired { .. }) => Some(None),
                Step::Generation(GenerationStep::FillerStopped { cause, .. }) => Some(Some(*cause)),
                _ => None,
            })
            .collect();
        assert_eq!(ends, [Some(FillerEnd::Refused)], "no fire, the filler ends");
    }
}

mod filler_while_draining {
    //! A filler whose say is refused while the session drains ends there.
    use lk_agents_fuzz::drive::{Case, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolArgs};

    // Catches: `fire_filler` counting a say the draining scheduler refused as spoken. The refused
    // speech is interrupted at birth, the session stays idle, and with no delay and no interval the
    // next fire comes in the same instant: the driver looped without virtual time moving (survey seed
    // 1120 at 4096 seeds).
    #[test]
    fn a_filler_refused_while_draining_ends() {
        let inputs = vec![
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            // The tool is still running when the close drains the scheduler.
            Input::Close { drain: true },
            Input::FillerStart {
                tool: Ref::new(0),
                speech: Ref::new(0),
                delay: Millis(0),
                interval: Some(Millis(0)),
                text: Some(AgentText("One moment.".into())),
            },
            Input::Advance { ms: Millis(50) },
        ];
        let outcome = run(Case { setup: Setup::default(), inputs, ..Case::default() });
        assert!(outcome.ok(), "{:#?}", outcome.failures);
    }
}

mod filler_without_voice_ends {
    //! A filler whose say finishes at birth does not fire forever in one instant.
    use lk_agents_fuzz::drive::{Case, RealtimeSetup, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};

    // Catches: `with_filler(delay=0, interval=0)` on a realtime model's tool: the filler's say has no
    // voice (a realtime session has no TTS), so it finished in the instant it was made, the session
    // was idle again at once, and the next fire came in the same instant, forever: the session
    // livelocked (`virtual_time_advances`, campaign seed 1090, once the fake realtime server called
    // tools). A second fire in the instant of the last now ends the filler.
    #[test]
    fn a_voiceless_filler_with_no_interval_ends() {
        let setup = Setup {
            realtime: Some(RealtimeSetup {
                turn_detection: false,
                tool_calls: Some(100),
                ..RealtimeSetup::default()
            }),
            tools: vec!["lookup".into()],
            ..Setup::default()
        };
        let inputs = vec![
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(3_000) },
            Input::FillerStart {
                tool: Ref::new(0),
                speech: Ref::new(0),
                delay: Millis(0),
                interval: Some(Millis(0)),
                text: Some(AgentText("One moment.".into())),
            },
            // The reply's words play out; its tool runs on, and the agent is idle.
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
        ];
        let outcome = run(Case { setup, inputs, ..Case::default() });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"virtual_time_advances"), "{failed:?}");
    }
}

mod silent_audio_say {
    //! A `say(audio=)` with no audio path speaks its text alone and leaves the frames unread.
    use lk_agents_fuzz::check::check_all;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};

    // Catches: `no_premature_completion` judging an audio say spoken with the audio output detached.
    // With no sink the reference's `_tts_task` forwards the text only and never reads `audio`, so the
    // speech completes with the caller's stream still open (180 of the rule's 222 failures over seeds
    // 0 to 1023, e.g. seed 2).
    #[test]
    fn an_audio_say_with_no_output_completes_on_its_text() {
        let inputs = [
            Input::SetAudioOutput { attach: false },
            Input::Advance { ms: Millis(20) },
            Input::Say {
                text: AgentText("Hello.".into()),
                input: SayInput::Audio,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(100) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let host = run.record.host_view();
        let speeches = host.speeches();
        let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
        assert!(speech.completed_normally(), "the text was spoken");
        let (failures, _) = check_all(&run.record);
        let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"no_premature_completion"), "{failed:?}");
    }
}

mod silent_segment {
    //! A speech authorized after its audio output was detached plays a silent segment, and a cut
    //! still ends it.
    use lk_agents_fuzz::drive::{Case, Outcome, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};

    // Catches: the no-output path of `perform_audio_forwarding` awaiting the next frame without
    // racing the interruption. Synthesis opened while the output was attached; once the speech is
    // cut the producer stops without ending the segment, so the reply task never finished, its
    // synthesis stream was never closed, and the run never went quiet (survey seed 25750 at 6000
    // seeds from 21000, where joining the task parked the campaign for good).
    #[test]
    fn a_cut_silent_segment_ends() {
        let outcome = cut_silent_segment(vec![]);
        assert!(outcome.ok(), "{:#?}", outcome.failures);
    }

    // The same cut, with the synthesis failed after the close: the run goes quiet, and the reply
    // task is left awaiting a frame at the end-of-run join, which parked the run forever before a
    // task that can never finish was reported as `every_task_finishes`.
    #[test]
    fn a_cut_silent_segment_ends_after_its_synthesis_failed() {
        let outcome = cut_silent_segment(vec![Input::TtsFail { tts: Ref::new(0) }]);
        assert!(outcome.ok(), "{:#?}", outcome.failures);
    }

    fn cut_silent_segment(after_close: Vec<Input>) -> Outcome {
        let mut inputs = vec![
            Input::PauseAuthorization {},
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            // The words start a synthesis, which never ends: no `LlmEnd`.
            Input::LlmText { chat: Ref::new(0), text: AgentText("Hello there.".into()) },
            Input::Advance { ms: Millis(20) },
            Input::SetAudioOutput { attach: false },
            Input::ResumeAuthorization {},
            Input::Advance { ms: Millis(20) },
            Input::Close { drain: false },
            Input::Advance { ms: Millis(20) },
        ];
        inputs.extend(after_close);
        run(Case { setup: Setup::default(), inputs, ..Case::default() })
    }
}

mod tts_first_byte_anchor {
    //! A reply's TTS timings count from when the provider was sent text, when it says so.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
    use lk_agents_voice::events::SessionEvent;

    // Catches: `tts_node_ttfb` measured from the first text the TTS node was given, though the
    // gateway client stamps when its first sentence went out; with sentence buffering that read
    // every reply's TTFB a few hundred milliseconds slow. `llm_node_ttfs` was the same stamp, so it
    // equalled the LLM's time to first token. The reference anchors both on the stream's
    // `USERDATA_TTS_STARTED_TIME`, and reports no time to first sentence for a TTS without one, as
    // this fake.
    #[test]
    fn a_tts_without_a_start_stamp_reports_no_time_to_first_sentence() {
        let setup = Setup::default();
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(300) },
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
        let [reply] = replies.as_slice() else { panic!("one reply: {replies:?}") };
        assert!(reply.tts_node_ttfb.is_some(), "{reply:?}");
        assert!(reply.llm_node_ttfs.is_none(), "{reply:?}");
    }
}

mod tts_text_transforms {
    //! The session's `tts_text_transforms` reach the voice: by default markdown and emoji never do.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};

    // Catches: the resolved `tts_text_transforms` option (default `filter_markdown` + `filter_emoji`)
    // never reaching the synthesis pipeline, which read only a constructor argument no host passes, so
    // every reply's markdown and emoji were spoken; and the markdown stage starting mid-line, so a
    // reply's leading `# ` heading was spoken too.
    #[test]
    fn markdown_and_emoji_never_reach_the_voice() {
        let say = Input::Say {
            text: AgentText("# Today\n**Great news!** Your order shipped \u{1F389}".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        };
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in [say, Input::Advance { ms: Millis(100) }] {
                run.apply(&input).await;
            }
            run
        });
        let spoken: String =
            run.record.produced().iter().map(|(_, _, _, text)| text.as_str()).collect();
        assert!(spoken.contains("Great news!"), "the words reached the voice: {spoken:?}");
        assert!(!spoken.contains(['#', '*', '\u{1F389}']), "markup reached the voice: {spoken:?}");
    }
}

mod wordless_audio_say {
    //! A `say(audio=)` with no words still plays its frames to the end.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, SayInput};

    // Catches: an audio say with empty text produced no segment, so its frames were never read and
    // the speech completed at once while the caller still held the stream open (survey seed 245).
    // The reference's `_tts_task` forwards `audio` whatever the text.
    #[test]
    fn a_wordless_audio_say_waits_for_its_frames() {
        let say = Input::Say {
            text: AgentText(String::new()),
            input: SayInput::Audio,
            allow: None,
            to_ctx: true,
        };
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in [say, Input::Advance { ms: Millis(100) }] {
                run.apply(&input).await;
            }
            let open = run.record.host_view().speeches().iter().all(|s| !s.is_done());
            assert!(open, "the speech waits for the caller's frames");
            let feed = [
                Input::FeedAudio { stream: Ref::new(0), chunks: Count(2) },
                Input::FeedEnd { stream: Ref::new(0) },
                Input::Advance { ms: Millis(100) },
            ];
            for input in &feed {
                run.apply(input).await;
            }
            run
        });
        let pushed = run.record.host_view().audio_pushed();
        assert!(!pushed.is_empty(), "the frames reached the output");
    }
}

mod pre_connect_behind_a_slow_connect {
    //! Audio buffered before a slow recogniser connects: the pre-connect replay and the seconds
    //! spoken while the session started reach the recogniser and the detector in full.
    use lk_agents_fuzz::drive::{
        Case, Outcome, PreConnect, RoomLegs, Run, Setup, finish, simulate,
    };
    use lk_agents_fuzz::fakes::{SAMPLE_RATE, SAMPLES_PER_CHUNK};
    use lk_agents_fuzz::input::{Count, Input, Millis, Ref};
    use lk_agents_fuzz::record::{Kind, Tag};

    /// What one listening stream heard: microphone samples, replayed samples, and when it heard
    /// its first.
    #[derive(Debug, PartialEq)]
    struct Heard {
        mic: usize,
        replay: usize,
        first_ms: Option<u64>,
    }

    // Catches: the console's `realtime write backlog exceeded` on the listening side: seconds of
    // audio queued behind a slow connect reach each consumer as one burst, which a bounded
    // channel with fatal overflow turns into a failed session, or a bound drops from the replay.
    #[test]
    fn seconds_of_audio_behind_a_slow_connect_reach_every_consumer() {
        let setup = Setup {
            stt: true,
            vad: true,
            connect_ms: Some(3_000),
            room_legs: Some(RoomLegs { pre_connect: Some(PreConnect::Sent) }),
            ..Setup::default()
        };
        // Three seconds of speech once the microphone is read, while the recogniser connects,
        // then a second to drain.
        let mut inputs = vec![Input::Advance { ms: Millis(100) }];
        for _ in 0..6 {
            inputs.push(Input::MicFrames { mic: Ref::new(0), chunks: Count(50) });
            inputs.push(Input::Advance { ms: Millis(500) });
        }
        inputs.push(Input::Advance { ms: Millis(1_000) });
        let case = Case { setup: setup.clone(), inputs: inputs.clone(), ..Case::default() };
        let (heard, outcome) = simulate(async move {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            let frames = run.record.inference_view().frames_heard();
            let heard = |kind: Kind| {
                let frames = frames.iter().filter(|heard| heard.kind == kind);
                let tags = frames.clone().flat_map(|heard| heard.tags);
                let mic = tags.clone().filter(|tag| matches!(tag, Tag::Mic { .. })).count();
                let replay = tags.filter(|tag| matches!(tag, Tag::Unknown(0))).count();
                let first_ms = frames.map(|heard| heard.at.at_ms).min();
                Heard { mic, replay, first_ms }
            };
            let heard = [heard(Kind::Stt), heard(Kind::Vad)];
            (heard, finish(run, Outcome { case, ..Outcome::default() }).await)
        });
        assert!(outcome.ok(), "{:#?}", outcome.failures);
        for heard in heard {
            assert_eq!(heard.mic, 300 * SAMPLES_PER_CHUNK, "every frame spoken: {heard:?}");
            assert_eq!(heard.replay, SAMPLE_RATE as usize / 2, "the half second replayed");
            assert!(heard.first_ms >= Some(3_000), "it waited for the connect: {heard:?}");
        }
    }
}

mod stalled_plugin_detector {
    //! A plugin's detector that takes no audio for seconds: its queue keeps the pre-connect
    //! replay in full and drops the rest to the newest audio, and nothing fails.
    use lk_agents_fuzz::drive::{
        Case, Outcome, PreConnect, RoomLegs, Run, Setup, finish, simulate,
    };
    use lk_agents_fuzz::fakes::{SAMPLE_RATE, SAMPLES_PER_CHUNK};
    use lk_agents_fuzz::input::{Count, Input, Millis, Ref};
    use lk_agents_fuzz::record::{Kind, Tag};

    // Catches: a stuck consumer failing the session (the fatal 256-frame input of the plugin
    // VAD base), its `Now` bound dropping the pre-connect head, or it starving the recogniser.
    #[test]
    fn a_stalled_plugin_detector_keeps_the_replay_and_resumes_on_current_audio() {
        let setup = Setup {
            stt: true,
            vad: true,
            vad_stall_ms: Some(3_000),
            room_legs: Some(RoomLegs { pre_connect: Some(PreConnect::Sent) }),
            ..Setup::default()
        };
        let mut inputs = vec![Input::Advance { ms: Millis(100) }];
        for _ in 0..6 {
            inputs.push(Input::MicFrames { mic: Ref::new(0), chunks: Count(50) });
            inputs.push(Input::Advance { ms: Millis(500) });
        }
        inputs.push(Input::Advance { ms: Millis(1_000) });
        let case = Case { setup: setup.clone(), inputs: inputs.clone(), ..Case::default() };
        let (heard, outcome) = simulate(async move {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            let frames = run.record.inference_view().frames_heard();
            let heard = |kind: Kind| {
                let tags = frames.iter().filter(|heard| heard.kind == kind).flat_map(|h| h.tags);
                let mic = tags.clone().filter(|tag| matches!(tag, Tag::Mic { .. })).count();
                (mic, tags.filter(|tag| matches!(tag, Tag::Unknown(0))).count())
            };
            let heard = (heard(Kind::Stt), heard(Kind::Vad));
            (heard, finish(run, Outcome { case, ..Outcome::default() }).await)
        });
        assert!(outcome.ok(), "{:#?}", outcome.failures);
        let ((stt_mic, stt_replay), (vad_mic, vad_replay)) = heard;
        let replay = SAMPLE_RATE as usize / 2;
        assert_eq!(
            (stt_mic, stt_replay),
            (300 * SAMPLES_PER_CHUNK, replay),
            "the recogniser heard all"
        );
        assert_eq!(vad_replay, replay, "the detector heard the whole replay");
        // `Now(500ms)`: the newest half second, and the frame that tipped it over at most.
        assert!(
            vad_mic > 0 && vad_mic <= 51 * SAMPLES_PER_CHUNK,
            "the detector resumed on current audio: {vad_mic} samples"
        );
    }
}
