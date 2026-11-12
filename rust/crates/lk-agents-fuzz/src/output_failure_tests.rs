mod a_raising_audio_output {
    //! An audio output whose capture raises cuts the speech it was playing, which ends interrupted
    //! with the output's error, and leaves the output to the next speech, which plays on it whole.
    use super::super::playout::apply_with_playout;
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_core::markup::ChatMessageExt;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::fakes::CaptureFault;
    use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};
    use lk_agents_fuzz::record::{Recording, Sink};
    use lk_agents_voice::events::{ErrorSource, SessionEvent};

    const FIRST: &str =
        "One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve.";
    const SECOND: &str = "Thirteen, fourteen.";

    fn say(text: &str) -> Input {
        Input::Say {
            text: AgentText(text.into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        }
    }

    fn raises(at: u32) -> Option<CaptureFault> {
        Some(CaptureFault { at, hangs: false })
    }

    /// Two speeches, one after the other, on an output that fails on one frame; the speaker plays
    /// whatever it took.
    fn two_speeches(setup: &Setup) -> Recording {
        simulate(async {
            let mut run = Run::start(setup, 0, Default::default()).await;
            for input in [say(FIRST), Input::Advance { ms: Millis(8_000) }] {
                apply_with_playout(&mut run, &input).await;
            }
            for input in [say(SECOND), Input::Advance { ms: Millis(8_000) }] {
                apply_with_playout(&mut run, &input).await;
            }
            run.record
        })
    }

    /// The agent's messages, and whether each was marked interrupted.
    fn assistant_said(record: &Recording) -> Vec<(String, bool)> {
        record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::Message(m) if m.role == Role::Assistant => {
                        m.text_content().map(|text| (text, m.interrupted))
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    /// The second speech played whole on the same output and joined the history; the first ended
    /// interrupted with the output's error, told once as a recoverable `error` event, and what
    /// it left in the history is an interruption's.
    fn assert_recovered(record: &Recording, synchronised: bool) {
        let host = record.host_view();
        let speeches = host.speeches();
        let [first, second] = speeches.as_slice() else {
            panic!("two speeches: {}", speeches.len())
        };
        assert!(
            second.completed_normally(),
            "the output plays the next speech: {:?}",
            second.error()
        );
        let pushed = record.sink().iter().any(|(at, sink)| {
            matches!(sink, Sink::Audio { .. })
                && second.created_at().is_some_and(|created| created < *at)
        });
        assert!(pushed, "the next speech's audio reached the output");
        let said = assistant_said(record);
        let (last, interrupted) = said.last().expect("the next speech joined the history");
        assert_eq!((last.as_str(), *interrupted), (SECOND, false), "{said:?}");
        // What the failed speech left is an interruption's: marked so, and with the synchroniser
        // only the words heard (none played before the failure).
        for (text, interrupted) in &said[..said.len() - 1] {
            assert!(*interrupted, "the failed speech's message is an interruption's: {said:?}");
            assert!(!synchronised || text != FIRST, "committed as if heard whole: {said:?}");
        }

        assert!(first.is_done() && first.is_interrupted(), "the failed speech ends interrupted");
        let error = first.error().expect("the failed speech carries the output's error");
        assert!(error.contains("the host audio output raised"), "{error}");
        let errors: Vec<_> = host
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::Error(e) => Some(e),
                _ => None,
            })
            .collect();
        let [error] = errors.as_slice() else { panic!("one error event: {errors:?}") };
        assert_eq!(error.source, ErrorSource::AudioOutput);
        assert!(error.recoverable, "an output failure does not end the call");
    }

    // Catches: a capture that raised on a segment's first frame left the segment open on the
    // output's tracker with no audio its sink would ever report, the cleanup cleared without
    // closing it, and the binding was retained for a report that never came: every later speech
    // was refused with `output admission refused: the output binding could not be activated`.
    #[test]
    fn a_first_frame_that_raises_leaves_the_output_to_the_next_speech() {
        let record = two_speeches(&Setup { output_fault: raises(0), ..Setup::default() });
        assert_recovered(&record, false);
    }

    // Catches: a capture that raised mid-speech, after the output took some of its audio, was cut
    // as if nothing had happened: the handle read neither interrupted nor failed, and no `error`
    // event or log told anyone.
    #[test]
    fn a_frame_that_raises_mid_speech_cuts_the_speech_and_reports_the_error() {
        assert_recovered(
            &two_speeches(&Setup { output_fault: raises(3), ..Setup::default() }),
            false,
        );
        let setup = Setup { output_fault: raises(3), sync_transcription: true, ..Setup::default() };
        assert_recovered(&two_speeches(&setup), true);
    }

    // Catches: the same lock-up below a wrapper: the synchroniser's tracker and the output's own
    // each kept the refused frame's segment open.
    #[test]
    fn a_first_frame_that_raises_below_the_synchroniser_leaves_the_output_usable() {
        let setup = Setup { output_fault: raises(0), sync_transcription: true, ..Setup::default() };
        assert_recovered(&two_speeches(&setup), true);
    }
}

mod a_capture_that_never_returns {
    //! A speech interrupted while its output holds its first frame (an avatar that never joins)
    //! leaves the output to the next speech.
    use super::super::playout::apply_with_playout;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::fakes::CaptureFault;
    use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};

    fn say(text: &str) -> Input {
        Input::Say {
            text: AgentText(text.into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        }
    }

    // Catches: the interrupt cancelled the capture after the output's tracker had opened the
    // segment, the cleanup cleared without closing it, and the binding was retained for a report
    // that never came: the next `say()` failed with `output admission refused`.
    #[test]
    fn interrupting_a_held_first_frame_leaves_the_output_to_the_next_speech() {
        let fault = Some(CaptureFault { at: 0, hangs: true });
        let record = simulate(async {
            let mut run = Run::start(
                &Setup { output_fault: fault, ..Setup::default() },
                0,
                Default::default(),
            )
            .await;
            let inputs = [
                say("Hello there."),
                Input::Advance { ms: Millis(2_000) },
                Input::Interrupt { force: true, wait: false },
                Input::Advance { ms: Millis(2_000) },
                say("Second."),
                Input::Advance { ms: Millis(8_000) },
            ];
            for input in &inputs {
                apply_with_playout(&mut run, input).await;
            }
            run.record
        });
        let host = record.host_view();
        let speeches = host.speeches();
        let [first, second] = speeches.as_slice() else {
            panic!("two speeches: {}", speeches.len())
        };
        assert!(first.is_done() && first.is_interrupted() && first.error().is_none());
        assert!(
            second.completed_normally(),
            "the output plays the next speech: {:?}",
            second.error()
        );
    }
}

mod a_clear_the_output_never_reports {
    //! An audio output that never reports playback finished after a clear (a custom output that
    //! skips `on_playback_finished`, a dead avatar worker) holds the output for 5 s from the cut at
    //! most: the cut segment settles where the output last reported, one recoverable `error`
    //! event says so, and the next speech plays.
    use super::super::playout::apply_with_playout;
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_core::markup::ChatMessageExt;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::fakes::UnreportedClear;
    use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};
    use lk_agents_fuzz::record::{RecorderCall, Recording, Sink};
    use lk_agents_voice::events::{ErrorEvent, ErrorSource, SessionEvent};
    use std::time::Duration;

    /// About ten seconds of speech: still being shown when the bound runs out.
    const FIRST: &str = "One, two, three, four, five, six, seven, eight, nine, ten, eleven, \
                         twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, \
                         nineteen, twenty, twenty-one, twenty-two, twenty-three, twenty-four, \
                         twenty-five, twenty-six, twenty-seven, twenty-eight, twenty-nine, thirty.";
    const SECOND: &str = "Thirty-one, thirty-two.";
    /// When the first speech is cut, after it played this long.
    const CUT_MS: u64 = 1_500;
    /// When the second speech is said, after the cut: past the bound.
    const NEXT_MS: u64 = 7_000;

    fn say(text: &str) -> Input {
        Input::Say {
            text: AgentText(text.into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        }
    }

    /// A recorded speech cut after some of it played, then another said after the bound.
    fn cut_then_speak(late: Option<Duration>) -> Recording {
        cut_then_speak_on(true, Some(UnreportedClear { late }))
    }

    /// The same, with or without the synchroniser, on an output that reports its clear as told.
    fn cut_then_speak_on(synchronised: bool, unreported: Option<UnreportedClear>) -> Recording {
        let setup = Setup {
            sync_transcription: synchronised,
            unreported_clear: unreported,
            ..Setup::default()
        };
        simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            let inputs = [
                Input::Recorder { attach: true },
                say(FIRST),
                Input::Advance { ms: Millis(CUT_MS) },
                Input::Interrupt { force: true, wait: false },
                Input::Advance { ms: Millis(NEXT_MS) },
                say(SECOND),
                Input::Advance { ms: Millis(12_000) },
            ];
            for input in &inputs {
                apply_with_playout(&mut run, input).await;
            }
            run.record
        })
    }

    fn assistant_said(record: &Recording) -> Vec<(String, bool)> {
        record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::Message(m) if m.role == Role::Assistant => {
                        m.text_content().map(|text| (text, m.interrupted))
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    fn errors(record: &Recording) -> Vec<(u64, ErrorEvent)> {
        record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(at, ev)| match ev {
                SessionEvent::Error(e) => Some((at.at_ms, (**e).clone())),
                _ => None,
            })
            .collect()
    }

    /// Where the recording was told each segment stopped, in order.
    fn recorded_finishes(record: &Recording) -> Vec<u64> {
        record
            .sink()
            .iter()
            .filter_map(|(_, sink)| match sink {
                Sink::Recorder(RecorderCall::PlaybackFinished { position_ms }) => {
                    Some(*position_ms)
                }
                _ => None,
            })
            .collect()
    }

    /// The words the transcription output was given.
    fn shown(record: &Recording) -> String {
        record
            .sink()
            .iter()
            .filter_map(|(_, sink)| match sink {
                Sink::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The cut speech ended interrupted, recorded as stopped where the output last reported it
    /// played, its words shown no further; the next played whole on the same output.
    fn assert_spoke_after_the_cut(record: &Recording) {
        let host = record.host_view();
        let speeches = host.speeches();
        let [first, second] = speeches.as_slice() else {
            panic!("two speeches: {}", speeches.len())
        };
        assert!(first.is_done() && first.is_interrupted(), "the cut speech ends interrupted");
        assert!(
            second.completed_normally(),
            "the output plays the next speech: {:?}",
            second.error()
        );
        let pushed = record.sink().iter().any(|(at, sink)| {
            matches!(sink, Sink::Audio { .. })
                && second.created_at().is_some_and(|created| created < *at)
        });
        assert!(pushed, "the next speech's audio reached the output");
        let said = assistant_said(record);
        let [(_, true), (whole, false)] = said.as_slice() else {
            panic!("the cut speech, then the whole next one: {said:?}")
        };
        assert_eq!(whole, SECOND);
        let finishes = recorded_finishes(record);
        assert_eq!(
            finishes.first(),
            Some(&CUT_MS),
            "recorded as stopped where the output last reported: {finishes:?}"
        );
        assert_eq!(finishes.len(), 2, "each segment finishes once: {finishes:?}");
        let shown = shown(record);
        assert!(!shown.contains("thirty."), "words shown past the cut: {shown:?}");
    }

    // Catches: the output binding of a speech cut on an output that never reports the clear was
    // retained for that report forever: every later speech was refused with `output admission
    // refused`, the cut speech's words went on being shown, and nobody was told.
    #[test]
    fn the_output_goes_to_the_next_speech_five_seconds_after_the_cut() {
        let record = cut_then_speak(None);
        assert_spoke_after_the_cut(&record);
        let errors = errors(&record);
        let [(at, error)] = errors.as_slice() else { panic!("one error event: {errors:?}") };
        assert_eq!(error.source, ErrorSource::AudioOutput);
        assert!(error.recoverable, "a silent output does not end the call");
        let message = error.error.to_string();
        assert!(message.contains("within 5 s of being cleared"), "{message}");
        // The bound runs from the cut's cleanup, at most the cut receipt's wait after the cut.
        assert!((CUT_MS + 5_000..=CUT_MS + 6_500).contains(at), "released at {at} ms");
    }

    // Catches: a report that came after the bound settled the segment a second time.
    #[test]
    fn a_report_after_the_bound_changes_nothing() {
        let late = cut_then_speak(Some(Duration::from_millis(NEXT_MS + 2_000)));
        let never = cut_then_speak(None);
        assert_spoke_after_the_cut(&late);
        assert_eq!(assistant_said(&late), assistant_said(&never));
        assert_eq!(shown(&late), shown(&never));
        assert_eq!(errors(&late).len(), 1, "{:?}", errors(&late));
    }

    // Catches: an output that reports its clear within the bound told as stuck.
    #[test]
    fn a_report_within_the_bound_is_no_error() {
        let record = cut_then_speak(Some(Duration::from_millis(3_000)));
        assert_spoke_after_the_cut(&record);
        assert!(errors(&record).is_empty(), "{:?}", errors(&record));
    }

    /// When the speech was cut: the interrupt's instant.
    fn cut_at(record: &Recording) -> u64 {
        let interrupt =
            record.applied().iter().find(|a| matches!(a.input, Input::Interrupt { .. }));
        interrupt.expect("the speech was interrupted").at.at_ms
    }

    /// The words the transcription output was given up to the cut, and from it until the next
    /// speech was said.
    fn shown_around_the_cut(record: &Recording) -> (String, String) {
        let (cut, next) = (cut_at(record), record.host_view().speeches()[1].created_at());
        let next = next.expect("the next speech was said");
        let (mut before, mut after) = (String::new(), String::new());
        for (at, sink) in record.sink() {
            let Sink::Text { text, .. } = sink else { continue };
            if at.at_ms <= cut {
                before.push_str(text);
            } else if *at < next {
                after.push_str(text);
            }
        }
        (before, after)
    }

    fn squash(text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// The cut speech's message: interrupted, and only words the transcript showed by the cut,
    /// with none shown after it. Returns the message.
    fn assert_history_is_what_was_shown(record: &Recording) -> String {
        let said = assistant_said(record);
        let [(cut, true), (_, false)] = said.as_slice() else {
            panic!("the cut speech, interrupted, then the next one: {said:?}")
        };
        let (before, after) = shown_around_the_cut(record);
        let (heard, before) = (squash(cut), squash(&before));
        assert!(!heard.is_empty(), "some of the speech was heard before the cut: {said:?}");
        assert_ne!(heard, squash(FIRST), "committed as if heard whole: {said:?}");
        assert!(
            before.starts_with(&heard),
            "the message holds words the transcript did not show by the cut: {heard:?} after \
             showing {before:?}"
        );
        assert_eq!(squash(&after), "", "the transcript went on after the cut");
        let (failures, expectations) = lk_agents_fuzz::check::check_all(record);
        let rule = "cut_holds_only_shown_words";
        assert!(expectations[rule] > 0, "{rule} judged no speech");
        let failed: Vec<_> = failures.iter().filter(|f| f.contract == rule).collect();
        assert!(failed.is_empty(), "{failed:?}");
        cut.clone()
    }

    // Catches: an output that reported its clear after the cut receipt's 1 s wait left the
    // history claiming the whole speech was said (marked interrupted), though the transcript had
    // shown only its first words.
    #[test]
    fn a_late_report_commits_only_the_words_shown_by_the_cut() {
        let late = Some(UnreportedClear { late: Some(Duration::from_millis(3_000)) });
        let late = cut_then_speak_on(true, late);
        let heard = assert_history_is_what_was_shown(&late);
        let prompt = cut_then_speak_on(true, None);
        assert_eq!(
            heard,
            assistant_said(&prompt)[0].0,
            "a late report commits what a prompt one does"
        );
    }

    // Catches: the same with a report that never comes: the whole speech committed.
    #[test]
    fn a_report_that_never_comes_commits_only_the_words_shown_by_the_cut() {
        let never = cut_then_speak_on(true, Some(UnreportedClear { late: None }));
        let heard = assert_history_is_what_was_shown(&never);
        let prompt = cut_then_speak_on(true, None);
        assert_eq!(
            heard,
            assistant_said(&prompt)[0].0,
            "a lost report commits what a prompt one does"
        );
    }

    // Pins: an output that reports its clear at once commits the synchroniser's transcript.
    #[test]
    fn a_prompt_report_commits_the_synchronised_transcript() {
        let prompt = cut_then_speak_on(true, None);
        assert_history_is_what_was_shown(&prompt);
    }

    // Pins: with no synchroniser there is no shown text to go by: the whole speech, interrupted.
    #[test]
    fn without_the_synchroniser_a_late_report_commits_the_whole_speech_interrupted() {
        let late = Some(UnreportedClear { late: Some(Duration::from_millis(3_000)) });
        let said = assistant_said(&cut_then_speak_on(false, late));
        let [(cut, true), (next, false)] = said.as_slice() else {
            panic!("the cut speech, interrupted, then the next one: {said:?}")
        };
        assert_eq!((cut.as_str(), next.as_str()), (FIRST, SECOND));
    }
}

mod a_body_detaching_as_the_session_closes {
    //! A body whose call a host update already ended, detaching as the session closes: it is over
    //! at its answer, since nothing (no call of the driver's, no runtime task) is left to end it.
    //! The survey reached it through an output failure that only steered the run there.
    use lk_agents_fuzz::drive::{Case, Policy, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
    use lk_agents_fuzz::search;

    fn failed(outcome: &lk_agents_fuzz::drive::Outcome) -> Vec<String> {
        outcome.failures.iter().map(|f| format!("{}: {}", f.contract, f.detail)).collect()
    }

    // Catches: the harness kept the body running until the driver refused its report, but past
    // the close's hard edge, or once the loop had exited, the report is dropped without a word:
    // `Tool #0 still open after the driver exited`.
    #[test]
    fn it_is_over_at_its_answer() {
        let case = Case {
            setup: Setup { tools: vec!["lookup".into()], ..Setup::default() },
            inputs: vec![
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(100) },
                Input::LlmToolCall {
                    chat: Ref::new(0),
                    name: "lookup".into(),
                    args: ToolArgs("{}".into()),
                },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(100) },
                Input::ToolUpdate { tool: Ref::new(0), progress: None, end: ToolAnswer::Return },
                Input::Advance { ms: Millis(100) },
                Input::Close { drain: true },
                Input::ToolAnswered {
                    tool: Ref::new(0),
                    answer: ToolAnswer::Detached,
                    text: AgentText("Checking.".into()),
                },
                Input::Advance { ms: Millis(100) },
            ],
            ..Case::default()
        };
        let outcome = run(case);
        assert!(outcome.ok(), "{:?}", failed(&outcome));
        assert!(!outcome.stopped_early, "{:?}", outcome.ending);
    }

    // The survey finding itself (seed 209 TaskBridge, nothing_left_live).
    #[test]
    fn survey_seed_209() {
        let seed = 209;
        let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
        let dials = search::Dials::of(seed);
        let outcome = search::explore_with(setup, seed, 3000, dials, Policy::TaskBridge);
        assert!(outcome.ok(), "{:?}", failed(&outcome));
    }
}
