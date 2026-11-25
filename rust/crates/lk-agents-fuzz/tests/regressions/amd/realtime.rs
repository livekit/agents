//! AMD's executor on a realtime model (proposal, Realtime; implementation step 4). The realtime
//! fake records every command the session sends it, so each story asserts what the server was
//! asked: its buffer commits, its response requests and their per-response instructions, and the
//! tools it held when each response was asked; and what the conversation saved. Each story names
//! the `tests/test_amd.py` scenario on `main` it ports (the "Realtime models" region). Python's
//! realtime fixture runs with `machine_silence_threshold=0`; these run with the 1.5 s default, so
//! machine-stage replies also wait for silence.

use super::{Call, detecting};
use lk_agents_fuzz::drive::{RealtimeSetup, Setup};
use lk_agents_fuzz::fakes::RealtimeDelays;
use lk_agents_fuzz::fakes::amd::{AmdSetup, HUMAN_AFTER_MACHINE, IVR, SCREENING, SEND_DTMF_TOOL};
use lk_agents_fuzz::input::RtEvent;
use lk_agents_fuzz::input::{
    AgentText, AmdSays, Count, HookAnswer, Input, Millis, Ref, Stt, TurnMode, UserText,
};
use lk_agents_fuzz::record::amd::RtResponse;
use lk_agents_fuzz::record::amd_run::AskKind;
use lk_agents_fuzz::record::{Detail, Kind, RtSent};
use lk_agents_voice::amd::events::{AmdReason, Stage};

/// `machine_silence_threshold`'s default (spec §4.1).
const SILENCE_MS: u64 = 1_500;

/// A realtime model AMD enters (spec §5.1): client-side turn detection, per-response tools and
/// client tool replies (the fake's capabilities), no tool calls of its own. Its transcripts come
/// half a second after the buffer's commit.
fn model(user_transcription: bool) -> RealtimeSetup {
    RealtimeSetup {
        turn_detection: false,
        tool_calls: Some(0),
        user_transcription,
        delays: RealtimeDelays { transcription: Some(500), ..Default::default() },
        ..RealtimeSetup::default()
    }
}

/// The session's recogniser and detector, a realtime model that transcribes, and AMD's executor.
fn on_realtime(amd: AmdSetup) -> Setup {
    Setup { realtime: Some(model(true)), ..detecting(amd) }
}

/// No session recogniser: only the detector, and AMD's own recogniser.
fn without_session_stt(user_transcription: bool) -> Setup {
    let amd = AmdSetup { own_stt: true, ..AmdSetup::default() };
    Setup { stt: false, realtime: Some(model(user_transcription)), ..detecting(amd) }
}

impl Call {
    /// The live recognition streams: the session's, then AMD's own.
    fn recognisers(&self) -> (Option<Ref<Stt>>, Option<Ref<Stt>>) {
        let live = self.run.record.live().of(Kind::Stt);
        let mut session = None;
        let mut amd = None;
        for r in self.run.record.resources(Kind::Stt).filter(|r| live.contains(&r.nth())) {
            match r.detail() {
                Detail::AmdStt { .. } => amd = Some(Ref::new(r.nth())),
                _ => session = Some(Ref::new(r.nth())),
            }
        }
        (session, amd)
    }

    /// The person speaks for 300 ms into the room, the realtime server buffering the audio; the
    /// session's recogniser hears `session` and AMD's hears `amd`, each when it is open. Returns
    /// when the speech ended, by the detector's ground truth.
    async fn speaks(&mut self, session: &str, amd: &str) -> u64 {
        let (session_stt, amd_stt) = self.recognisers();
        self.send(Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) }).await;
        let mut finals = Vec::new();
        for (stt, words) in [(session_stt, session), (amd_stt, amd)] {
            let Some(stt) = stt else { continue };
            self.send(Input::SttStart { stt }).await;
            finals.push((Ref::new(self.utterances), words.to_owned()));
            self.utterances += 1;
        }
        self.send(Input::MicFrames { mic: Ref::new(0), chunks: Count(30) }).await;
        self.advance(300).await;
        for (utterance, words) in &finals {
            let text = UserText(words.clone());
            self.send(Input::SttFinal { utterance: *utterance, text, speaker: None }).await;
        }
        self.send(Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(0) }).await;
        let ended = self.now_ms();
        for (utterance, _) in &finals {
            self.send(Input::SttEnd { utterance: *utterance }).await;
        }
        self.advance(600).await;
        ended
    }

    /// Every response the realtime server was asked for, in order.
    fn responses(&self) -> Vec<RtResponse> {
        self.run.record.amd_view().rt_responses()
    }

    /// When the session committed the server's audio buffer, in ms.
    fn commits(&self) -> Vec<u64> {
        self.run.record.amd_view().rt_commits().into_iter().map(|at| at.at_ms).collect()
    }

    /// The tools the session gave the server first, at the open.
    fn opening_tools(&self) -> Option<Vec<String>> {
        self.run.record.rt_sent().iter().find_map(|(_, _, sent)| match sent {
            RtSent::Tools(tools) => Some(tools.clone()),
            _ => None,
        })
    }

    /// The words of each classification, in order.
    fn classified(&self) -> Vec<String> {
        (self.amd_asks_of(AskKind::Classify).into_iter())
            .filter_map(|a| a.messages.last().map(|m| m.text.clone()))
            .collect()
    }

    fn entered_ok(&self) -> bool {
        let answers = self.run.record.amd_run().enter_answers().to_vec();
        answers.len() == 1 && answers[0].1.is_ok()
    }

    fn refusal(&self) -> String {
        let answers = self.run.record.amd_run().enter_answers().to_vec();
        assert_eq!(answers.len(), 1, "{answers:?}");
        answers[0].1.clone().expect_err("AMD refused the realtime model")
    }

    fn clean(&self) {
        let failures = self.failures();
        assert!(failures.is_empty(), "{failures:#?}");
    }
}

// Entry (spec §5.1).

/// Server-initiated replies never reach the gate: AMD refuses a realtime model whose server
/// detects turns, and the session installs no gate. Python:
/// `test_unsupported_configuration_does_not_install_amd[server_turn_detection]`.
#[test]
fn amd_refuses_a_realtime_model_with_server_turn_detection() {
    let setup = Setup {
        realtime: Some(RealtimeSetup { turn_detection: true, ..model(true) }),
        ..detecting(AmdSetup::default())
    };
    let call = Call::run(setup, async |call| call.enter().await);
    let refusal = call.refusal();
    assert!(refusal.contains("client-side turn detection"), "{refusal}");
    assert!(call.run.record.amd_view().attachments().is_empty(), "no gate");
    call.clean();
}

/// A realtime model with neither the session's recogniser nor AMD's is refused; AMD's own
/// recogniser is enough. Python:
/// `test_unsupported_configuration_does_not_install_amd[missing_stt]`.
#[test]
fn amd_refuses_a_realtime_model_with_no_stt() {
    let setup = Setup { stt: false, realtime: Some(model(true)), ..detecting(AmdSetup::default()) };
    let call = Call::run(setup, async |call| call.enter().await);
    let refusal = call.refusal();
    assert!(refusal.contains("session STT or AMD STT"), "{refusal}");
    call.clean();

    let call = Call::run(without_session_stt(true), async |call| call.enter().await);
    assert!(call.entered_ok(), "{:?}", call.run.record.amd_run().enter_answers());
    call.clean();
}

/// A realtime model with client-side detection, per-response tools and client tool replies, and
/// the session's recogniser, is entered. Python: `running_realtime`'s setup.
#[test]
fn amd_enters_a_realtime_model_with_client_side_detection() {
    let call = Call::run(on_realtime(AmdSetup::default()), async |call| call.enter().await);
    assert!(call.entered_ok(), "{:?}", call.run.record.amd_run().enter_answers());
    assert_eq!(call.run.record.amd_view().attachments().len(), 1, "the gate attached");
    call.clean();
}

// The reply (proposal, Realtime).

/// A screener: the buffer is committed at the commit, before the classification; nothing is
/// asked of the model until the prediction allows a reply and the participant has kept silent
/// for the threshold; then one response is asked, carrying the screening instructions unchanged.
/// Python: `test_reply_waits_for_classification_and_silence`.
#[test]
fn a_screened_reply_is_asked_after_its_prediction_and_the_silence() {
    let mut classified_at = 0;
    let mut ended = 0;
    let call = Call::run(on_realtime(AmdSetup::default()), async |call| {
        call.enter().await;
        ended = call.speaks("hi this is call assist please state your name", "").await;
        call.advance(100).await;
        classified_at = call.now_ms();
        call.classifier_says(AmdSays::Screening).await;
        call.advance(4_000).await;
    });
    let responses = call.responses();
    assert_eq!(responses.len(), 1, "one response: {responses:?}");
    let asked = responses[0].at.at_ms;
    assert!(asked > classified_at, "asked at {asked} ms, before the prediction");
    assert!(asked >= ended + SILENCE_MS, "asked at {asked} ms, inside the silence from {ended} ms");
    assert_eq!(responses[0].instructions.as_deref(), Some(SCREENING));
    let commits = call.commits();
    assert!(
        commits.iter().any(|at| *at >= ended && *at < classified_at),
        "the buffer was committed at the commit, before the prediction: {commits:?}"
    );
    assert_eq!(commits.len(), 1, "committed once: {commits:?}");
    call.clean();
}

/// Stage instructions ride one reply each and never the session's instructions: the screener's
/// reply carries them, a first `human` after it (likely human) carries none, the second `human`
/// carries the human instructions, and once AMD completed a host reply carries none. Python:
/// `test_stage_instructions_expire_after_first_human_reply`.
#[test]
fn stage_instructions_ride_their_reply_and_expire() {
    let call = Call::run(on_realtime(AmdSetup::default()), async |call| {
        call.enter().await;
        for (words, says) in [
            ("please state your name and the purpose of your call", AmdSays::Screening),
            ("hello", AmdSays::Human),
            ("yes this is Sam", AmdSays::Human),
        ] {
            call.speaks(words, "").await;
            call.classifier_says(says).await;
            call.advance(6_000).await;
        }
        call.send(Input::GenerateReply { user_input: None, instructions: None, allow: None }).await;
        call.advance(4_000).await;
    });
    let outcome = call.outcomes();
    assert_eq!(outcome.len(), 1, "{outcome:?}");
    assert_eq!((outcome[0].category, outcome[0].reason), (Stage::Human, AmdReason::Finished));
    let instructions: Vec<_> = call.responses().into_iter().map(|q| q.instructions).collect();
    assert_eq!(
        instructions,
        vec![Some(SCREENING.to_owned()), None, Some(HUMAN_AFTER_MACHINE.to_owned()), None],
        "one response per reply, each with its own instructions"
    );
    for (_, _, sent) in call.run.record.rt_sent() {
        if let RtSent::Instructions(text) = sent {
            assert_eq!(text, "You are a test agent.", "the session's instructions are kept");
        }
    }
    assert!(!call.recorded_anywhere(SCREENING) && !call.recorded_anywhere(HUMAN_AFTER_MACHINE));
    call.clean();
}

/// A `wait` turn is never answered, and its audio is committed so it cannot ride the next reply;
/// the conversation keeps only the model's own transcript of it, not the session recogniser's
/// words. Python: `test_suppressed_reply_keeps_only_the_realtime_transcript[wait]`.
#[test]
fn a_wait_turn_commits_its_audio_and_keeps_only_the_models_transcript() {
    let call = Call::run(on_realtime(AmdSetup::default()), async |call| {
        call.enter().await;
        call.speaks("please hold while we connect your call", "").await;
        call.classifier_says(AmdSays::Wait).await;
        call.advance(5_000).await;
    });
    assert!(call.responses().is_empty(), "a wait turn asks for no response");
    assert_eq!(call.commits().len(), 1, "its audio was committed");
    let users = call.saved_user_words();
    assert_eq!(users.len(), 1, "one user item, the model's transcript: {users:?}");
    assert_ne!(users[0], "please hold while we connect your call", "not the recogniser's words");
    call.clean();
}

/// A reply held for silence is asked only once the silence ends: a newer turn during the hold
/// supersedes it before anything was asked, and its `wait` keeps the model silent. Python:
/// `test_amd_interrupts_a_pending_reply_despite_disabled_interruptions` (with the hold, the
/// held reply is never generated at all, spec §10.3).
#[test]
fn a_reply_superseded_during_its_silence_is_never_asked() {
    let call = Call::run(on_realtime(AmdSetup::default()), async |call| {
        call.enter().await;
        call.speaks("please state your name", "").await;
        call.classifier_says(AmdSays::Screening).await;
        call.advance(200).await;
        call.speaks("please hold", "").await;
        call.classifier_says(AmdSays::Wait).await;
        call.advance(5_000).await;
    });
    assert!(call.responses().is_empty(), "{:?}", call.responses());
    assert_eq!(call.commits().len(), 2, "each turn's audio was committed");
    call.clean();
}

/// A menu's reply offers AMD's DTMF tool through the session's tools, set for that response and
/// restored after: the first `human` after the menu is likely human and keeps the keypad, the
/// completing `human` finds the agent's own tools again, and the agent's tools never gain it.
/// Python: `test_ivr_dtmf_only_generates_a_reply_on_failure`.
#[test]
fn a_dtmf_reply_sets_the_session_tools_and_restores_them() {
    let call = Call::run(on_realtime(AmdSetup::default()), async |call| {
        call.enter().await;
        call.speaks("for billing press one for support press two", "").await;
        call.classifier_says(AmdSays::Ivr).await;
        call.advance(50).await;
        call.menu_says(AmdSays::NoMenu).await;
        call.advance(6_000).await;
        for words in ["hello", "yes this is Sam"] {
            call.speaks(words, "").await;
            call.classifier_says(AmdSays::Human).await;
            call.advance(6_000).await;
        }
    });
    let own = call.opening_tools().expect("the agent's tools at the open");
    assert!(!own.iter().any(|t| t == SEND_DTMF_TOOL));
    let responses = call.responses();
    assert_eq!(responses.len(), 3, "{responses:?}");
    let dtmf = |q: &RtResponse| q.tools.iter().flatten().filter(|t| *t == SEND_DTMF_TOOL).count();
    assert_eq!(responses[0].instructions.as_deref(), Some(IVR));
    assert_eq!(dtmf(&responses[0]), 1, "the menu's reply offers the keypad once: {responses:?}");
    assert_eq!(dtmf(&responses[1]), 1, "a likely-human reply in the menu keeps the keypad");
    assert_eq!(responses[1].instructions, None);
    assert_eq!(responses[2].tools.as_ref(), Some(&own), "the agent's tools are restored");
    assert_eq!(responses[2].instructions.as_deref(), Some(HUMAN_AFTER_MACHINE));
    call.clean();
}

/// A customer hook that stops the response keeps the model silent though AMD allowed the reply.
/// Python: `test_customer_stop_response_prevents_realtime_generation`.
#[test]
fn a_stopped_turn_asks_for_no_response() {
    let setup = Setup { hook_bound_ms: Some(10_000), ..on_realtime(AmdSetup::default()) };
    let call = Call::run(setup, async |call| {
        call.enter().await;
        call.speaks("please state your name", "").await;
        call.classifier_says(AmdSays::Screening).await;
        call.advance(100).await;
        let hook = call.run.record.resources(Kind::Hook).count().saturating_sub(1) as u32;
        let stop = Input::HookDone {
            hook: Ref::new(hook),
            answer: HookAnswer::TurnStop,
            text: AgentText(String::new()),
        };
        call.send(stop).await;
        call.advance(4_000).await;
    });
    assert!(call.responses().is_empty(), "{:?}", call.responses());
    call.clean();
}

// Manual commits (proposal, Realtime: both realtime commit paths go through the gate).

/// `commit_user_turn()` under AMD answers only on the turn's prediction: the buffer is committed
/// at once, the response only after `human`, and never after `wait`. Python:
/// `test_manual_realtime_commit_returns_the_reply_amd_allows`.
#[test]
fn a_manual_commit_replies_only_on_its_prediction() {
    for (says, replies) in [(AmdSays::Human, 1), (AmdSays::Wait, 0)] {
        let setup = Setup { turn: TurnMode::Manual, ..on_realtime(AmdSetup::default()) };
        let mut committed = 0;
        let mut classified = 0;
        // The reply may be asked in the prediction's own millisecond: ordered by observation.
        let mut predicted = lk_agents_fuzz::record::Observed::END;
        let call = Call::run(setup, async |call| {
            call.enter().await;
            call.speaks("Hello?", "").await;
            committed = call.now_ms();
            let commit = Input::CommitUserTurn {
                transcript_timeout: Millis(0),
                stt_flush: Millis(0),
                skip_reply: false,
            };
            call.send(commit).await;
            call.advance(200).await;
            classified = call.now_ms();
            predicted = call.run.record.now();
            call.classifier_says(says).await;
            call.advance(4_000).await;
        });
        assert_eq!(call.classified(), vec!["Hello?".to_owned()], "{says:?}");
        let responses = call.responses();
        assert_eq!(responses.len(), replies, "{says:?}: {responses:?}");
        assert!(responses.iter().all(|q| q.at > predicted && q.instructions.is_none()));
        let commits = call.commits();
        assert!(
            commits.len() == 1 && commits[0] >= committed && commits[0] < classified,
            "{says:?}: the buffer was committed at the commit: {commits:?}"
        );
        call.clean();
    }
}

/// With no session recogniser, `commit_user_turn()` still goes through the gate: AMD's own
/// recogniser supplies the turn's words, the buffer is committed at the commit, and the screened
/// reply is asked after the prediction with its instructions. Python:
/// `test_amd_stt_supplies_transcripts_at_eot[manual-False]`.
#[test]
fn a_manual_commit_without_session_stt_is_gated() {
    let setup = Setup { turn: TurnMode::Manual, ..without_session_stt(true) };
    let mut committed = 0;
    let mut classified = 0;
    let call = Call::run(setup, async |call| {
        call.enter().await;
        call.speaks("", "Please state your name.").await;
        committed = call.now_ms();
        let commit = Input::CommitUserTurn {
            transcript_timeout: Millis(0),
            stt_flush: Millis(0),
            skip_reply: false,
        };
        call.send(commit).await;
        call.advance(200).await;
        classified = call.now_ms();
        call.classifier_says(AmdSays::Screening).await;
        call.advance(4_000).await;
    });
    assert_eq!(call.classified(), vec!["Please state your name.".to_owned()]);
    let responses = call.responses();
    assert_eq!(responses.len(), 1, "{responses:?}");
    assert!(responses[0].at.at_ms > classified);
    assert_eq!(responses[0].instructions.as_deref(), Some(SCREENING));
    let commits = call.commits();
    assert!(
        commits.len() == 1 && commits[0] >= committed && commits[0] < classified,
        "{commits:?}"
    );
    call.clean();
}

/// The session's own commit (`TurnCommand::Commit`, `on_turn(Commit)`) with no session
/// recogniser goes through the gate as well: nothing is asked before AMD allows the reply.
/// Proposal, Realtime ("both realtime commit paths ... go through the gate").
#[test]
fn a_turn_commit_without_session_stt_is_gated() {
    let setup = Setup { turn: TurnMode::Manual, ..without_session_stt(true) };
    // The reply may be asked in the prediction's own millisecond: ordered by observation.
    let mut classified = lk_agents_fuzz::record::Observed::END;
    let call = Call::run(setup, async |call| {
        call.enter().await;
        call.speaks("", "Hello, who is this?").await;
        call.send(Input::CommitTurn { skip_reply: false }).await;
        call.advance(200).await;
        classified = call.run.record.now();
        call.classifier_says(AmdSays::Human).await;
        call.advance(4_000).await;
    });
    assert_eq!(call.classified(), vec!["Hello, who is this?".to_owned()]);
    let responses = call.responses();
    assert_eq!(responses.len(), 1, "{responses:?}");
    assert!(responses[0].at > classified, "{responses:?}");
    assert_eq!(call.commits().len(), 1, "{:?}", call.commits());
    call.clean();
}

// History without the model's own transcription (spec §7.2).

/// A model that does not transcribe, with no session recogniser: AMD inserts its own transcript
/// as the user item, once, whatever the prediction; the reply follows only `human`. Python:
/// `test_external_stt_preserves_history_without_native_transcription[amd-*]`.
#[test]
fn amd_supplies_the_user_item_without_native_transcription() {
    for (says, replies) in [(AmdSays::Human, 1), (AmdSays::Wait, 0), (AmdSays::Unavailable, 0)] {
        let call = Call::run(without_session_stt(false), async |call| {
            call.enter().await;
            call.speaks("", "AMD transcript.").await;
            call.classifier_says(says).await;
            call.advance(4_000).await;
        });
        assert_eq!(call.classified(), vec!["AMD transcript.".to_owned()], "{says:?}");
        assert_eq!(call.responses().len(), replies, "{says:?}");
        let users: Vec<_> = call.saved_user_words().into_iter().filter(|w| !w.is_empty()).collect();
        assert_eq!(users, vec!["AMD transcript.".to_owned()], "{says:?}: once, AMD's words");
        assert_eq!(call.commits().len(), 1, "{says:?}");
        call.clean();
    }
}

/// A model that does not transcribe, with the session's recogniser: the session's words are the
/// user item, once, and AMD inserts nothing. Python:
/// `test_external_stt_preserves_history_without_native_transcription[session-*]`.
#[test]
fn the_session_supplies_the_user_item_without_native_transcription() {
    for (says, replies) in [(AmdSays::Human, 1), (AmdSays::Wait, 0)] {
        let setup = Setup { realtime: Some(model(false)), ..detecting(AmdSetup::default()) };
        let call = Call::run(setup, async |call| {
            call.enter().await;
            call.speaks("Session transcript.", "").await;
            call.classifier_says(says).await;
            call.advance(4_000).await;
        });
        assert_eq!(call.responses().len(), replies, "{says:?}");
        let users: Vec<_> = call.saved_user_words().into_iter().filter(|w| !w.is_empty()).collect();
        assert_eq!(users, vec!["Session transcript.".to_owned()], "{says:?}: once");
        assert!(call.run.record.amd_view().inserted_items().is_empty(), "AMD inserted nothing");
        call.clean();
    }
}

// The words rule's one owner (proposal, Realtime).

/// A turn superseded while its customer hook ran keeps its words by the gate's rule: on a model
/// that transcribes, only the server's transcript, never the session recogniser's words; on one
/// that does not, the session's words, once. Python adds no STT message for a realtime turn and
/// inserts one only when the model does not transcribe (`agent_activity.py`
/// `_user_turn_completed_impl`).
#[test]
fn a_turn_superseded_in_its_hook_keeps_its_words_once() {
    for transcribes in [true, false] {
        let setup = Setup {
            hook_bound_ms: Some(10_000),
            realtime: Some(model(transcribes)),
            ..detecting(AmdSetup::default())
        };
        let call = Call::run(setup, async |call| {
            call.enter().await;
            // The first turn's hook runs; the second waits behind it, then supersedes it.
            call.speaks("is anyone there", "").await;
            call.speaks("hello", "").await;
            call.hook_returns().await;
            call.advance(100).await;
            call.hook_returns().await;
            call.classifier_says(AmdSays::Human).await;
            call.advance(4_000).await;
        });
        let users = call.saved_user_words();
        let first = users.iter().filter(|w| *w == "is anyone there").count();
        assert_eq!(first, usize::from(!transcribes), "transcribes={transcribes}: {users:?}");
        call.clean();
    }
}

/// `InsertUserItem` follows the live model: after a handoff to a model that does not transcribe,
/// with no session recogniser, AMD's words are the user item once. The runtime used to decide from
/// the model the session opened on, which transcribed, and sent nothing.
#[test]
fn amd_supplies_the_words_after_a_handoff_to_a_model_without_transcription() {
    let realtime = RealtimeSetup { handoff_user_transcription: Some(false), ..model(true) };
    let amd = AmdSetup { own_stt: true, ..AmdSetup::default() };
    let setup = Setup { stt: false, realtime: Some(realtime), ..detecting(amd) };
    let call = Call::run(setup, async |call| {
        let handoff = Input::Handoff {
            agent: "second".into(),
            instructions: AgentText("You are the second agent.".into()),
            tools: Vec::new(),
            turn: None,
        };
        call.send(handoff).await;
        call.advance(1_000).await;
        call.enter().await;
        call.speaks("", "AMD transcript.").await;
        call.classifier_says(AmdSays::Human).await;
        call.advance(4_000).await;
    });
    assert!(call.entered_ok(), "{:?}", call.run.record.amd_run().enter_answers());
    let users: Vec<_> = call.saved_user_words().into_iter().filter(|w| !w.is_empty()).collect();
    assert_eq!(users, vec!["AMD transcript.".to_owned()], "once, AMD's words");
    call.clean();
}

// A DTMF reply cut short (proposal, Realtime: the agent's tools come back when the reply's
// speech ends).

/// A model that presses a key whenever it has a tool: under a menu's reply, the only one it has
/// is AMD's `send_dtmf_events`.
fn pressing(fallback: bool) -> Setup {
    let realtime = RealtimeSetup { tool_calls: Some(100), fallback, ..model(true) };
    Setup { realtime: Some(realtime), tools: Vec::new(), ..detecting(AmdSetup::default()) }
}

impl Call {
    /// A menu, classified as one: its reply is asked once the silence ends, with the keypad.
    async fn menu_reply(&mut self) {
        self.enter().await;
        self.speaks("for billing press one for support press two", "").await;
        self.classifier_says(AmdSays::Ivr).await;
        self.advance(50).await;
        self.menu_says(AmdSays::NoMenu).await;
        self.advance(2_000).await;
    }

    /// Advances until the model pressed a key (its reply called AMD's tool), at most `ms`.
    async fn until_pressed(&mut self, ms: u64) -> bool {
        use lk_agents_voice::trace::{Step, ToolStep};
        for _ in 0..ms / 100 {
            let pressed = self.run.record.journal().iter().any(|(_, step)| {
                matches!(step, Step::Tool(ToolStep::Calls { names, .. })
                    if names.iter().any(|n| n == SEND_DTMF_TOOL))
            });
            if pressed {
                return true;
            }
            self.advance(100).await;
        }
        false
    }

    /// Every tool still running returns.
    async fn tools_return(&mut self) {
        for tool in self.run.record.live().of(Kind::Tool) {
            let answer = lk_agents_fuzz::input::ToolAnswer::Return;
            let text = AgentText(String::new());
            self.send(Input::ToolAnswered { tool: Ref::new(tool), answer, text }).await;
        }
    }

    /// The tools the last socket received last.
    fn last_tools(&self) -> Option<Vec<String>> {
        let sent = self.run.record.rt_sent();
        let socket = sent.iter().map(|(_, socket, _)| *socket).max()?;
        sent.iter().rev().find_map(|(_, s, sent)| match sent {
            RtSent::Tools(tools) if *s == socket => Some(tools.clone()),
            _ => None,
        })
    }

    fn keypad_offered(&self) -> bool {
        self.last_tools().is_some_and(|tools| tools.iter().any(|t| t == SEND_DTMF_TOOL))
    }
}

/// The menu's reply is interrupted while its key press runs: the speech ends with the tool, and
/// the agent's tools come back on the server.
#[test]
fn a_dtmf_reply_interrupted_during_its_key_press_restores_the_tools() {
    let mut pressed = false;
    let call = Call::run(pressing(false), async |call| {
        call.menu_reply().await;
        pressed = call.until_pressed(6_000).await;
        call.send(Input::Interrupt { force: true, wait: false }).await;
        call.advance(200).await;
        call.tools_return().await;
        call.advance(4_000).await;
    });
    assert!(pressed, "the model pressed a key: {:?}", call.responses());
    let responses = call.responses();
    assert!(responses.first().is_some_and(|q| q.offers(SEND_DTMF_TOOL)), "{responses:?}");
    assert!(!call.keypad_offered(), "the keypad left the server: {:?}", call.last_tools());
    call.clean();
}

/// A fallback swap while the menu's reply runs (the fake swaps to a fresh server behind the same
/// socket): the new session gets the agent's tools with its frame, the keypad only for a
/// response asked again for the reply, and the agent's tools back once the reply's speech ends.
#[test]
fn a_fallback_swap_during_a_dtmf_reply_restores_the_tools() {
    let call = Call::run(pressing(true), async |call| {
        call.menu_reply().await;
        call.until_pressed(6_000).await;
        let fatal = Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::FatalError };
        call.send(fatal).await;
        call.advance(2_000).await;
        call.tools_return().await;
        call.advance(6_000).await;
        call.tools_return().await;
        call.advance(4_000).await;
    });
    // The swap's frame: the session's instructions go to the fresh session a second time.
    let frames = (call.run.record.rt_sent().iter())
        .filter(|(_, _, sent)| matches!(sent, RtSent::Instructions(_)))
        .count();
    assert!(frames > 1, "the fallback swapped the session and sent it its frame");
    assert!(!call.keypad_offered(), "the keypad left the new server: {:?}", call.last_tools());
    call.clean();
}

/// AMD completes while the menu's reply runs: the reply plays on with the keypad, and the
/// agent's tools come back when its speech ends, after the detach.
#[test]
fn a_detach_during_a_dtmf_reply_restores_the_tools_when_the_reply_ends() {
    let mut detached_with_keypad = false;
    let call = Call::run(pressing(false), async |call| {
        call.menu_reply().await;
        call.until_pressed(6_000).await;
        detached_with_keypad = call.keypad_offered();
        call.close_amd().await;
        call.advance(200).await;
        call.tools_return().await;
        call.advance(8_000).await;
        call.tools_return().await;
        call.advance(4_000).await;
    });
    assert!(detached_with_keypad, "the reply held the keypad at the detach");
    assert!(call.detached_at().is_some(), "AMD detached");
    assert!(!call.keypad_offered(), "the keypad left the server: {:?}", call.last_tools());
    call.clean();
}
