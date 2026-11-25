//! The AMD reply gate on a realtime session with client-side turn detection: what the session
//! sends the server, and when. Every test reads the realtime commands the session emits.

use super::*;
use crate::agent_session::machine::activity::realtime::RealtimeWake;
use crate::agent_session::{Command, Now, Sent, SpeechCommand, VoiceCommandOrigin, VoiceEvent};
use crate::audio_recognition::TurnDetectionMode;
use crate::trace::Step;
use lk_agents_abi::RealtimeCommand;
use lk_agents_abi::llm::realtime::{Event, RealtimeCapabilities};
use lk_agents_abi::voice::events::{EndOfSpeechEvent, StartOfSpeechEvent};

const RUN: AmdRunId = AmdRunId(3);
const STAGE: &str = "You reached a voicemail box. Leave a short message.";

fn now(ms: u64) -> Now {
    Now { at: Ts::from_nanos(ms * 1_000_000), wall: std::time::UNIX_EPOCH }
}

/// A client-side realtime model AMD can run on: per-response tool choice, no tool auto-reply.
fn client_side() -> RealtimeCapabilities {
    RealtimeCapabilities {
        user_transcription: true,
        per_response_tool_choice: true,
        ..RealtimeCapabilities::default()
    }
}

fn driver(caps: RealtimeCapabilities) -> VoiceDriver {
    let mut driver = crate::agent_session::tests::realtime_driver();
    driver.journalled = true;
    let rt = driver.activity.realtime_activity_mut().expect("a realtime activity");
    rt.state = crate::agent_session::machine::activity::realtime::RealtimeSessionState::new(caps);
    rt.session.capabilities = caps;
    driver
}

fn host(driver: &mut VoiceDriver, ms: u64, command: Command) -> Vec<VoiceEffect> {
    let sent = Sent { origin: VoiceCommandOrigin::Host, command };
    driver.handle(Some(VoiceEvent::Host(sent)), now(ms))
}

fn answer(driver: &mut VoiceDriver, ms: u64, answer: AmdAnswer) -> Vec<VoiceEffect> {
    driver.handle(Some(VoiceEvent::Amd(answer)), now(ms))
}

fn attach(driver: &mut VoiceDriver) -> Vec<VoiceEffect> {
    let dtmf_tool = super::rule_tests::dtmf_tool();
    host(driver, 0, Command::Amd(AmdCommand::Attach { run: RUN, dtmf_tool, call: None }))
}

/// Attached and active, with no opening.
fn attached(caps: RealtimeCapabilities) -> VoiceDriver {
    let mut driver = driver(caps);
    attach(&mut driver);
    answer(&mut driver, 0, AmdAnswer::Activated { run: RUN, opening: None });
    driver
}

fn info(turn: u64, transcript: &str) -> EndOfTurnInfo {
    EndOfTurnInfo {
        turn_id: TurnId(turn),
        owner: TurnDetectionMode::Vad,
        stt_enabled: !transcript.is_empty(),
        skip_reply: false,
        new_transcript: transcript.into(),
        transcript_confidence: 0.9,
        metrics: Default::default(),
        backchannel_over_agent: false,
        late: false,
        snapshot: Default::default(),
        promoted: None,
    }
}

fn commit(driver: &mut VoiceDriver, ms: u64, turn: u64, transcript: &str) -> Vec<VoiceEffect> {
    driver.handle(None, now(ms));
    driver.on_end_of_turn(driver.step.now, info(turn, transcript));
    driver.handle(None, now(ms))
}

fn decide(turn: u64, decision: ReplyDecision) -> AmdAnswer {
    AmdAnswer::ReplyDecided { run: RUN, turn: TurnId(turn), decision }
}

fn commands(effects: &[VoiceEffect]) -> Vec<&RealtimeCommand> {
    effects
        .iter()
        .filter_map(|effect| match effect {
            VoiceEffect::Realtime { command, .. } => Some(command),
            _ => None,
        })
        .collect()
}

fn commits_audio(effects: &[VoiceEffect]) -> bool {
    commands(effects).iter().any(|c| matches!(c, RealtimeCommand::CommitAudio(_)))
}

fn replies(effects: &[VoiceEffect]) -> Vec<&lk_agents_abi::llm::realtime::GenerateReply> {
    (commands(effects).into_iter())
        .filter_map(|c| match c {
            RealtimeCommand::GenerateReply(reply) => Some(reply),
            _ => None,
        })
        .collect()
}

fn tool_names(command: &RealtimeCommand) -> Option<Vec<&str>> {
    match command {
        RealtimeCommand::UpdateTools(update) => {
            Some(update.tools.iter().map(|t| t.name.as_str()).collect())
        }
        _ => None,
    }
}

fn reply_speech(effects: &[VoiceEffect]) -> Option<SpeechId> {
    effects.iter().find_map(|effect| match effect {
        VoiceEffect::Amd { observation: AmdObservation::ReplyCreated { speech, .. }, .. } => {
            Some(*speech)
        }
        _ => None,
    })
}

fn user_items(driver: &VoiceDriver) -> Vec<String> {
    (driver.session.chat_ctx.history())
        .filter_map(|item| item.as_message())
        .filter(|message| message.role == lk_agents_abi::Role::User)
        .filter_map(|message| message.raw_text_content())
        .collect()
}

fn refusal(caps: RealtimeCapabilities) -> Option<AmdAttachRefused> {
    let mut driver = driver(caps);
    let effects = attach(&mut driver);
    steps(&effects).find_map(|step| match step {
        Step::Gate(GateStep::AttachRefused { why, .. }) => Some(*why),
        _ => None,
    })
}

fn steps(effects: &[VoiceEffect]) -> impl Iterator<Item = &Step> {
    effects.iter().filter_map(|effect| match effect {
        VoiceEffect::Journal(step) => Some(step.as_ref()),
        _ => None,
    })
}

// Catches: AMD attaching where a server-initiated reply would bypass the gate, or where a response
// cannot carry the decision's tools, or where the server answers tool results on its own (spec
// §5.1). A client-side model with per-response tools attaches.
#[test]
fn attach_takes_only_a_client_side_realtime_model_with_client_tool_replies() {
    let server = RealtimeCapabilities { turn_detection: true, ..client_side() };
    assert_eq!(refusal(server), Some(AmdAttachRefused::RealtimeServerTurnDetection));
    let shared = RealtimeCapabilities { per_response_tool_choice: false, ..client_side() };
    assert_eq!(refusal(shared), Some(AmdAttachRefused::RealtimeTools));
    let auto = RealtimeCapabilities { auto_tool_reply_generation: true, ..client_side() };
    assert_eq!(refusal(auto), Some(AmdAttachRefused::RealtimeTools));
    assert_eq!(refusal(client_side()), None);
    // A model that can hand detection to the client, and did, attaches.
    let handed = RealtimeCapabilities {
        turn_detection: true,
        can_disable_turn_detection: true,
        ..client_side()
    };
    let mut driver = driver(handed);
    driver.activity.realtime_activity_mut().expect("realtime").turn_detection_disabled = true;
    let effects = attach(&mut driver);
    assert!(steps(&effects).any(|s| matches!(s, Step::Gate(GateStep::Attached { .. }))));
}

// Catches: a gated realtime turn asking the server for a reply before its decision, or a denied
// turn's audio left in the buffer to prefix the next allowed reply. The audio is committed at
// commit; nothing is asked on a denial.
#[test]
fn a_gated_turn_commits_its_audio_at_commit_and_asks_nothing_until_allowed() {
    let mut driver = attached(client_side());
    let effects = commit(&mut driver, 10, 1, "");
    assert!(commits_audio(&effects), "the audio becomes the server's user item at commit");
    assert!(replies(&effects).is_empty(), "no response before the decision");
    assert!(effects.iter().any(|e| matches!(
        e,
        VoiceEffect::Amd { observation: AmdObservation::TurnCommitted { turn: TurnId(1), .. }, .. }
    )));

    let effects = answer(&mut driver, 20, decide(1, ReplyDecision::deny()));
    assert!(replies(&effects).is_empty(), "a denied turn asks for no response");
    assert!(!commits_audio(&effects), "the audio was committed once, at commit");

    commit(&mut driver, 30, 2, "");
    let effects = answer(&mut driver, 40, decide(2, ReplyDecision::allow()));
    let asked = replies(&effects);
    assert_eq!(asked.len(), 1, "an allowed turn asks once");
    assert_eq!(asked[0].instructions, None, "a plain allow adds no instructions");
    assert!(!commits_audio(&effects), "the reply does not commit the buffer again");
}

// Catches: the decision's stage instructions not reaching the server, or AMD's DTMF tool missing
// from (or outliving) the response. The tools go up before the request, and come back to the
// agent's when the reply's speech ends (spec §10.2, §11.1).
#[test]
fn an_allowed_reply_carries_its_stage_instructions_and_dtmf_tool() {
    let mut driver = attached(client_side());
    commit(&mut driver, 10, 1, "");
    let decision =
        ReplyDecision { allow: true, instructions: Some(STAGE.into()), dtmf: true, silence: None };
    let effects = answer(&mut driver, 20, decide(1, decision));
    let sent = commands(&effects);
    let tools = sent.iter().position(|c| {
        tool_names(c).is_some_and(|names| names.contains(&super::rule_tests::SEND_DTMF_TOOL))
    });
    let reply = sent.iter().position(|c| matches!(c, RealtimeCommand::GenerateReply(_)));
    let (Some(tools), Some(reply)) = (tools, reply) else {
        panic!("the DTMF tools and the request are both sent: {sent:?}")
    };
    assert!(tools < reply, "the tools reach the server before the request");
    assert_eq!(replies(&effects)[0].instructions.as_deref(), Some(STAGE), "passed unchanged");

    let speech = reply_speech(&effects).expect("the decision made a reply");
    driver.on_realtime(driver.step.now, RealtimeWake::PlayerEnded(speech));
    let effects = driver.handle(None, now(30));
    let restored = commands(&effects).into_iter().find_map(tool_names);
    assert_eq!(restored, Some(vec![]), "the agent's own tools come back");
}

// Catches: a realtime reply generating during the silence hold (spec §10.3: a realtime reply
// starts generating only after the silence).
#[test]
fn a_reply_held_for_silence_asks_the_server_after_the_silence() {
    let mut driver = attached(client_side());
    commit(&mut driver, 10, 1, "");
    let silence = Duration::from_millis(1500);
    let decision = ReplyDecision { silence: Some(silence), ..ReplyDecision::allow() };
    let effects = answer(&mut driver, 20, decide(1, decision));
    assert!(replies(&effects).is_empty(), "held: nothing is asked yet");
    let effects = driver.handle(None, now(1_600));
    assert_eq!(replies(&effects).len(), 1, "the silence ended: the server is asked once");
}

// Catches: `commit_user_turn()` on a realtime session without session STT bypassing the gate:
// it used to ask the server at once and never commit a turn AMD could classify.
#[test]
fn a_host_commit_without_session_stt_goes_through_the_gate() {
    let mut driver = attached(client_side());
    let commit = SpeechCommand::CommitUserTurn {
        transcript_timeout: Duration::from_secs(2),
        stt_flush_duration: Duration::from_millis(100),
        skip_reply: false,
        call: None,
    };
    let effects = host(&mut driver, 10, Command::Speech(commit));
    assert!(replies(&effects).is_empty(), "nothing is asked at the command");
    // The manual commit lands after the endpointing delay.
    let effects = driver.handle(None, now(600));
    assert!(effects.iter().any(|e| matches!(
        e,
        VoiceEffect::Amd { observation: AmdObservation::TurnCommitted { .. }, .. }
    )));
    assert!(commits_audio(&effects));
    assert!(replies(&effects).is_empty(), "the reply waits for the decision");
}

// Catches: AMD's transcript added for a model whose server transcribes the user, which would add
// the turn's words twice; and missing for one that does not (spec §7.2).
#[test]
fn amd_inserts_the_user_item_only_for_a_model_without_transcription_or_stt() {
    let insert = AmdAnswer::InsertUserItem { run: RUN, turn: TurnId(1), text: "hello?".into() };

    let mut driver = attached(client_side());
    commit(&mut driver, 10, 1, "");
    let effects = answer(&mut driver, 20, insert.clone());
    assert!(steps(&effects).any(|s| matches!(
        s,
        Step::Gate(GateStep::AnswerDropped { why: AnswerDrop::NotNeeded, .. })
    )));
    assert!(user_items(&driver).is_empty());

    let deaf = RealtimeCapabilities { user_transcription: false, ..client_side() };
    let mut driver = attached(deaf);
    commit(&mut driver, 10, 1, "");
    answer(&mut driver, 20, insert);
    assert_eq!(user_items(&driver), ["hello?"]);
}

// Catches: a realtime turn adding the session STT's words next to the server's own transcript.
// Only a model without user transcription keeps them, on a denial as on an allow.
#[test]
fn a_gated_turn_keeps_its_stt_words_only_without_server_transcription() {
    let mut driver = attached(client_side());
    driver.activity.audio_recognition.set_legs(true, false);
    commit(&mut driver, 10, 1, "is anyone there");
    answer(&mut driver, 20, decide(1, ReplyDecision::deny()));
    assert!(user_items(&driver).is_empty(), "the server's transcript fills the item");

    let deaf = RealtimeCapabilities { user_transcription: false, ..client_side() };
    let mut driver = attached(deaf);
    driver.activity.audio_recognition.set_legs(true, false);
    commit(&mut driver, 10, 1, "is anyone there");
    answer(&mut driver, 20, decide(1, ReplyDecision::deny()));
    commit(&mut driver, 30, 2, "hello");
    answer(&mut driver, 40, decide(2, ReplyDecision::allow()));
    assert_eq!(user_items(&driver), ["is anyone there", "hello"]);
}

// Catches: participant speech during the hold not restarting the realtime reply's silence.
#[test]
fn speech_during_the_hold_restarts_the_silence_before_the_request() {
    let mut driver = attached(client_side());
    commit(&mut driver, 10, 1, "");
    let decision =
        ReplyDecision { silence: Some(Duration::from_millis(1500)), ..ReplyDecision::allow() };
    answer(&mut driver, 20, decide(1, decision));
    let at = |ms: u64| Ts::from_nanos(ms * 1_000_000);
    driver.handle(None, now(1_000));
    let start =
        StartOfSpeechEvent { at: at(1000), speech_duration: Duration::ZERO, from_vad: true };
    driver.publish_event(SessionEvent::StartOfSpeech(start));
    driver.handle(None, now(1_000));
    let effects = driver.handle(None, now(1_600));
    assert!(replies(&effects).is_empty(), "speech restarted the wait");
    let end = EndOfSpeechEvent { at: at(2000), from_vad: true, silence_duration: 0.0 };
    driver.publish_event(SessionEvent::EndOfSpeech(end));
    driver.handle(None, now(2_000));
    let effects = driver.handle(None, now(3_400));
    assert!(replies(&effects).is_empty(), "the new silence has not run out");
    let effects = driver.handle(None, now(3_600));
    assert_eq!(replies(&effects).len(), 1);
}

// Catches: a realtime opening dropped for want of a pipeline TTS or LLM (spec §10.6): the model
// says the text itself, or generates from the instructions with no tools; both bypass the hold.
#[test]
fn a_realtime_opening_is_said_or_generated_by_the_model() {
    let says = RealtimeCapabilities { supports_say: true, ..client_side() };
    let mut first = driver(says);
    attach(&mut first);
    let opening = Some(Opening::Text("Hi, this is Ada from the clinic.".into()));
    let effects = answer(&mut first, 0, AmdAnswer::Activated { run: RUN, opening });
    assert!(steps(&effects).any(|s| matches!(s, Step::Gate(GateStep::OpeningBypass { .. }))));
    assert!(commands(&effects).iter().any(|c| matches!(c,
        RealtimeCommand::Say(say) if say.text == "Hi, this is Ada from the clinic.")));

    let mut driver = driver(client_side());
    attach(&mut driver);
    let opening = Some(Opening::Instructions("Greet the callee.".into()));
    let effects = answer(&mut driver, 0, AmdAnswer::Activated { run: RUN, opening });
    let asked = replies(&effects);
    assert_eq!(asked.len(), 1, "the opening is asked at once, past the hold");
    assert_eq!(asked[0].instructions.as_deref(), Some("Greet the callee."));
    assert_eq!(asked[0].tool_choice, Some(lk_agents_abi::ToolChoice::Forbidden));
}

// Catches: a realtime fallback swap creating a reply AMD never allowed (spec §5.3); held by the
// gate, it would play once AMD detaches. Without AMD, the same swap regenerates the reply it cut.
#[test]
fn a_model_swap_regenerates_nothing_while_amd_is_attached() {
    let swap = |mut driver: VoiceDriver| {
        driver.set_agent_state(driver.step.now, AgentState::Thinking);
        let swapped = Event::SessionSwapped { regenerate: true };
        driver.on_realtime(driver.step.now, RealtimeWake::Event(swapped));
        driver.handle(None, now(30));
        driver.activity.speeches().count()
    };
    assert_eq!(swap(driver(client_side())), 1, "a session without AMD regenerates");
    assert_eq!(swap(attached(client_side())), 0, "AMD decides every reply");
}

// Catches: a backchannel over agent speech reaching AMD as a turn on a realtime model without
// session STT whose server detection the client switched off (spec §5.3).
#[test]
fn a_backchannel_over_agent_speech_never_reaches_amd() {
    let handed = RealtimeCapabilities {
        turn_detection: true,
        can_disable_turn_detection: true,
        ..client_side()
    };
    let mut driver = driver(handed);
    driver.activity.realtime_activity_mut().expect("realtime").turn_detection_disabled = true;
    attach(&mut driver);
    answer(&mut driver, 0, AmdAnswer::Activated { run: RUN, opening: None });
    // The adaptive detector judged the overlap, VAD detects the turns (a manual commit is never a
    // backchannel); set after the attach, which re-resolves both.
    driver.session.config.effective.turn.interruption.detection_enabled = Some(true);
    let eff =
        driver.activity.audio_recognition.set_turn_detection_mode(Ts::ZERO, TurnDetectionMode::Vad);
    driver.apply_recognition_effects(eff);
    driver.handle(None, now(10));
    let info = EndOfTurnInfo { backchannel_over_agent: true, ..info(1, "") };
    driver.on_end_of_turn(driver.step.now, info);
    let effects = driver.handle(None, now(10));
    assert!(!effects.iter().any(|e| matches!(
        e,
        VoiceEffect::Amd { observation: AmdObservation::TurnCommitted { .. }, .. }
    )));
    assert!(!commits_audio(&effects), "its audio stays out of the next turn");
}
