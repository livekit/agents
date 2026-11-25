//! The AMD gate over a realtime model (step 4 of the multi-turn AMD proposal): the entry checks
//! of spec §5.1, both realtime commit paths gated, the server buffer committed at commit, the
//! response asked once at authorization with the decision's per-response instructions and tools,
//! `InsertUserItem`, and the realtime session rules while attached (spec §5.3). Outcomes are read
//! at what the realtime session was sent (`VoiceEffect::Realtime`) and at the chat record.
//! Expected values come from `voice/amd/detector.py`, `voice/agent_activity.py` and
//! `tests/test_amd.py` on `main`.

use super::*;
use crate::agent_session::machine::activity::realtime::RealtimeWake;
use crate::agent_session::{Command, Now, RequestId, Sent, SpeechCommand, TurnCommand, VoiceEvent};
use crate::audio_recognition::TurnDetectionMode;
use lk_agents_abi::RealtimeCommand;
use lk_agents_abi::llm::realtime as rt;

const RUN: AmdRunId = AmdRunId(9);
const SEND_DTMF_TOOL: &str = "send_dtmf_events";

fn at(ms: u64) -> Ts {
    Ts::from_nanos(ms * 1_000_000)
}

fn now(ms: u64) -> Now {
    Now { at: at(ms), wall: std::time::UNIX_EPOCH }
}

/// What the realtime model reports and how the session resolved its turn detection.
#[derive(Clone, Copy)]
struct Model {
    /// The model detects turns on the server.
    server_vad: bool,
    /// The session turned the server's detection off (client-side detection).
    vad_disabled: bool,
    per_response_tool_choice: bool,
    auto_tool_reply_generation: bool,
    user_transcription: bool,
}

/// What AMD needs (spec §5.1): client-side detection over a model whose server detection is
/// switched off, per-response tools, client tool replies; with user transcription.
const CLIENT_SIDE: Model = Model {
    server_vad: true,
    vad_disabled: true,
    per_response_tool_choice: true,
    auto_tool_reply_generation: false,
    user_transcription: true,
};

/// A realtime session over `model`, with or without session STT.
fn rt_driver(model: Model, session_stt: bool) -> VoiceDriver {
    let mut driver = crate::agent_session::tests::realtime_driver();
    driver.journalled = true;
    let rt = driver.activity.realtime_activity_mut().expect("a realtime activity");
    let caps = &mut rt.session.capabilities;
    caps.turn_detection = model.server_vad;
    caps.can_disable_turn_detection = model.vad_disabled;
    caps.per_response_tool_choice = model.per_response_tool_choice;
    caps.auto_tool_reply_generation = model.auto_tool_reply_generation;
    caps.user_transcription = model.user_transcription;
    caps.audio_output = true;
    caps.mutable_tools = true;
    caps.mutable_instructions = true;
    caps.mutable_chat_context = true;
    rt.state = crate::agent_session::machine::activity::realtime::RealtimeSessionState::new(
        rt.session.capabilities,
    );
    rt.turn_detection_disabled = model.server_vad && model.vad_disabled;
    rt.resolved_mode = TurnDetectionMode::Vad;
    driver.activity.audio_recognition.set_legs(session_stt, true);
    let _ =
        driver.activity.audio_recognition.set_turn_detection_mode(Ts::ZERO, TurnDetectionMode::Vad);
    driver.session.stt = session_stt.then(|| lk_agents_abi::resources::SttRef {
        id: lk_agents_abi::resources::ResourceId(2),
        provider: lk_agents_abi::ProviderId::of_label("stt"),
        capabilities: Default::default(),
        instance: 2,
    });
    driver
}

fn host(driver: &mut VoiceDriver, ms: u64, command: Command) -> Vec<VoiceEffect> {
    let sent = Sent { origin: crate::agent_session::VoiceCommandOrigin::Host, command };
    driver.handle(Some(VoiceEvent::Host(sent)), now(ms))
}

fn answer(driver: &mut VoiceDriver, ms: u64, answer: AmdAnswer) -> Vec<VoiceEffect> {
    driver.handle(Some(VoiceEvent::Amd(answer)), now(ms))
}

/// The session's answer to an attach with or without AMD's own STT.
fn attach(driver: &mut VoiceDriver, amd_stt: bool) -> Result<(), AmdAttachRefused> {
    let attach = AmdCommand::Attach {
        run: RUN,
        dtmf_tool: super::rule_tests::dtmf_tool(),
        amd_stt,
        call: Some(RequestId(77)),
    };
    let effects = host(driver, 0, Command::Amd(attach));
    effects
        .iter()
        .find_map(|effect| match effect {
            VoiceEffect::Answer(RequestId(77), crate::agent_session::Answer::AmdAttach(result)) => {
                Some(*result)
            }
            _ => None,
        })
        .expect("the attach is answered")
}

/// A realtime session with the run attached and active. Without session STT, AMD has its own.
fn attached(model: Model, session_stt: bool) -> VoiceDriver {
    let mut driver = rt_driver(model, session_stt);
    attach(&mut driver, !session_stt).expect("a client-side realtime model is accepted");
    answer(&mut driver, 0, AmdAnswer::Activated { run: RUN, opening: None });
    // The attach re-resolves the options; the session's own detection ends turns.
    let _ =
        driver.activity.audio_recognition.set_turn_detection_mode(at(0), TurnDetectionMode::Vad);
    driver
}

fn info(turn: u64, transcript: &str, stt: bool) -> EndOfTurnInfo {
    EndOfTurnInfo {
        turn_id: TurnId(turn),
        owner: TurnDetectionMode::Vad,
        stt_enabled: stt,
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

/// A turn the session's own detection ended, at `ms`.
fn end_of_turn(driver: &mut VoiceDriver, ms: u64, info: EndOfTurnInfo) -> Vec<VoiceEffect> {
    driver.handle(None, now(ms));
    driver.on_end_of_turn(at(ms), info);
    driver.handle(None, now(ms))
}

fn decide(turn: u64, decision: ReplyDecision) -> AmdAnswer {
    AmdAnswer::ReplyDecided { run: RUN, turn: TurnId(turn), decision }
}

fn sent(effects: &[VoiceEffect]) -> Vec<&RealtimeCommand> {
    effects
        .iter()
        .filter_map(|effect| match effect {
            VoiceEffect::Realtime { command, .. } => Some(command),
            _ => None,
        })
        .collect()
}

/// The response requests the server was sent.
fn responses(effects: &[VoiceEffect]) -> Vec<&rt::GenerateReply> {
    sent(effects)
        .into_iter()
        .filter_map(|command| match command {
            RealtimeCommand::GenerateReply(request) => Some(request),
            _ => None,
        })
        .collect()
}

fn buffer_commits(effects: &[VoiceEffect]) -> usize {
    sent(effects).iter().filter(|c| matches!(c, RealtimeCommand::CommitAudio(_))).count()
}

/// Each session tool update, as the tool names it sets.
fn tool_updates(effects: &[VoiceEffect]) -> Vec<Vec<String>> {
    sent(effects)
        .into_iter()
        .filter_map(|command| match command {
            RealtimeCommand::UpdateTools(update) => {
                Some(update.tools.iter().map(|tool| tool.name.clone()).collect())
            }
            _ => None,
        })
        .collect()
}

fn observations(effects: &[VoiceEffect]) -> Vec<&AmdObservation> {
    effects
        .iter()
        .filter_map(|effect| match effect {
            VoiceEffect::Amd { run, observation } if *run == RUN => Some(observation),
            _ => None,
        })
        .collect()
}

fn committed(effects: &[VoiceEffect]) -> Vec<TurnId> {
    observations(effects)
        .into_iter()
        .filter_map(|o| match o {
            AmdObservation::TurnCommitted { turn, .. } => Some(*turn),
            _ => None,
        })
        .collect()
}

fn replies(effects: &[VoiceEffect]) -> Vec<(TurnId, SpeechId)> {
    observations(effects)
        .into_iter()
        .filter_map(|o| match o {
            AmdObservation::ReplyCreated { turn, speech } => Some((*turn, *speech)),
            _ => None,
        })
        .collect()
}

fn user_items(driver: &VoiceDriver) -> Vec<String> {
    driver
        .session
        .chat_ctx
        .history()
        .filter_map(|item| item.as_message())
        .filter(|message| message.role == lk_agents_abi::Role::User)
        .filter_map(|message| message.raw_text_content())
        .collect()
}

fn items_added(effects: &[VoiceEffect], text: &str) -> usize {
    effects
        .iter()
        .filter(|effect| {
            matches!(effect,
            VoiceEffect::Emit(SessionEvent::ConversationItemAdded { item, .. })
                if item.as_message().and_then(|m| m.raw_text_content()).as_deref() == Some(text))
        })
        .count()
}

/// Every effect from `ms` to `until` in `step` ms steps.
fn run_until(driver: &mut VoiceDriver, from: u64, until: u64, step: u64) -> Vec<VoiceEffect> {
    let mut effects = Vec::new();
    let mut ms = from;
    while ms <= until {
        effects.extend(driver.handle(None, now(ms)));
        ms += step;
    }
    effects
}

/// The server answers `request`: the response is created and done, its player ends and its
/// generation finishes.
fn finish_response(
    driver: &mut VoiceDriver,
    ms: u64,
    request: &str,
    speech: SpeechId,
) -> Vec<VoiceEffect> {
    let response = format!("resp_{request}");
    let mut effects = Vec::new();
    for event in [
        rt::Event::GenerationCreated(rt::GenerationCreated {
            response_id: response.clone(),
            req_id: Some(request.to_owned()),
        }),
        rt::Event::GenerationDone(rt::GenerationDone {
            response_id: response.clone(),
            status: rt::GenerationStatus::Completed,
            usage: Default::default(),
        }),
    ] {
        driver.on_realtime(at(ms), RealtimeWake::Event(event));
        effects.extend(driver.handle(None, now(ms)));
    }
    driver.on_realtime(at(ms), RealtimeWake::PlayerEnded(speech));
    effects.extend(driver.handle(None, now(ms)));
    let outcome = crate::generation::TurnOutcome {
        forwarded_text: String::new(),
        assistant_message: None,
        cut_at: None,
    };
    let done = crate::generation::GenEvent::Done {
        speech,
        outcome: Box::new(outcome),
        interrupted: false,
    };
    driver.on_generation(at(ms), done);
    effects.extend(driver.handle(None, now(ms)));
    effects.extend(run_until(driver, ms, ms + 500, 50));
    effects
}

// --- Entry (spec §5.1, `detector.py` `__aenter__`) ---

// Catches: a realtime model whose server detects turns accepted, whose server-initiated replies
// never reach the gate; and a client-side model refused with it. Python checks the detection
// first, so a model failing every check reports the detection.
#[test]
fn attach_refuses_a_realtime_model_with_server_turn_detection() {
    let server = Model { vad_disabled: false, ..CLIENT_SIDE };
    let mut driver = rt_driver(server, true);
    assert_eq!(attach(&mut driver, true), Err(AmdAttachRefused::RealtimeServerTurnDetection));
    assert!(driver.activity.gate.is_none());

    let every_check_fails = Model {
        vad_disabled: false,
        per_response_tool_choice: false,
        auto_tool_reply_generation: true,
        ..CLIENT_SIDE
    };
    let mut driver = rt_driver(every_check_fails, false);
    assert_eq!(attach(&mut driver, false), Err(AmdAttachRefused::RealtimeServerTurnDetection));

    // A model whose server cannot detect turns at all is client-side.
    let no_server_vad = Model { server_vad: false, vad_disabled: false, ..CLIENT_SIDE };
    assert_eq!(attach(&mut rt_driver(no_server_vad, true), false), Ok(()));
    assert_eq!(attach(&mut rt_driver(CLIENT_SIDE, true), false), Ok(()));
}

// Catches: a realtime model accepted that cannot take AMD's tools and instructions per response,
// or that answers its own tool calls past the gate (spec §5.1 rule 2).
#[test]
fn attach_refuses_a_realtime_model_without_per_response_tools_or_client_tool_replies() {
    let no_per_response = Model { per_response_tool_choice: false, ..CLIENT_SIDE };
    let mut driver = rt_driver(no_per_response, true);
    assert_eq!(attach(&mut driver, false), Err(AmdAttachRefused::RealtimeTools));
    let auto_tool_reply = Model { auto_tool_reply_generation: true, ..CLIENT_SIDE };
    let mut driver = rt_driver(auto_tool_reply, true);
    assert_eq!(attach(&mut driver, false), Err(AmdAttachRefused::RealtimeTools));
    assert!(driver.activity.gate.is_none());
}

// Catches: a realtime run with nothing to transcribe the callee, or one refused although AMD
// brings its own STT. A pipeline session needs neither: its LLM classifies session finals or
// AMD's.
#[test]
fn attach_refuses_a_realtime_model_with_neither_session_stt_nor_amd_stt() {
    let mut driver = rt_driver(CLIENT_SIDE, false);
    assert_eq!(attach(&mut driver, false), Err(AmdAttachRefused::RealtimeNoStt));
    assert!(driver.activity.gate.is_none());
    assert_eq!(attach(&mut rt_driver(CLIENT_SIDE, false), true), Ok(()), "AMD's own STT");
    assert_eq!(attach(&mut rt_driver(CLIENT_SIDE, true), false), Ok(()), "session STT");
    let mut pipeline = crate::agent_session::tests::driver();
    assert!(
        pipeline.activity.realtime().is_none()
            && !pipeline.activity.audio_recognition.stt_enabled()
    );
    assert_eq!(attach(&mut pipeline, false), Ok(()), "the rule is for realtime models only");
}

// --- Commit paths ---

// Catches: a client-detected realtime turn asking the server for a response before AMD decided
// it, or its audio left in the buffer until the reply, so a denied turn's audio prefixes the next
// reply (Python commits the buffer in `_user_turn_completed_impl`, before `should_reply`).
#[test]
fn a_client_detected_turn_commits_the_buffer_at_commit_and_asks_only_after_an_allow() {
    let mut driver = attached(CLIENT_SIDE, true);
    let effects = end_of_turn(&mut driver, 10, info(1, "hello?", true));
    assert_eq!(committed(&effects), [TurnId(1)], "the turn goes through the gate");
    assert_eq!(buffer_commits(&effects), 1, "the buffer is committed at commit");
    assert!(responses(&effects).is_empty(), "no response before the decision");

    let effects = answer(&mut driver, 20, decide(1, ReplyDecision::allow()));
    assert_eq!(replies(&effects).len(), 1);
    let asked = responses(&effects);
    assert_eq!(asked.len(), 1, "one response request at authorization: {effects:?}");
    assert_eq!(asked[0].instructions, None, "a plain allow adds no instructions");
    assert_eq!(buffer_commits(&effects), 0, "the buffer is not committed again");
    assert!(tool_updates(&effects).is_empty(), "a plain allow keeps the session tools");
    assert!(responses(&run_until(&mut driver, 30, 2000, 100)).is_empty(), "asked once");
}

// Catches: the realtime `commit_user_turn` path (no session STT) skipping the gate, as it does
// without AMD: it forces `skip_reply` and asks the server at once (spec §5.3, proposal
// "Realtime").
#[test]
fn commit_user_turn_without_session_stt_goes_through_the_gate() {
    let mut driver = attached(CLIENT_SIDE, false);
    let commit = SpeechCommand::CommitUserTurn {
        transcript_timeout: Duration::ZERO,
        stt_flush_duration: Duration::ZERO,
        skip_reply: false,
        call: None,
    };
    let mut effects = host(&mut driver, 10, Command::Speech(commit));
    effects.extend(driver.handle(None, now(10)));
    let turns = committed(&effects);
    assert_eq!(turns.len(), 1, "one gated turn: {effects:?}");
    assert_eq!(buffer_commits(&effects), 1);
    assert!(responses(&effects).is_empty(), "no response before the decision");

    let effects = answer(&mut driver, 20, decide(turns[0].0, ReplyDecision::allow()));
    assert_eq!(responses(&effects).len(), 1);
    assert_eq!(buffer_commits(&effects), 0);
}

// Catches: the realtime `on_turn(Commit)` path (`realtime_user_turn`) skipping the gate.
#[test]
fn the_turn_commit_command_goes_through_the_gate() {
    let mut driver = attached(CLIENT_SIDE, false);
    let mut effects =
        host(&mut driver, 10, Command::Turn(TurnCommand::Commit { skip_reply: false }));
    effects.extend(driver.handle(None, now(10)));
    let turns = committed(&effects);
    assert_eq!(turns.len(), 1, "one gated turn: {effects:?}");
    assert_eq!(buffer_commits(&effects), 1);
    assert!(responses(&effects).is_empty(), "no response before the decision");

    let effects = answer(&mut driver, 20, decide(turns[0].0, ReplyDecision::allow()));
    assert_eq!(responses(&effects).len(), 1);
}

// Catches: a denied realtime turn still answered by the server (spec §14.3: `wait` never
// replies), or its buffer left uncommitted.
#[test]
fn a_denied_realtime_turn_sends_no_response() {
    let mut driver = attached(CLIENT_SIDE, true);
    let effects = end_of_turn(&mut driver, 10, info(1, "please hold", true));
    assert_eq!(buffer_commits(&effects), 1, "the denied turn's audio is committed, as in Python");
    let mut effects = answer(&mut driver, 20, decide(1, ReplyDecision::deny()));
    effects.extend(run_until(&mut driver, 30, 3000, 100));
    assert!(responses(&effects).is_empty(), "a denied turn has no response: {effects:?}");
    assert!(replies(&effects).is_empty());

    // The next allowed turn is asked once.
    let effects = end_of_turn(&mut driver, 3100, info(2, "hello, who is this?", true));
    assert_eq!(buffer_commits(&effects), 1);
    let effects = answer(&mut driver, 3200, decide(2, ReplyDecision::allow()));
    assert_eq!(responses(&effects).len(), 1);
}

// Catches: a realtime reply generated while the callee may still speak (spec §10.3: "A realtime
// reply starts generating only after the silence").
#[test]
fn a_realtime_reply_held_for_silence_is_asked_only_when_the_silence_ends() {
    let mut driver = attached(CLIENT_SIDE, true);
    end_of_turn(&mut driver, 0, info(1, "please leave a message after the tone", true));
    let silence =
        ReplyDecision { silence: Some(Duration::from_millis(1500)), ..ReplyDecision::allow() };
    let mut effects = answer(&mut driver, 100, decide(1, silence));
    assert_eq!(replies(&effects).len(), 1, "the reply exists, held");
    // No boundary since the commit: the hold runs from the commit, to 1.5 s.
    effects.extend(run_until(&mut driver, 200, 1400, 100));
    assert!(responses(&effects).is_empty(), "nothing is generated during the hold: {effects:?}");
    let effects = run_until(&mut driver, 1500, 1600, 100);
    assert_eq!(responses(&effects).len(), 1, "asked when the silence ends");
}

// Catches: a held realtime reply asked of the server after a newer turn superseded it (spec
// §10.3: AMD drops it and does not resend it; the newer turn's reply replaces it).
#[test]
fn a_superseded_held_realtime_reply_is_never_asked() {
    let mut driver = attached(CLIENT_SIDE, true);
    end_of_turn(&mut driver, 0, info(1, "please hold", true));
    let silence =
        ReplyDecision { silence: Some(Duration::from_millis(1500)), ..ReplyDecision::allow() };
    answer(&mut driver, 100, decide(1, silence.clone()));
    let mut effects = end_of_turn(&mut driver, 800, info(2, "your call is important", true));
    assert_eq!(buffer_commits(&effects), 1);
    effects.extend(run_until(&mut driver, 900, 4000, 100));
    assert!(responses(&effects).is_empty(), "the superseded reply is never asked: {effects:?}");

    let effects = answer(&mut driver, 4100, decide(2, ReplyDecision::allow()));
    assert_eq!(responses(&effects).len(), 1, "the newer turn's reply is asked once");
}

// --- Reply payload ---

// Catches: stage instructions lost, rewritten or written into the session's instructions for a
// realtime reply (spec §10.2: "per-response instructions for the reply"; proposal Decisions:
// "passed through unchanged").
#[test]
fn stage_instructions_ride_the_response_request_unchanged() {
    let mut driver = attached(CLIENT_SIDE, true);
    end_of_turn(&mut driver, 0, info(1, "please state your name", true));
    let stage = "  You reached a call screener.\nState your name and purpose briefly.  ";
    let decision = ReplyDecision { instructions: Some(stage.into()), ..ReplyDecision::allow() };
    let effects = answer(&mut driver, 100, decide(1, decision));
    let asked = responses(&effects);
    assert_eq!(asked.len(), 1);
    assert_eq!(asked[0].instructions.as_deref(), Some(stage), "byte for byte");
    assert!(
        !sent(&effects).iter().any(|c| matches!(c, RealtimeCommand::UpdateInstructions(_))),
        "the session's base instructions do not change"
    );
    let saved = driver.session.chat_ctx.history().any(|item| {
        item.as_message()
            .and_then(|m| m.raw_text_content())
            .is_some_and(|text| text.contains("call screener"))
    });
    assert!(!saved, "stage instructions never enter the record");

    // The next reply carries only its own decision's instructions.
    end_of_turn(&mut driver, 3000, info(2, "this is Bob", true));
    let effects = answer(&mut driver, 3100, decide(2, ReplyDecision::allow()));
    assert_eq!(
        responses(&effects)[..].iter().map(|r| &r.instructions).collect::<Vec<_>>(),
        [&None]
    );
}

// Catches: a realtime tool follow-up losing the stage instructions (spec §10.2: "per-response
// instructions for the reply and its tool follow-ups"; Python passes `instructions` to the
// follow-up's `_realtime_reply_task`), or the DTMF tool restored before the follow-up.
#[test]
fn a_tool_follow_up_keeps_the_stage_instructions_and_tools() {
    let mut driver = attached(CLIENT_SIDE, true);
    end_of_turn(&mut driver, 0, info(1, "press one for sales", true));
    let ivr = ReplyDecision {
        instructions: Some("Navigate the menu.".into()),
        dtmf: true,
        ..ReplyDecision::allow()
    };
    let effects = answer(&mut driver, 100, decide(1, ivr));
    let [(_, speech)] = replies(&effects)[..] else { panic!("one reply: {effects:?}") };
    let request = responses(&effects).first().map(|r| r.req_id.clone()).expect("asked");
    let created = rt::GenerationCreated { response_id: "resp_ivr".into(), req_id: Some(request) };
    driver.on_realtime(at(200), RealtimeWake::Event(rt::Event::GenerationCreated(created)));
    let mut effects = driver.handle(None, now(200));
    let generated = rt::GenerationDone {
        response_id: "resp_ivr".into(),
        status: rt::GenerationStatus::Completed,
        usage: Default::default(),
    };
    driver.on_realtime(at(250), RealtimeWake::Event(rt::Event::GenerationDone(generated)));
    effects.extend(driver.handle(None, now(250)));
    // The reply pressed a key; its tool asks for a follow-up.
    let done = RealtimeWake::ToolsDone { speech, step: 1, ask: true, tool_choice: None };
    driver.on_realtime(at(300), done);
    effects.extend(driver.handle(None, now(300)));
    let follow_ups = responses(&effects);
    assert_eq!(follow_ups.len(), 1, "one follow-up: {effects:?}");
    assert_eq!(follow_ups[0].instructions.as_deref(), Some("Navigate the menu."));
    assert!(
        tool_updates(&effects).iter().all(|tools| tools.iter().any(|t| t == SEND_DTMF_TOOL)),
        "the DTMF tool stays for the follow-up: {effects:?}"
    );
}

// Catches: AMD's DTMF tool missing from a `machine-ivr` realtime reply because
// `SpeechSpec.toolset` alone does not reach the server, or left on the session after the reply
// (spec §11.1, §14.9; proposal "Realtime").
#[test]
fn a_dtmf_reply_sets_the_session_tools_for_its_response_and_restores_them() {
    let mut driver = attached(CLIENT_SIDE, true);
    driver.activity.toolset = std::sync::Arc::new(crate::tool_executor::Toolset {
        specs: vec![lk_agents_abi::ToolSpec {
            name: "lookup_account".into(),
            description: "Look up the account.".into(),
            json_schema: serde_json::json!({"type": "object"}),
            flags: Default::default(),
            provider_tool: None,
        }],
        host: None,
        agent: None,
    });
    end_of_turn(&mut driver, 0, info(1, "press one for sales, two for support", true));
    let ivr = ReplyDecision {
        instructions: Some("Navigate the menu.".into()),
        dtmf: true,
        ..ReplyDecision::allow()
    };
    let effects = answer(&mut driver, 100, decide(1, ivr));
    let [(_, speech)] = replies(&effects)[..] else { panic!("one reply: {effects:?}") };
    let commands = sent(&effects);
    let update = commands.iter().position(|c| {
        matches!(c, RealtimeCommand::UpdateTools(u)
            if u.tools.iter().any(|t| t.name == SEND_DTMF_TOOL))
    });
    let asked = commands.iter().position(|c| matches!(c, RealtimeCommand::GenerateReply(_)));
    let (Some(update), Some(asked)) = (update, asked) else {
        panic!("the DTMF tool is set and the response asked: {commands:?}")
    };
    assert!(update < asked, "the tools are set before the response is asked");
    let mut with_dtmf = tool_updates(&effects)[0].clone();
    with_dtmf.sort();
    assert_eq!(with_dtmf, ["lookup_account", SEND_DTMF_TOOL], "the agent's tools plus DTMF");
    assert!(
        tool_updates(&effects).last().is_some_and(|t| t.iter().any(|n| n == SEND_DTMF_TOOL)),
        "not restored before the response runs"
    );
    let request = responses(&effects)[0].req_id.clone();
    assert_eq!(responses(&effects)[0].instructions.as_deref(), Some("Navigate the menu."));

    let effects = finish_response(&mut driver, 600, &request, speech);
    assert_eq!(
        tool_updates(&effects).last().map(Vec::as_slice),
        Some(["lookup_account".to_owned()].as_slice()),
        "restored after the response: {effects:?}"
    );
}

// Catches: a reply whose decision does not allow DTMF offered AMD's tool, or the session tools
// touched for it.
#[test]
fn a_reply_without_dtmf_leaves_the_session_tools_alone() {
    let mut driver = attached(CLIENT_SIDE, true);
    end_of_turn(&mut driver, 0, info(1, "please state your name", true));
    let screening =
        ReplyDecision { instructions: Some("Say who you are.".into()), ..ReplyDecision::allow() };
    let effects = answer(&mut driver, 100, decide(1, screening));
    let [(_, speech)] = replies(&effects)[..] else { panic!("one reply") };
    let request = responses(&effects)[0].req_id.clone();
    let mut all = effects;
    all.extend(finish_response(&mut driver, 600, &request, speech));
    assert!(tool_updates(&all).is_empty(), "no tool update: {all:?}");
}

// --- InsertUserItem (spec §7.2; `detector.py` `_add_realtime_transcript`) ---

// Catches: a model without user transcription and without session STT keeping no user item for
// the callee's words, including an older turn's late final.
#[test]
fn insert_user_item_adds_the_turn_for_a_model_without_transcription_or_stt() {
    let untranscribed = Model { user_transcription: false, ..CLIENT_SIDE };
    let mut driver = attached(untranscribed, false);
    end_of_turn(&mut driver, 0, info(1, "", false));
    let insert = AmdAnswer::InsertUserItem { run: RUN, turn: TurnId(1), text: "hello?".into() };
    let effects = answer(&mut driver, 50, insert);
    assert_eq!(user_items(&driver), ["hello?"]);
    assert_eq!(items_added(&effects, "hello?"), 1, "conversation_item_added fires once");
    answer(&mut driver, 60, decide(1, ReplyDecision::allow()));
    assert_eq!(user_items(&driver), ["hello?"], "the reply adds no second copy");

    // An older turn's late final, after a newer turn committed.
    end_of_turn(&mut driver, 1000, info(2, "", false));
    end_of_turn(&mut driver, 1500, info(3, "", false));
    let late = AmdAnswer::InsertUserItem { run: RUN, turn: TurnId(2), text: "is anyone".into() };
    answer(&mut driver, 1600, late);
    assert!(user_items(&driver).contains(&"is anyone".to_owned()), "{:?}", user_items(&driver));
}

// Catches: the callee's words entering the record twice, from the server's transcript or the
// session STT and again from AMD (proposal "Realtime": "never added twice"). With session STT,
// Python's session inserts the STT words for a model without user transcription
// (`agent_activity.py` `_user_turn_completed_impl`); with user transcription the server's
// transcript is the item.
#[test]
fn the_user_item_comes_from_exactly_one_source() {
    // A model with user transcription: AMD's insert is dropped, and the session keeps no STT copy.
    for allow in [true, false] {
        let mut driver = attached(CLIENT_SIDE, true);
        end_of_turn(&mut driver, 0, info(1, "hi there", true));
        answer(&mut driver, 50, decide(1, ReplyDecision { allow, ..ReplyDecision::allow() }));
        let insert =
            AmdAnswer::InsertUserItem { run: RUN, turn: TurnId(1), text: "hi there".into() };
        answer(&mut driver, 60, insert);
        assert!(
            user_items(&driver).is_empty(),
            "the server's transcript is the item (allow={allow})"
        );
    }
    // A model without user transcription, with session STT: the session's STT words, once.
    let untranscribed = Model { user_transcription: false, ..CLIENT_SIDE };
    for allow in [true, false] {
        let mut driver = attached(untranscribed, true);
        end_of_turn(&mut driver, 0, info(1, "hi there", true));
        answer(&mut driver, 50, decide(1, ReplyDecision { allow, ..ReplyDecision::allow() }));
        let insert =
            AmdAnswer::InsertUserItem { run: RUN, turn: TurnId(1), text: "hi there".into() };
        answer(&mut driver, 60, insert);
        assert_eq!(user_items(&driver), ["hi there"], "allow={allow}");
    }
}

// --- Session rules while attached (spec §5.3) ---

// Catches: a realtime fallback swap regenerating the interrupted reply past the gate (spec
// §5.3: "a realtime fallback swap does not regenerate the interrupted reply").
#[test]
fn a_fallback_swap_does_not_regenerate_while_attached() {
    use lk_agents_abi::voice::events::AgentState;
    let swap = |driver: &mut VoiceDriver| {
        driver.set_agent_state(at(10), AgentState::Speaking);
        driver.handle(None, now(10));
        let swapped = rt::Event::SessionSwapped { regenerate: true };
        driver.on_realtime(at(20), RealtimeWake::Event(swapped));
        let mut effects = driver.handle(None, now(20));
        effects.extend(run_until(driver, 30, 1000, 100));
        effects
    };
    // Without AMD the swap regenerates: the setup reaches the regeneration.
    let mut plain = rt_driver(CLIENT_SIDE, true);
    assert_eq!(responses(&swap(&mut plain)).len(), 1, "the setup regenerates without AMD");

    let mut driver = attached(CLIENT_SIDE, true);
    let mut effects = swap(&mut driver);
    assert!(replies(&effects).is_empty());
    // Not merely held: nothing plays once the run detaches and authorization is free.
    let detach =
        AmdAnswer::Detach { run: RUN, pending: ReplyDecision::deny(), cancel: DetachCancel::Keep };
    effects.extend(answer(&mut driver, 1100, detach));
    effects.extend(run_until(&mut driver, 1200, 3000, 100));
    assert!(driver.activity.gate.is_none());
    assert!(responses(&effects).is_empty(), "no regeneration while attached: {effects:?}");
}

// Catches: a backchannel over agent speech reaching AMD as a turn on a realtime model without
// session STT (spec §5.3: "That turn does not reach AMD"), or its audio committed.
#[test]
fn a_backchannel_over_agent_speech_is_dropped_without_session_stt() {
    let no_server_vad = Model { server_vad: false, vad_disabled: false, ..CLIENT_SIDE };
    let mut driver = attached(no_server_vad, false);
    driver.session.config.effective.turn.interruption.detection_enabled = Some(true);
    let backchannel = EndOfTurnInfo { backchannel_over_agent: true, ..info(1, "", false) };
    let effects = end_of_turn(&mut driver, 10, backchannel);
    assert!(committed(&effects).is_empty(), "the backchannel never reaches AMD: {effects:?}");
    assert_eq!(buffer_commits(&effects), 0, "its audio is not committed");
    assert!(responses(&effects).is_empty());

    // The same turn that is not a backchannel reaches AMD.
    let effects = end_of_turn(&mut driver, 100, info(2, "", false));
    assert_eq!(committed(&effects), [TurnId(2)]);
}
