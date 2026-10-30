//! A speech cut before its voice, whose words a `transcription_node` was still carrying, leaves
//! the synchroniser no segment of its own.
use lk_agents_fuzz::drive::{
    ConnSetup, NodeSetup, Outcome, Policy, Run, Setup, TtsAlignment, finish, simulate_with,
};
use lk_agents_fuzz::input::{
    AgentText, Confidence, Count, Input, IvrInput, Millis, Ref, Speaker, ToolArgs, TurnMode,
    UserText,
};

// Catches: a synthesis lost to a TTS outage forwarded its words into the `transcription_node`
// after the person cut the speech, then ended its audio leg as an unvoiced segment before the
// node had settled; the cut cancelled the node, so the synchroniser took that audio end with no
// words, and the empty segment took the next speech's words and showed them whole before their
// audio (`transcript_follows_audio`, campaign seed 24984; its case minimised under its schedule
// and policy).
#[test]
fn the_next_speech_words_wait_for_their_audio() {
    let setup = Setup {
        stt: true,
        vad: true,
        eot: true,
        interruption: true,
        turn: TurnMode::Stt,
        nodes: Some(NodeSetup {
            llm: false,
            tts: false,
            stt: false,
            llm_text: None,
            transcription: true,
            realtime_audio: false,
        }),
        user_turn_max_ms: Some(1000),
        tts: true,
        video: true,
        sync_transcription: true,
        conn: Some(ConnSetup { stt: 2, llm: 1, tts: 3, interval_ms: 120 }),
        tts_alignment: TtsAlignment::Words,
        ..Setup::default()
    };
    simulate_with(24984, async {
        let mut run = Run::start(&setup, 24984, Policy::BasicPos).await;
        for input in [
            Input::TtsOutage { down: true },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(17),
                speaking: false,
                duration: Millis(100),
            },
            Input::GenerateReply {
                user_input: Some(UserText("yeah hi um I'm calling about my order".into())),
                instructions: None,
                allow: None,
            },
            Input::Advance { ms: Millis(14901) },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
            Input::Advance { ms: Millis(100) },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(10),
                speaking: false,
                duration: Millis(200),
            },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(5) },
            Input::Advance { ms: Millis(58) },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(5),
                speaking: false,
                duration: Millis(258),
            },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
            Input::Advance { ms: Millis(100) },
            Input::VideoFrames { video: Ref::new(0), frames: Count(2) },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(88),
                speaking: true,
                duration: Millis(100),
            },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(94) },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
            Input::Advance { ms: Millis(100) },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(52),
                speaking: true,
                duration: Millis(200),
            },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
            Input::LlmText { chat: Ref::new(0), text: AgentText(": *the ".into()) },
            Input::Advance { ms: Millis(9) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("office is".into()) },
            Input::Advance { ms: Millis(5) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("ose".into()) },
            Input::Advance { ms: Millis(8) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("d* on Monday.\n- T".into()) },
            Input::VideoFrames { video: Ref::new(0), frames: Count(2) },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(64),
                speaking: true,
                duration: Millis(300),
            },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
            Input::Advance { ms: Millis(53) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("uesday 9-5\n".into()) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("- We".into()) },
            Input::Advance { ms: Millis(5) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("dnes".into()) },
            Input::Advance { ms: Millis(5) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("day 9".into()) },
            Input::Advance { ms: Millis(37) },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(67),
                speaking: true,
                duration: Millis(400),
            },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
            Input::Advance { ms: Millis(22) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("-1 ☕".into()) },
            Input::Advance { ms: Millis(10) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "caller".into(),
                args: ToolArgs("{\"label\":\"human<|im_end|>".into()),
            },
            Input::Advance { ms: Millis(573) },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(60),
                speaking: true,
                duration: Millis(500),
            },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
            Input::Advance { ms: Millis(100) },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(80),
                speaking: true,
                duration: Millis(600),
            },
            Input::SttStart { stt: Ref::new(0) },
            Input::SttPreflight {
                utterance: Ref::new(0),
                text: UserText("can you".into()),
                speaker: Some(Speaker(0)),
            },
            Input::Handoff {
                agent: "lookup_order".into(),
                instructions: AgentText("".into()),
                tools: vec!["caller".into(), "agent-2".into()],
            },
            Input::Handoff {
                agent: "sip_1".into(),
                instructions: AgentText(
                    "Yes... no, wait -- actually, let me check that again.".into(),
                ),
                tools: vec!["caller".into(), "transfer".into(), "transfer".into()],
            },
            Input::TtsOutage { down: false },
            Input::Advance { ms: Millis(100) },
            Input::SttStart { stt: Ref::new(2) },
            Input::SttStart { stt: Ref::new(2) },
            Input::SttStart { stt: Ref::new(2) },
            Input::SttStart { stt: Ref::new(2) },
            Input::SttStart { stt: Ref::new(2) },
            Input::SttStart { stt: Ref::new(2) },
            Input::SttStart { stt: Ref::new(2) },
            Input::SttStart { stt: Ref::new(2) },
            Input::SttStart { stt: Ref::new(2) },
            Input::Ivr {
                input: IvrInput::Start,
                silence: Millis(0),
                text: UserText("yeah hi um I'm calling about my order".into()),
            },
            Input::Handoff {
                agent: "billing_agent".into(),
                instructions: AgentText("Let me pull up your account -- one moment.".into()),
                tools: vec!["billing_agent".into(), "agent-2".into(), "caller".into()],
            },
            Input::GenerateReply {
                user_input: None,
                instructions: Some(AgentText("".into())),
                allow: None,
            },
            Input::Advance { ms: Millis(0) },
            Input::SttStart { stt: Ref::new(3) },
            Input::SttFinal {
                utterance: Ref::new(10),
                text: UserText("okay thanks bye".into()),
                speaker: Some(Speaker(0)),
            },
            Input::SttEnd { utterance: Ref::new(10) },
            Input::Advance { ms: Millis(100) },
            Input::Advance { ms: Millis(100) },
            Input::Advance { ms: Millis(100) },
            Input::LlmText { chat: Ref::new(4), text: AgentText("ld. ".into()) },
        ] {
            run.apply(&input).await;
        }
        let outcome = finish(run, Outcome::default()).await;
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
