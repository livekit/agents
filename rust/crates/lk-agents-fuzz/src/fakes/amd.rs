//! The scripted AMD answerer's side of the reply gate's protocol. Until the AMD machine
//! exists, these are the answers a run's AMD sends: the
//! spec §10.1 rows a verdict stands for, and `Detach.pending` built as the machine will build it.

use std::time::Duration;

use lk_agents_voice::amd::{DetachCancel, Opening, ReplyDecision};

use crate::input::{AmdOpening, AmdVerdict};

/// `machine_silence_threshold`'s default (spec §4.1).
pub const SILENCE: Duration = Duration::from_millis(1500);

/// The stage instructions a staged reply carries: distinct words a rule can find in a request
/// and must never find in the record.
pub const STAGE: &str = "AMD stage: you reached a machine. Leave the callback number.";

/// The name the host gives AMD's built-in DTMF tool.
pub const SEND_DTMF_TOOL: &str = "send_dtmf_events";

/// AMD's built-in DTMF tool, as `livekit.agents.beta`'s `send_dtmf_events(events: list[DtmfEvent])`
/// declares it. The host passes it at attach and runs its body.
pub fn dtmf_tool() -> lk_agents_abi::ToolSpec {
    let keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "*", "#", "A", "B", "C", "D"];
    lk_agents_abi::ToolSpec {
        name: SEND_DTMF_TOOL.into(),
        description: "Send a list of DTMF events to the telephony provider.\n\nCall when:\n- \
                      User wants to send DTMF events"
            .into(),
        json_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "events": { "type": "array", "items": { "type": "string", "enum": keys } }
            },
            "required": ["events"],
        }),
        flags: Default::default(),
        provider_tool: None,
    }
}

/// What the opening says, or is generated from.
pub const OPENING_TEXT: &str = "Hi, this is the clinic calling about your appointment.";
pub const OPENING_INSTRUCTIONS: &str = "AMD opening: greet the callee and say who is calling.";

/// The decision a verdict sends for its turn.
pub fn decision(verdict: AmdVerdict) -> ReplyDecision {
    let staged = || ReplyDecision { instructions: Some(STAGE.into()), ..ReplyDecision::allow() };
    match verdict {
        AmdVerdict::Allow => ReplyDecision::allow(),
        AmdVerdict::Deny | AmdVerdict::Wait => ReplyDecision::deny(),
        AmdVerdict::Stage => staged(),
        AmdVerdict::Silence => ReplyDecision { silence: Some(SILENCE), ..staged() },
        AmdVerdict::Ivr => ReplyDecision { silence: Some(SILENCE), dtmf: true, ..staged() },
        AmdVerdict::Held => ReplyDecision { silence: Some(SILENCE), ..ReplyDecision::allow() },
        AmdVerdict::HeldIvr => {
            ReplyDecision { silence: Some(SILENCE), dtmf: true, ..ReplyDecision::allow() }
        }
    }
}

/// `Detach.pending`, as the machine builds it: a `wait` category or
/// replies forbidden deny; anything else allows with no instructions, silence or DTMF.
pub fn pending(category: AmdVerdict, forbid: bool) -> ReplyDecision {
    if forbid || category == AmdVerdict::Wait {
        ReplyDecision::deny()
    } else {
        ReplyDecision::allow()
    }
}

pub fn cancel(forbid: bool) -> DetachCancel {
    if forbid { DetachCancel::ForbidReplies } else { DetachCancel::Keep }
}

pub fn opening(kind: AmdOpening) -> Opening {
    match kind {
        AmdOpening::Text => Opening::Text(OPENING_TEXT.into()),
        AmdOpening::Instructions => Opening::Instructions(OPENING_INSTRUCTIONS.into()),
    }
}

/// Production probes for the AMD rules: a real session with the scripted answerer's inputs, as
/// the authored stories in `tests/regressions/amd` drive it.
pub mod probe {
    use crate::drive::{Case, Setup};
    use crate::input::{AgentText, AmdOpening, AmdVerdict, Input, Millis, Ref, UserText};

    /// A detector and a recogniser on the call; the turn commits 500 ms after the detector's end
    /// of speech.
    pub fn call(inputs: Vec<Input>) -> Case {
        Case {
            setup: Setup { stt: true, vad: true, ..Setup::default() },
            inputs,
            ..Case::default()
        }
    }

    /// AMD attaches and starts listening, with this opening.
    pub fn attached(opening: Option<AmdOpening>) -> Vec<Input> {
        vec![Input::AmdAttach {}, Input::AmdActivate { opening }]
    }

    /// The person says `words` as the case's utterance `utterance`, for 300 ms, and the turn
    /// commits.
    pub fn says(utterance: u32, words: &str) -> Vec<Input> {
        let u = Ref::new(utterance);
        vec![
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(300) },
            Input::SttFinal { utterance: u, text: UserText(words.into()), speaker: None },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(0) },
            Input::SttEnd { utterance: u },
            Input::Advance { ms: Millis(600) },
        ]
    }

    /// AMD's decision for the case's `turn`th gated turn.
    pub fn decide(turn: u32, verdict: AmdVerdict) -> Input {
        Input::AmdDecide { turn: Ref::new(turn), verdict }
    }

    /// The model answers each of the first `chats` requests with a line; a request already
    /// closed refuses it.
    pub fn answered(chats: u32) -> Vec<Input> {
        let mut inputs = vec![Input::Advance { ms: Millis(50) }];
        for chat in 0..chats {
            let text = AgentText("Hello, this is the clinic.".into());
            inputs.push(Input::LlmText { chat: Ref::new(chat), text });
            inputs.push(Input::LlmEnd { chat: Ref::new(chat) });
        }
        inputs
    }

    pub fn advance(ms: u64) -> Input {
        Input::Advance { ms: Millis(ms) }
    }

    /// One turn decided with `verdict`, answered, and the call carried past any hold.
    pub fn one_turn(words: &str, verdict: AmdVerdict) -> Case {
        let mut inputs = attached(None);
        inputs.extend(says(0, words));
        inputs.push(decide(0, verdict));
        inputs.extend(answered(3));
        inputs.push(advance(4_000));
        call(inputs)
    }
}

/// What a case enters AMD with through the runtime's executor (`Input::AmdEnter`).
#[derive(Clone, Debug, PartialEq)]
pub struct AmdSetup {
    /// A decision model classifies, instead of an LLM.
    pub decision: bool,
    /// AMD has its own streaming recogniser: the stream it opens is `Detail::AmdStt`.
    pub own_stt: bool,
    /// Its models reach only LiveKit Inference.
    pub inference: bool,
    pub options: lk_agents_voice::amd::options::AmdOptions,
    /// The participant to listen to; `None` listens to the linked one.
    pub participant: Option<String>,
    /// The caller joins the room as a SIP participant whose `sip.callStatus` starts at this;
    /// `None` joins them as a standard participant.
    pub sip_status: Option<String>,
}

/// Instructions for each machine stage and for a person after one, each carrying [`STAGE`] so the
/// gate's rules find them as they find the scripted answerer's.
pub const SCREENING: &str =
    "AMD stage: you reached a machine. Leave the callback number. Answer the screener.";
pub const VOICEMAIL: &str =
    "AMD stage: you reached a machine. Leave the callback number. Leave one message.";
pub const IVR: &str =
    "AMD stage: you reached a machine. Leave the callback number. Choose from the menu.";
pub const HUMAN_AFTER_MACHINE: &str =
    "AMD stage: you reached a machine. Leave the callback number. A person took over.";

/// AMD's options in the DST: Python's defaults, with the stage instructions above.
pub fn options() -> lk_agents_voice::amd::options::AmdOptions {
    lk_agents_voice::amd::options::AmdOptions {
        screening_instructions: Some(SCREENING.into()),
        voicemail_instructions: Some(VOICEMAIL.into()),
        ivr_instructions: Some(IVR.into()),
        human_instructions: Some(HUMAN_AFTER_MACHINE.into()),
        ..Default::default()
    }
}

impl Default for AmdSetup {
    fn default() -> Self {
        AmdSetup {
            decision: false,
            own_stt: false,
            inference: false,
            options: options(),
            participant: None,
            sip_status: None,
        }
    }
}

/// The executor's configuration over this run's fakes: one attempt per model request, so each
/// request is one `AmdAsk` the case answers.
pub fn config(
    setup: &AmdSetup,
    registry: &std::sync::Arc<super::Registry>,
    stt_speech: super::SttSpeech,
) -> lk_agents_runtime::voice::amd::AmdConfig {
    use lk_agents_runtime::voice::amd::{AmdConfig, AmdModel, Classifier, DecisionModel};
    use std::sync::Arc;
    let classifier = if setup.decision {
        let model = Arc::new(FakeDecider { registry: registry.clone() }) as Arc<dyn DecisionModel>;
        Classifier::Decision(AmdModel::supplied(model, setup.inference))
    } else {
        let model =
            Arc::new(FakeClassifier { registry: registry.clone() }) as Arc<dyn lk_agents_abi::Llm>;
        Classifier::Llm(AmdModel::supplied(model, setup.inference))
    };
    let stt = setup.own_stt.then(|| {
        let model =
            super::FakeStt::for_amd(registry.clone(), stt_speech) as Arc<dyn lk_agents_abi::Stt>;
        AmdModel::supplied(model, setup.inference)
    });
    let once = lk_agents_abi::ApiConnectOptions {
        max_retry: 0,
        retry_interval: 0.0,
        ..Default::default()
    };
    AmdConfig {
        options: setup.options.clone(),
        classifier,
        stt,
        participant: setup.participant.clone(),
        dtmf_tool: dtmf_tool(),
        llm_conn_options: once,
        stt_conn_options: once,
    }
}

/// How one ask is answered: the open stream of a classifier request, or a choice's reply.
pub enum Answering {
    Llm(
        tokio::sync::mpsc::UnboundedSender<
            Result<lk_agents_abi::LlmChunk, lk_agents_abi::errors::ProviderError>,
        >,
    ),
    Choice(
        tokio::sync::oneshot::Sender<Result<Option<String>, lk_agents_abi::errors::ProviderError>>,
    ),
}

/// The words of the menu a `Menu` answer extracts.
pub const MENU: &str = "Choose a department";

fn provider_failure() -> lk_agents_abi::errors::ProviderError {
    let error = lk_agents_abi::errors::ApiError::new(
        lk_agents_abi::errors::ErrorKind::Server,
        "injected AMD model failure",
    );
    lk_agents_abi::errors::ProviderError::Local(error)
}

fn crash() -> lk_agents_abi::errors::ProviderError {
    lk_agents_abi::errors::HostError::new("RuntimeError: injected AMD model bug").into()
}

/// Hands `says` to the ask; true when its model call still took it.
pub(super) fn deliver(answering: Answering, says: crate::input::AmdSays) -> bool {
    use crate::input::AmdSays;
    use crate::record::amd_run::says_category;
    match answering {
        Answering::Llm(sender) => {
            let arguments = match says {
                AmdSays::Menu => serde_json::json!({
                    "menu": MENU,
                    "options": [{"label": "Sales", "dtmf": "1"}, {"label": "Support", "dtmf": "2"}],
                })
                .to_string(),
                AmdSays::NoMenu => serde_json::json!({"menu": "", "options": []}).to_string(),
                AmdSays::Garbled => "not JSON".into(),
                AmdSays::Failed => return sender.send(Err(provider_failure())).is_ok(),
                AmdSays::Crashed => return sender.send(Err(crash())).is_ok(),
                category => {
                    let value = says_category(category).map_or("", |c| c.as_str());
                    serde_json::json!({ "category": value }).to_string()
                }
            };
            let call = lk_agents_abi::FunctionToolCall {
                name: "record_result".into(),
                arguments,
                call_id: "call_amd".into(),
                extra: None,
            };
            let delta = lk_agents_abi::ChoiceDelta {
                role: Some(lk_agents_abi::Role::Assistant),
                tool_calls: vec![call],
                ..Default::default()
            };
            let chunk = lk_agents_abi::LlmChunk {
                id: "resp_amd".into(),
                delta: Some(delta),
                ..Default::default()
            };
            // The stream ends after the call: dropping the sender is its end.
            sender.send(Ok(chunk)).is_ok()
        }
        Answering::Choice(reply) => {
            let answer = match says {
                AmdSays::Failed => Err(provider_failure()),
                AmdSays::Crashed => Err(crash()),
                category => Ok(says_category(category).map(|c| c.as_str().to_owned())),
            };
            reply.send(answer).is_ok()
        }
    }
}

fn messages(request: &lk_agents_abi::ChatRequest) -> Vec<crate::record::amd::RequestMessage> {
    (request.chat_ctx.items.iter())
        .filter_map(|item| item.as_message())
        .map(|m| crate::record::amd::RequestMessage {
            id: m.id.clone(),
            role: m.role,
            text: m.raw_text_content().unwrap_or_default(),
        })
        .collect()
}

fn tagged(extra: Option<&lk_agents_abi::llm::chat_context::Extra>) -> bool {
    extra.and_then(|extra| extra.get("extra_headers")).is_some_and(|headers| {
        headers.get("X-LiveKit-Client-Feature").and_then(|v| v.as_str()) == Some("AMD")
    })
}

/// AMD's classifier LLM: every request is an `AmdAsk` the case answers with `AmdReply`.
pub struct FakeClassifier {
    registry: std::sync::Arc<super::Registry>,
}

/// An ask's open stream. Closing or dropping it ends the ask.
struct Asking {
    slot: super::Slot,
    registry: std::sync::Arc<super::Registry>,
    receiver: tokio::sync::mpsc::UnboundedReceiver<
        Result<lk_agents_abi::LlmChunk, lk_agents_abi::errors::ProviderError>,
    >,
    closed: bool,
}

impl Asking {
    fn end(&mut self) {
        if !std::mem::replace(&mut self.closed, true) {
            self.registry.mark_closed(self.slot);
        }
    }
}

impl Drop for Asking {
    fn drop(&mut self) {
        self.end();
    }
}

impl lk_agents_abi::utils::stream::StreamOutput<lk_agents_abi::LlmChunk> for Asking {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::LlmChunk>, lk_agents_abi::errors::ProviderError>,
    > {
        Box::pin(async move {
            if self.closed {
                return Ok(None);
            }
            match self.receiver.recv().await {
                Some(Ok(chunk)) => {
                    // One answer per ask: the stream ends after it.
                    self.receiver.close();
                    Ok(Some(chunk))
                }
                Some(Err(error)) => Err(error),
                None => Ok(None),
            }
        })
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(
        mut self: Box<Self>,
        _: lk_agents_abi::Ts,
    ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::utils::stream::Closed> {
        self.end();
        Box::pin(async { lk_agents_abi::utils::stream::Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<lk_agents_abi::LlmChunk> for Asking {}

impl lk_agents_abi::Llm for FakeClassifier {
    fn label(&self) -> String {
        "fake::amd_classifier".into()
    }

    fn chat<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        request: &'a lk_agents_abi::ChatRequest,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<lk_agents_abi::LlmStream, lk_agents_abi::errors::ProviderError>,
    > {
        Box::pin(async move {
            let messages = messages(request);
            let menu =
                messages.first().is_some_and(|m| m.text.starts_with("Extract observed IVR menu"));
            let asked = crate::record::amd_run::Asked {
                kind: if menu {
                    crate::record::amd_run::AskKind::Menu
                } else {
                    crate::record::amd_run::AskKind::Classify
                },
                messages,
                tools: request.tools.iter().map(|t| t.name.clone()).collect(),
                required: request.tool_choice == Some(lk_agents_abi::ToolChoice::Required),
                tagged: tagged(request.extra.as_ref()),
                input: None,
                options: Vec::new(),
            };
            let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
            let slot = self.registry.register(
                crate::record::Kind::AmdAsk,
                crate::record::Detail::AmdAsk(Box::new(asked)),
                super::Control::AmdAsk(Answering::Llm(sender)),
            );
            let registry = self.registry.clone();
            Ok(Box::new(Asking { slot, registry, receiver, closed: false })
                as lk_agents_abi::LlmStream)
        })
    }
}

/// AMD's decision model: every choice is an `AmdAsk` the case answers with `AmdReply`.
pub struct FakeDecider {
    registry: std::sync::Arc<super::Registry>,
}

/// Ends the ask when the choice returns or its caller gives up.
struct ChoiceAsk {
    slot: super::Slot,
    registry: std::sync::Arc<super::Registry>,
}

impl Drop for ChoiceAsk {
    fn drop(&mut self) {
        self.registry.mark_closed(self.slot);
    }
}

impl lk_agents_runtime::voice::amd::DecisionModel for FakeDecider {
    fn label(&self) -> String {
        "fake::amd_decision_model".into()
    }

    fn choose<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        request: &'a lk_agents_runtime::voice::amd::ChoiceRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Option<String>, lk_agents_abi::errors::ProviderError>>
    {
        let asked = crate::record::amd_run::Asked {
            kind: crate::record::amd_run::AskKind::Choice,
            messages: Vec::new(),
            tools: Vec::new(),
            required: false,
            tagged: request.extra_headers.get("X-LiveKit-Client-Feature").map(String::as_str)
                == Some("AMD"),
            input: Some(request.input.clone()),
            options: request.options.clone(),
        };
        let (reply, answer) = tokio::sync::oneshot::channel();
        let slot = self.registry.register(
            crate::record::Kind::AmdAsk,
            crate::record::Detail::AmdAsk(Box::new(asked)),
            super::Control::AmdAsk(Answering::Choice(reply)),
        );
        let ask = ChoiceAsk { slot, registry: self.registry.clone() };
        Box::pin(async move {
            let _ask = ask;
            answer.await.unwrap_or_else(|_| Err(provider_failure()))
        })
    }
}

/// Production probes for the AMD executor's rules: a real session and the real executor over the
/// fake models, as `tests/regressions/amd/executor.rs` drives them.
pub mod executor_probe {
    use super::AmdSetup;
    use crate::drive::{Case, RoomLegs, Setup};
    use crate::input::{AmdSays, Input, Millis, Ref};

    /// The person in the room with their microphone, a detector and a recogniser on the call,
    /// and AMD entered through its executor.
    pub fn call(amd: AmdSetup, inputs: Vec<Input>) -> Case {
        let mut all = vec![Input::AmdEnter {}, Input::Advance { ms: Millis(100) }];
        all.extend(inputs);
        Case {
            setup: Setup {
                stt: true,
                vad: true,
                room_legs: Some(RoomLegs { pre_connect: None }),
                amd: Some(amd),
                ..Setup::default()
            },
            inputs: all,
            ..Case::default()
        }
    }

    /// The person says `words` as utterance `utterance`; then the classifier answers ask `ask`
    /// with `says`, the agent's model answers, and the call goes on past any hold.
    pub fn turn(utterance: u32, words: &str, ask: u32, says: AmdSays, chat: u32) -> Vec<Input> {
        let mut inputs = super::probe::says(utterance, words);
        inputs.push(Input::AmdReply { ask: Ref::new(ask), says });
        inputs.push(Input::Advance { ms: Millis(2_000) });
        inputs.push(Input::LlmText {
            chat: Ref::new(chat),
            text: crate::input::AgentText("Hello, this is the clinic.".into()),
        });
        inputs.push(Input::LlmEnd { chat: Ref::new(chat) });
        inputs.push(Input::Advance { ms: Millis(3_000) });
        inputs
    }
}
