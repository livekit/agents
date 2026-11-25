//! The AMD reply gate, driven by the scripted AMD answerer. Each story names the
//! `tests/test_amd.py` scenario on `main` it ports, if any, and asserts what the caller would notice: which words the model
//! was asked to answer, with which instructions and tools, what was heard and when, and what the
//! conversation saved. Every story also passes the run's contracts, the AMD rules included.

mod detach;
mod executor;
mod gate;
mod opening;
mod payload;
mod silence;

use lk_agents_abi::Role;
use lk_agents_fuzz::check::Failure;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{
    AgentText, AmdOpening, AmdTurn, AmdVerdict, Input, Millis, Ref, UserText,
};
use lk_agents_fuzz::record::Kind;
use lk_agents_fuzz::record::amd::RequestMessage;
use lk_agents_voice::amd::AmdObservation;
use lk_agents_voice::events::SessionEvent;

/// A call with AMD on it: the session, and the next utterance the recogniser will open.
pub struct Call {
    pub run: Run,
    utterances: u32,
}

/// One model request, as the fake model received it.
#[derive(Clone, Debug)]
pub struct Asked {
    pub chat: u32,
    pub speech: Option<u64>,
    pub at_ms: u64,
    pub messages: Vec<RequestMessage>,
    pub tools: Vec<String>,
}

impl Asked {
    pub fn carries(&self, text: &str) -> bool {
        self.messages.iter().any(|m| m.text.contains(text))
    }

    pub fn offers(&self, tool: &str) -> usize {
        self.tools.iter().filter(|t| *t == tool).count()
    }
}

/// A microphone, a detector and a recogniser on the call; the default endpointing commits 500 ms
/// after the detector's end of speech.
pub fn listening() -> Setup {
    Setup { stt: true, vad: true, ..Setup::default() }
}

impl Call {
    pub fn run(setup: Setup, story: impl AsyncFnOnce(&mut Call)) -> Call {
        simulate(async move {
            let run = Run::start(&setup, 0, Default::default()).await;
            let mut call = Call { run, utterances: 0 };
            story(&mut call).await;
            call
        })
    }

    pub async fn send(&mut self, input: Input) {
        crate::playout::apply_with_playout(&mut self.run, &input).await;
    }

    pub async fn advance(&mut self, ms: u64) {
        self.send(Input::Advance { ms: Millis(ms) }).await;
    }

    pub fn now_ms(&self) -> u64 {
        self.run.record.now().at_ms
    }

    /// AMD attaches and starts listening, with this opening.
    pub async fn attach(&mut self, opening: Option<AmdOpening>) {
        self.send(Input::AmdAttach {}).await;
        self.send(Input::AmdActivate { opening }).await;
    }

    /// The person speaks `words` for 300 ms and stops; the turn commits 500 ms after the
    /// detector's end of speech. Returns when the speech ended, by the detector's ground truth.
    pub async fn says(&mut self, words: &str) -> u64 {
        let utterance = Ref::new(self.utterances);
        self.utterances += 1;
        self.send(Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) }).await;
        self.send(Input::SttStart { stt: Ref::new(0) }).await;
        self.advance(300).await;
        self.send(Input::SttFinal { utterance, text: UserText(words.into()), speaker: None }).await;
        self.send(Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(0) }).await;
        let ended = self.now_ms();
        self.send(Input::SttEnd { utterance }).await;
        self.advance(600).await;
        ended
    }

    /// The latest turn AMD observed.
    pub fn latest(&self) -> Ref<AmdTurn> {
        Ref::new(self.run.record.live().opened(Kind::AmdTurn).saturating_sub(1))
    }

    pub async fn decide(&mut self, verdict: AmdVerdict) {
        let turn = self.latest();
        self.send(Input::AmdDecide { turn, verdict }).await;
    }

    pub async fn detach(&mut self, category: AmdVerdict, forbid: bool) {
        self.send(Input::AmdDetach { category, forbid }).await;
    }

    /// The model answers every request still open with `line`.
    pub async fn model_answers(&mut self, line: &str) {
        for chat in self.run.record.live().of(Kind::Chat) {
            self.send(Input::LlmText { chat: Ref::new(chat), text: AgentText(line.into()) }).await;
            self.send(Input::LlmEnd { chat: Ref::new(chat) }).await;
        }
    }

    /// The host answers the latest turn hook.
    pub async fn hook_returns(&mut self) {
        let hook = self.run.record.resources(Kind::Hook).count().saturating_sub(1) as u32;
        self.send(Input::HookDone {
            hook: Ref::new(hook),
            answer: lk_agents_fuzz::input::HookAnswer::Returned,
            text: AgentText(String::new()),
        })
        .await;
    }

    /// Every model request whose last user message is `words`.
    pub fn asked(&self, words: &str) -> Vec<Asked> {
        self.run
            .record
            .amd_view()
            .requests()
            .into_iter()
            .filter(|q| q.last_user().is_some_and(|m| m.text == words))
            .map(|q| Asked {
                chat: q.chat,
                speech: q.speech,
                at_ms: q.opened.at_ms,
                messages: q.messages.to_vec(),
                tools: q.tools.to_vec(),
            })
            .collect()
    }

    /// The requests made for `words` once their turn was decided: every one but a preemptive
    /// generation's guess, which the session may start while authorization is held.
    pub fn made(&self, words: &str) -> Vec<Asked> {
        let view = self.run.record.amd_view();
        let preemptive: Vec<u32> =
            view.requests().iter().filter(|q| view.preemptive(q)).map(|q| q.chat).collect();
        self.asked(words).into_iter().filter(|q| !preemptive.contains(&q.chat)).collect()
    }

    /// The requests for `words` whose speech was heard, one per speech.
    pub fn heard_requests(&self, words: &str) -> Vec<Asked> {
        let mut seen = Vec::new();
        let mut out = Vec::new();
        for q in self.asked(words) {
            if self.heard(q.speech).is_some() && !seen.contains(&q.speech) {
                seen.push(q.speech);
                out.push(q);
            }
        }
        out
    }

    /// Nothing answered `words` yet: no request but a preemptive guess, nothing heard.
    pub fn unanswered(&self, words: &str) -> bool {
        self.made(words).is_empty() && self.heard_reply(words).is_none()
    }

    /// When the speech's first audio reached the speaker.
    pub fn heard(&self, speech: Option<u64>) -> Option<u64> {
        let speech = speech?;
        self.run.record.amd_view().first_audio(speech).map(|at| at.at_ms)
    }

    /// When any reply to `words` was first heard.
    pub fn heard_reply(&self, words: &str) -> Option<u64> {
        self.asked(words).iter().filter_map(|q| self.heard(q.speech)).min()
    }

    /// What the attached AMD runs observed.
    pub fn observed(&self) -> Vec<(u64, AmdObservation)> {
        self.run.record.amd().iter().map(|(at, _, o)| (at.at_ms, o.clone())).collect()
    }

    pub fn detached_at(&self) -> Option<u64> {
        self.observed()
            .into_iter()
            .find_map(|(at, o)| matches!(o, AmdObservation::Detached { .. }).then_some(at))
    }

    /// The user words the conversation recorded.
    pub fn saved_user_words(&self) -> Vec<String> {
        self.recorded(Role::User)
    }

    pub fn recorded(&self, role: Role) -> Vec<String> {
        let events = self.run.record.host_view().events();
        events
            .into_iter()
            .filter_map(|(_, event)| match event {
                SessionEvent::ConversationItemAdded { item, .. } => item.as_message(),
                _ => None,
            })
            .filter(|m| m.role == role)
            .filter_map(|m| m.raw_text_content())
            .collect()
    }

    /// Whether any recorded item holds `text`.
    pub fn recorded_anywhere(&self, text: &str) -> bool {
        [Role::System, Role::User, Role::Assistant]
            .into_iter()
            .flat_map(|role| self.recorded(role))
            .any(|said| said.contains(text))
    }

    /// The run's contract failures, the AMD rules included.
    pub fn failures(&self) -> Vec<Failure> {
        lk_agents_fuzz::check::check_all(&self.run.record).0
    }
}
