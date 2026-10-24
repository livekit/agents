use crate::agent_session::effects::TurnTimer;
use crate::agent_session::{IvrCommand, Origin, PeerHandle, SpeechCommand, SystemTap};
use crate::events::SessionEvent;
use crate::trace::{CoreStep, DetectorStep, IvrLifecycle, Step};
use lk_agents_abi::voice::events::{AgentState, UserState};
use lk_agents_abi::{SpawnerExt, Ts};
use std::collections::BTreeMap;
use std::time::Duration;

/// `IVRActivity(max_silence_duration=5.0)`: the reference's default silence budget.
pub const IVR_MAX_SILENCE: Duration = Duration::from_secs(5);

/// `TfidfVectorizer`'s default `token_pattern` by hand: two or more word characters, so
/// punctuation-only differences don't count.
fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            current.extend(c.to_lowercase());
        } else if !current.is_empty() {
            if current.chars().count() >= 2 {
                out.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    if current.chars().count() >= 2 {
        out.push(current);
    }
    out
}

fn term_counts(text: &str) -> BTreeMap<String, f64> {
    let mut counts: BTreeMap<String, f64> = BTreeMap::new();
    for token in tokenize(text) {
        *counts.entry(token).or_insert(0.0) += 1.0;
    }
    counts
}

/// The sliding-window loop detector. Defaults are the reference's: 20 documents, `0.85`, and three
/// consecutive similar ones before it fires.
#[derive(Debug, Clone)]
pub struct TfidfLoopDetector {
    window_size: usize,
    similarity_threshold: f64,
    consecutive_threshold: u32,
    chunks: Vec<String>,
    consecutive: u32,
}

/// A constructor argument out of range; returned since host values must not abort the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LoopDetectorError {
    #[error("window_size must be greater than 0")]
    WindowSize,
    #[error("similarity_threshold must be between 0.0 and 1.0")]
    SimilarityThreshold,
    #[error("consecutive_threshold must be greater than 0")]
    ConsecutiveThreshold,
}

impl Default for TfidfLoopDetector {
    fn default() -> Self {
        TfidfLoopDetector {
            window_size: 20,
            similarity_threshold: 0.85,
            consecutive_threshold: 3,
            chunks: Vec::new(),
            consecutive: 0,
        }
    }
}

/// TF-IDF loop detection over a sliding window of final transcripts.
#[lk_agents_abi::host(object(name = "IvrLoopDetector", cell = "mutex"))]
impl TfidfLoopDetector {
    #[host(new(defaults(
        window_size = 20,
        similarity_threshold = 0.85,
        consecutive_threshold = 3
    )))]
    pub fn new(
        window_size: usize,
        similarity_threshold: f64,
        consecutive_threshold: u32,
    ) -> Result<Self, LoopDetectorError> {
        if window_size == 0 {
            return Err(LoopDetectorError::WindowSize);
        }
        if !(0.0..=1.0).contains(&similarity_threshold) || similarity_threshold.is_nan() {
            return Err(LoopDetectorError::SimilarityThreshold);
        }
        if consecutive_threshold == 0 {
            return Err(LoopDetectorError::ConsecutiveThreshold);
        }
        Ok(TfidfLoopDetector {
            window_size,
            similarity_threshold,
            consecutive_threshold,
            chunks: Vec::new(),
            consecutive: 0,
        })
    }

    /// Reset the window and the counter, so the next loop builds up from nothing.
    #[host(method)]
    pub fn reset(&mut self) {
        self.chunks.clear();
        self.consecutive = 0;
    }

    #[host(method)]
    pub fn add_chunk(&mut self, chunk: &str) {
        self.chunks.push(chunk.to_owned());
        if self.chunks.len() > self.window_size {
            let drop = self.chunks.len() - self.window_size;
            self.chunks.drain(..drop);
        }
    }

    /// Is the newest document a repeat? Stateful: advances or resets the counter, once per chunk.
    /// Fewer than two documents is `false`, counter untouched.
    #[host(method)]
    pub fn check_loop_detection(&mut self) -> bool {
        if self.chunks.len() < 2 {
            return false;
        }

        let docs: Vec<BTreeMap<String, f64>> = self.chunks.iter().map(|c| term_counts(c)).collect();
        let n = docs.len() as f64;

        // `smooth_idf=True`: a term in every document keeps a non-zero idf.
        let mut df: BTreeMap<&str, f64> = BTreeMap::new();
        for doc in &docs {
            for term in doc.keys() {
                *df.entry(term.as_str()).or_insert(0.0) += 1.0;
            }
        }
        let idf = |term: &str| -> f64 {
            let d = df.get(term).copied().unwrap_or(0.0);
            ((1.0 + n) / (1.0 + d)).ln() + 1.0
        };

        // l2-normalised tf-idf, so cosine similarity is a plain dot product.
        let vectors: Vec<BTreeMap<&str, f64>> = docs
            .iter()
            .map(|doc| {
                let mut v: BTreeMap<&str, f64> =
                    doc.iter().map(|(t, c)| (t.as_str(), c * idf(t))).collect();
                let norm = v.values().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 0.0 {
                    for x in v.values_mut() {
                        *x /= norm;
                    }
                }
                v
            })
            .collect();

        let last = vectors.last().expect("len >= 2");
        let max_similarity = vectors[..vectors.len() - 1]
            .iter()
            .map(|other| {
                last.iter().map(|(t, x)| other.get(t).copied().unwrap_or(0.0) * x).sum::<f64>()
            })
            .fold(f64::NEG_INFINITY, f64::max);

        // Strictly greater, as `np.max(...) > threshold`.
        if max_similarity > self.similarity_threshold {
            self.consecutive += 1;
        } else {
            self.consecutive = 0;
        }

        self.consecutive >= self.consecutive_threshold
    }
}

/// Why the navigator asked for a reply. The driver stamps it on the reply it creates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum IvrReplyOrigin {
    /// Three consecutive repeats of a menu: the model is nudged, uninterruptibly.
    Loop,
    /// Both sides quiet for `max_silence`: a plain, interruptible nudge.
    Silence,
}

struct Navigator {
    detector: TfidfLoopDetector,
    max_silence: Duration,
}

pub struct IvrSystem {
    peer: PeerHandle,
    nav: Option<Navigator>,
    /// The (agent, user) labels last seen (`IVRActivity._current_agent_state/_current_user_state`);
    /// `initializing` counts as quiet.
    states: (AgentState, UserState),
    /// The armed silence countdown (`_debounced_silence`), on the session clock. Armed only while a
    /// navigator runs.
    deadline: Option<Ts>,
    closed: bool,
}

impl IvrSystem {
    /// The system, subscribed to the feed and navigating now if `ivr_detection`.
    pub fn spawn(peer: PeerHandle, ivr_detection: bool) {
        let rx = peer.subscribe();
        let mut system = IvrSystem {
            peer,
            nav: None,
            states: (AgentState::Initializing, UserState::Listening),
            deadline: None,
            closed: false,
        };
        if ivr_detection {
            system.start(system.peer.time.now(), IVR_MAX_SILENCE);
        }
        let spawner = system.peer.spawner.clone();
        spawner.spawn_unboxed(system.run(rx));
    }

    /// The loop: a due deadline fires before same-instant feed items; ends at close.
    async fn run(mut self, mut rx: tokio::sync::mpsc::UnboundedReceiver<SystemTap>) {
        let time = self.peer.time.clone();
        let api = self.peer.api.clone();
        while !self.closed {
            let deadline = self.deadline;
            if deadline.is_some_and(|at| at <= time.now()) {
                self.on_timer(time.now());
                continue;
            }
            tokio::select! {
                biased;
                () = time.wait_until_opt(deadline) => self.on_timer(time.now()),
                item = rx.recv() => match item {
                    Some(item) => self.on_watched(time.now(), item),
                    None => break,
                },
                () = api.closed() => break,
            }
        }
    }

    fn on_watched(&mut self, now: Ts, item: SystemTap) {
        match item {
            // `_on_user_input_transcribed`: finals only, while a navigator runs.
            SystemTap::Session(SessionEvent::UserTranscript(t)) if t.is_final => {
                let text = t.ev.best_alternative().map(|a| a.text.clone()).unwrap_or_default();
                self.on_final(&text);
            }
            SystemTap::Session(SessionEvent::AgentStateChanged { new, .. }) => {
                self.states.0 = new;
                self.quiet_edge(now);
            }
            SystemTap::Session(SessionEvent::UserStateChanged { new, .. }) => {
                self.states.1 = new;
                self.quiet_edge(now);
            }
            SystemTap::Session(_) | SystemTap::Vad(_) | SystemTap::Stt(_) => {}
            SystemTap::Ivr(command) => match command {
                IvrCommand::Start { max_silence } => self.start(now, max_silence),
                IvrCommand::SeedTranscript { text } => self.on_final(&text),
                IvrCommand::Close => self.close(),
                // The session's half of the start, acted on by the core and never relayed.
                IvrCommand::AddTools(_) => {}
            },
            // `_ivr_activity.aclose()`: no nudge inside the close window.
            SystemTap::CloseRequested => {
                self.close();
                self.closed = true;
            }
        }
    }

    /// Start the navigator; one already running stands.
    fn start(&mut self, now: Ts, max_silence: Duration) {
        if self.nav.is_some() {
            self.peer
                .journal(Step::Detector(DetectorStep::IvrLifecycle(IvrLifecycle::AlreadyStarted)));
            return;
        }
        if self.closed {
            return;
        }
        self.nav = Some(Navigator { detector: TfidfLoopDetector::default(), max_silence });
        self.peer.journal(Step::Detector(DetectorStep::IvrLifecycle(IvrLifecycle::Started)));
        // Both sides already quiet counts as the transition.
        self.quiet_edge(now);
    }

    /// `IVRActivity.aclose()`: the host's, the session's close, or the AMD taking the lifecycle over.
    fn close(&mut self) {
        if self.nav.take().is_none() {
            return;
        }
        if self.deadline.is_some() {
            self.cancel_silence();
        }
        self.peer.journal(Step::Detector(DetectorStep::IvrLifecycle(IvrLifecycle::Closed)));
    }

    /// Count one final (the only kind that counts toward a loop); a completed loop is nudged and
    /// the window reset, so a new run re-fires.
    fn on_final(&mut self, text: &str) {
        let Some(nav) = self.nav.as_mut() else { return };
        nav.detector.add_chunk(text);
        if nav.detector.check_loop_detection() {
            nav.detector.reset();
            self.reply(IvrReplyOrigin::Loop, false);
        }
    }

    /// `_should_schedule_check`: both quiet arms the countdown on the transition only (a
    /// `listening <-> away` flicker leaves it alone); either side speaking or thinking cancels it.
    fn quiet_edge(&mut self, now: Ts) {
        let Some(nav) = self.nav.as_ref() else { return };
        let user_quiet = matches!(self.states.1, UserState::Listening | UserState::Away);
        let agent_quiet = matches!(
            self.states.0,
            AgentState::Listening | AgentState::Idle | AgentState::Initializing
        );
        match (user_quiet && agent_quiet, self.deadline.is_some()) {
            (true, false) => {
                let at = now.saturating_add(nav.max_silence);
                self.deadline = Some(at);
                self.peer
                    .journal(Step::Core(CoreStep::TimerArmed { timer: TurnTimer::IvrSilence, at }));
            }
            (false, true) => self.cancel_silence(),
            (true, true) | (false, false) => {}
        }
    }

    fn cancel_silence(&mut self) {
        self.deadline = None;
        self.peer.journal(Step::Core(CoreStep::TimerCancelled(TurnTimer::IvrSilence)));
    }

    /// The countdown elapsed: `_on_silence_detected`.
    fn on_timer(&mut self, now: Ts) {
        if self.deadline.take().is_none() {
            return;
        }
        self.peer.journal(Step::Core(CoreStep::TimerFired(TurnTimer::IvrSilence, now)));
        self.reply(IvrReplyOrigin::Silence, true);
    }

    /// `generate_reply(allow_interruptions=..)` with no input, under the navigator's origin; nobody
    /// awaits the handle, as in the reference.
    fn reply(&mut self, origin: IvrReplyOrigin, allow_interruptions: bool) {
        self.peer.journal(Step::Detector(DetectorStep::IvrTriggered { origin }));
        let _ = self.peer.send(
            Origin::Ivr,
            SpeechCommand::GenerateReply {
                user_input: None,
                instructions: None,
                allow_interruptions: Some(allow_interruptions),
                tool_choice: None,
                chat_ctx: None,
                toolset: None,
                on_enter: false,
                call: None,
            },
        );
    }
}
