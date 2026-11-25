//! AMD's own classifier history (`_chat_context.py`) and the attribution of session and AMD
//! transcripts to committed turns (`_stt.py`, without the stream, which is a leg).

use crate::amd::effects::{ClassifierItem, TurnTranscript};
use lk_agents_abi::{FunctionCall, FunctionCallOutput, Ts};

/// At most this many items go into a request (spec §8.1).
pub(super) const HISTORY_LIMIT: usize = 20;

/// User turns, agent replies and DTMF calls, in speaking order. Each item keeps the time it is
/// ordered by: an agent reply is inserted at its start, since the session reports it when its
/// speech ends and a turn committed meanwhile arrives first.
#[derive(Debug, Default)]
pub(super) struct ClassifierHistory {
    items: Vec<(Ts, ClassifierItem)>,
}

impl ClassifierHistory {
    /// `add_transcript`: a committed turn with text.
    pub(super) fn add_user(&mut self, at: Ts, turn_id: u64, transcript: &TurnTranscript) {
        let item = ClassifierItem::User { turn_id, transcript: transcript.clone() };
        self.items.push((at, item));
    }

    /// `update_transcript`: the turn's message gets its late AMD final.
    pub(super) fn update_user(&mut self, turn_id: u64, transcript: &TurnTranscript) {
        update_user(self.items.iter_mut().map(|(_, item)| item), turn_id, transcript);
    }

    /// `add_agent_reply`: inserted after the last item at or before `started_at`.
    pub(super) fn add_agent(&mut self, started_at: Ts, text: String) {
        let at = self.items.iter().rposition(|(t, _)| *t <= started_at).map_or(0, |i| i + 1);
        self.items.insert(at, (started_at, ClassifierItem::Agent { text }));
    }

    /// `add_tool_result`: a successful, non-empty `send_dtmf_events` call, once per call.
    pub(super) fn add_dtmf(&mut self, at: Ts, call: &FunctionCall, output: &FunctionCallOutput) {
        let seen = (self.items.iter())
            .any(|(_, item)| matches!(item, ClassifierItem::DtmfCall(c) if c.id == call.id));
        if call.name != "send_dtmf_events" || output.is_error || output.output.is_empty() || seen {
            return;
        }
        self.items.push((at, ClassifierItem::DtmfCall(call.clone())));
        self.items.push((at, ClassifierItem::DtmfOutput(output.clone())));
    }

    /// `truncate(max_items=20)`: the latest items. A cut never leaves a DTMF call or output
    /// first.
    pub(super) fn window(&self) -> Vec<ClassifierItem> {
        let items = self.items.iter().map(|(_, item)| item);
        if self.items.len() <= HISTORY_LIMIT {
            return items.cloned().collect();
        }
        (items.skip(self.items.len() - HISTORY_LIMIT))
            .skip_while(|item| {
                matches!(item, ClassifierItem::DtmfCall(_) | ClassifierItem::DtmfOutput(_))
            })
            .cloned()
            .collect()
    }
}

/// Replaces the first message of `turn_id` with the turn's transcript.
pub(super) fn update_user<'a>(
    items: impl Iterator<Item = &'a mut ClassifierItem>,
    turn_id: u64,
    transcript: &TurnTranscript,
) {
    for item in items {
        if let ClassifierItem::User { turn_id: id, transcript: t } = item
            && *id == turn_id
        {
            *t = transcript.clone();
            return;
        }
    }
}

/// `AMDTranscriber`'s bookkeeping: the open turn's session and AMD finals, and which STTs are in
/// use. Neither source wins: the classifier sees both.
#[derive(Debug)]
pub(super) struct Transcripts {
    /// Session STT exists, or AMD has no STT of its own; set again when AMD's STT fails.
    session_enabled: bool,
    /// AMD's STT exists and has not failed.
    amd_enabled: bool,
    session: String,
    amd: String,
}

/// Which STT a final came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Source {
    Session,
    Amd,
}

impl Transcripts {
    pub(super) fn new(session_stt: bool, amd_stt: bool) -> Self {
        Transcripts {
            session_enabled: session_stt || !amd_stt,
            amd_enabled: amd_stt,
            session: String::new(),
            amd: String::new(),
        }
    }

    pub(super) fn amd_active(&self) -> bool {
        self.amd_enabled
    }

    /// Joins a final to the open turn; a commit's own transcript replaces the session's. A blank
    /// text changes nothing.
    pub(super) fn push(&mut self, source: Source, text: &str, replace: bool) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        let slot = match source {
            Source::Session => &mut self.session,
            Source::Amd => &mut self.amd,
        };
        if replace || slot.is_empty() {
            *slot = text.to_owned();
        } else {
            slot.push(' ');
            slot.push_str(text);
        }
    }

    /// `end_turn`: the committed turn's transcripts, and whether its AMD final may still come
    /// (spec §6.2: AMD's STT is active, the turn has no AMD text, and it has session text or
    /// there is no session STT).
    pub(super) fn end_turn(&mut self, committed: &str) -> (TurnTranscript, bool) {
        self.push(Source::Session, committed, true);
        let session = std::mem::take(&mut self.session);
        let amd = std::mem::take(&mut self.amd);
        let late =
            self.amd_enabled && amd.is_empty() && (!session.is_empty() || !self.session_enabled);
        let transcript = TurnTranscript {
            session: self.session_enabled.then_some(session),
            amd: self.amd_enabled.then_some(amd),
        };
        (transcript, late)
    }

    /// `fail`: the open turn and later turns use the session transcript alone. False when AMD's
    /// STT had already stopped.
    pub(super) fn fail(&mut self) -> bool {
        if !self.amd_enabled {
            return false;
        }
        self.amd_enabled = false;
        self.session_enabled = true;
        true
    }
}
