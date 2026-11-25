//! Reply policy (spec §10.1), the speeches AMD watches, the voicemail record (spec §10.4) and
//! the idle timer.

use super::{Amd, VoicemailAttempt, WatchedSpeech, stage};
use crate::agent_session::machine::activity::scheduling::SpeechId;
use crate::amd::effects::{AGENT_REPLY, AmdEffect, AmdTimer};
use crate::amd::events::{AmdReason, Category, Stage};
use crate::amd::options::VOICEMAIL_PLAYED_NOTE;
use crate::amd::{AmdAnswer, ReplyDecision, SpeechObservation};
use crate::audio_recognition::TurnId;
use lk_agents_abi::Ts;

impl Amd {
    /// The latest turn's decision, once its prediction exists and AMD is still active. A
    /// prediction that completes AMD sends none: `Detach.pending` decides its turn.
    pub(super) fn decide(&mut self) {
        if !self.active() {
            return;
        }
        let Some(turn) = self.latest else { return };
        let Some(t) = self.turns.get_mut(&turn) else { return };
        if t.decided || t.prediction.is_none() {
            return;
        }
        t.decided = true;
        if t.agent_changed {
            return self.finish(AmdReason::AgentChanged);
        }
        let decision = self.reply_rule(turn);
        // Decided now: a later turn's prediction can arrive while the reply plays.
        let words = self.shows_agent_words();
        if let Some(t) = self.turns.get_mut(&turn) {
            t.reply_words = words;
        }
        // A turn that asked for no reply, or ended without one, has no reply to hold.
        let no_reply = self.turns.get(&turn).is_some_and(|t| t.no_reply);
        if decision.allow && decision.silence.is_some() && !no_reply {
            self.held = Some(turn);
        }
        self.answer(AmdAnswer::ReplyDecided { run: self.run, turn, decision });
    }

    /// `_reply_held_at`: the latest turn's allowed reply would wait for silence now. The
    /// participant speaks, or has been silent for less than the threshold.
    pub(super) fn reply_would_hold(&self, now: Ts) -> bool {
        let threshold = self.options.machine_silence_threshold;
        let Some(t) = self.latest.and_then(|turn| self.turns.get(&turn)) else { return false };
        if !self.active()
            || threshold.is_zero()
            || t.prediction.is_none()
            || t.no_reply
            || self.category == Category::Wait
            || !self.stage.is_machine()
        {
            return false;
        }
        let speech = &self.speech;
        speech.speaking || speech.ended_at.is_none_or(|ended| now < ended.saturating_add(threshold))
    }

    /// `_authorize_reply` for an active run, from the row after "AMD completed".
    fn reply_rule(&mut self, turn: TurnId) -> ReplyDecision {
        if self.category == Category::Wait {
            return ReplyDecision::deny();
        }
        let threshold = self.options.machine_silence_threshold;
        let silence = (self.stage.is_machine() && !threshold.is_zero()).then_some(threshold);
        let dtmf = self.stage == Stage::MachineIvr;
        if self.likely_human {
            // Possibly a person: reply without the stage's instructions.
            return ReplyDecision { allow: true, instructions: None, dtmf, silence };
        }
        match self.stage {
            Stage::MachineVm => {
                let vm = &self.voicemail;
                if vm.played || vm.attempt.is_some() || vm.reserved.is_some() {
                    return ReplyDecision::deny();
                }
                self.voicemail.reserved = Some(turn);
                let instructions = self.stage_instructions(Stage::MachineVm);
                ReplyDecision { allow: true, instructions, dtmf: false, silence }
            }
            Stage::MachineScreening | Stage::MachineIvr => {
                let instructions = self.stage_instructions(self.stage);
                ReplyDecision { allow: true, instructions, dtmf, silence }
            }
            Stage::Uncertain => ReplyDecision::allow(),
            // These complete AMD, so an active run never decides in them.
            Stage::Human | Stage::MachineUnavailable => ReplyDecision::deny(),
        }
    }

    /// `_get_instructions`.
    fn stage_instructions(&self, stage: Stage) -> Option<String> {
        let mut text = self.options.instructions(stage)?.to_owned();
        if stage == Stage::MachineIvr && self.voicemail.played {
            text.push_str(VOICEMAIL_PLAYED_NOTE);
        }
        Some(text)
    }

    /// The "AMD completed" row, after the `wait` row, for every turn `Detach` settles.
    /// `agent_changed` denies too: Python denies the turn that found the change.
    pub(super) fn completion_decision(&self, reason: AmdReason) -> ReplyDecision {
        if reason == AmdReason::AgentChanged
            || self.category == Category::Wait
            || self.forbids_reply()
        {
            return ReplyDecision::deny();
        }
        let human_after_machine = self.stage == Stage::Human && self.had_machine_stage;
        let instructions =
            if human_after_machine { self.options.human_instructions.clone() } else { None };
        ReplyDecision { allow: true, instructions, dtmf: false, silence: None }
    }

    /// The first row: a turn not bound to an AMD turn.
    pub(super) fn unbound_decision(&self) -> ReplyDecision {
        if self.forbids_reply() || self.category == Category::Wait {
            ReplyDecision::deny()
        } else {
            ReplyDecision::allow()
        }
    }

    /// `_forbids_reply`: the call is unavailable or in a stage the application handles.
    pub(super) fn forbids_reply(&self) -> bool {
        self.stage == Stage::MachineUnavailable || self.options.returns_early(self.stage)
    }

    /// `_shows_agent_words`: the next turn may answer the agent's words. True after a likely
    /// human turn, or after an uncertain turn in a menu or screener.
    pub(super) fn shows_agent_words(&self) -> bool {
        self.likely_human
            || (stage::is_likely_human_stage(self.stage) && self.category == Category::Uncertain)
    }

    // Speeches.

    pub(super) fn on_reply_created(&mut self, turn: TurnId, speech: SpeechId) {
        if self.finished() {
            return;
        }
        let words = self.turns.get(&turn).map(|t| t.reply_words);
        self.trace.push(crate::trace::AmdStep::ReplyBound { run: self.run, turn, speech });
        let watched = self.watch(speech);
        if let Some(words) = words {
            watched.words = words;
        }
        // The voicemail message: the reserved turn's reply, or a correction's replacement of it,
        // whose first speech's interrupted end no longer counts.
        let vm = &mut self.voicemail;
        let tracks = vm.reserved == Some(turn) || vm.attempt.is_some_and(|a| a.turn == turn);
        if self.lifecycle == super::Lifecycle::Active && tracks {
            vm.reserved = None;
            vm.attempt =
                Some(VoicemailAttempt { turn, speech, played_out: false, item_interrupted: false });
        }
    }

    fn watch(&mut self, speech: SpeechId) -> &mut WatchedSpeech {
        let words = self.shows_agent_words();
        self.speeches.entry(speech).or_insert(WatchedSpeech {
            words,
            add_to_chat_ctx: true,
            before_entry: false,
            started_at: None,
        })
    }

    pub(super) fn on_speech(&mut self, speech: SpeechId, event: SpeechObservation, at: Ts) {
        if self.finished() {
            return;
        }
        match event {
            SpeechObservation::Created { add_to_chat_ctx, playing, .. } => {
                let watched = self.watch(speech);
                watched.add_to_chat_ctx = add_to_chat_ctx;
                watched.before_entry |= playing;
            }
            SpeechObservation::ItemAdded { text, interrupted } => {
                if let Some(attempt) =
                    self.voicemail.attempt.as_mut().filter(|a| a.speech == speech)
                {
                    attempt.item_interrupted |= interrupted;
                }
                let Some(watched) = self.speeches.get(&speech) else { return };
                let Some(text) = text.filter(|text| !text.is_empty()) else { return };
                if !watched.add_to_chat_ctx || watched.before_entry {
                    return;
                }
                let started_at = watched.started_at.unwrap_or(at);
                let content = if watched.words { text } else { AGENT_REPLY.to_owned() };
                self.history.add_agent(started_at, content);
            }
            SpeechObservation::PlayoutAdvanced => {
                if let Some(watched) = self.speeches.get_mut(&speech) {
                    watched.started_at.get_or_insert(at);
                }
                if let Some(attempt) =
                    self.voicemail.attempt.as_mut().filter(|a| a.speech == speech)
                {
                    attempt.played_out = true;
                }
            }
            SpeechObservation::Done { interrupted, failed } => {
                self.speeches.remove(&speech);
                self.voicemail_done(speech, interrupted || failed);
            }
        }
    }

    /// `_on_voicemail_done`: the message counts as played only if it ended whole, played out,
    /// while AMD was active. An interrupted attempt can be retried.
    fn voicemail_done(&mut self, speech: SpeechId, cut: bool) {
        let Some(attempt) = self.voicemail.attempt.take_if(|a| a.speech == speech) else { return };
        if !self.active() || cut || attempt.item_interrupted || !attempt.played_out {
            return;
        }
        self.voicemail.played = true;
        self.trace.push(crate::trace::AmdStep::VoicemailPlayed { run: self.run, speech });
        if self.stage == Stage::MachineVm {
            // The idle timer restarts, so a menu after the message can still arrive.
            self.clear_idle();
        }
    }

    // Idle.

    /// `_update_idle`, after every input while active.
    pub(super) fn update_idle(&mut self, now: Ts) {
        if !self.active() {
            return;
        }
        if self.timeouts >= self.options.max_inference_timeouts
            && self.request.is_none()
            && self.held.is_none()
        {
            return self.finish(AmdReason::InferenceTimeout);
        }
        let busy = self.request.is_some()
            // Hold music or an advertisement; the overall timeout still applies.
            || self.category == Category::Wait
            || self.held.is_some()
            || self.speech.speaking
            // Every queued or playing speech except the preemptive generation.
            || self.agent_busy;
        if busy {
            self.clear_idle();
        } else if self.idle_deadline.is_none() {
            let timeout = if self.stage == Stage::MachineVm {
                self.options.voicemail_idle_timeout
            } else {
                self.options.idle_timeout
            };
            let deadline = now.saturating_add(timeout);
            self.idle_deadline = Some(deadline);
            self.push(AmdEffect::SetDeadline(AmdTimer::Idle, deadline));
        }
    }
}
