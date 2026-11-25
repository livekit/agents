//! The scripted AMD answerer (the AMD proposal, "DST": "Scripted AMD"). Before the AMD
//! machine exists, it plays AMD's side of the reply gate's protocol: it attaches, activates,
//! answers each turn it observed, and completes. Like every actor it acts only on what it
//! observed: a turn it can answer exists only once the session reported its commit.
//!
//! The search chooses when it answers: at once (before the customer hook returns), late, for a
//! turn a newer one already superseded, or by completing while a turn is still in its hook.

use std::collections::BTreeMap;

use fastrand::Rng;

use super::{Halt, Talk};
use crate::{
    input::{AmdOpening, AmdVerdict, Input, Ref},
    record::Kind,
};

/// Who answered the call, drawn once per run: the categories AMD's classifier would report turn
/// by turn (the proposal's personas: screener -> human -> human; voicemail -> message -> menu;
/// hold -> `wait`).
#[derive(Clone, Copy, Debug)]
enum Callee {
    Human,
    Screener,
    Voicemail,
    Hold,
}

impl Callee {
    fn verdict(self, turn: usize) -> AmdVerdict {
        let script: &[AmdVerdict] = match self {
            Callee::Human => &[AmdVerdict::Allow],
            Callee::Screener => &[AmdVerdict::Stage, AmdVerdict::Allow],
            Callee::Voicemail => &[AmdVerdict::Silence, AmdVerdict::Silence, AmdVerdict::Ivr],
            Callee::Hold => &[AmdVerdict::Wait, AmdVerdict::Wait, AmdVerdict::Allow],
        };
        script[turn.min(script.len() - 1)]
    }
}

/// How often a run has AMD on its call.
const ATTACHES: u8 = 20;

/// How often a decision goes out the instant its turn is observed: before the customer hook
/// returns, often.
const AT_ONCE: u8 = 30;

/// How often, per beat, AMD answers an older turn than the latest: a stale answer.
const STALE: u8 = 3;

/// The scripted answerer's plan and state, drawn once per run from its own stream.
pub(super) struct Answerer {
    rng: Rng,
    on: bool,
    callee: Callee,
    attached: bool,
    activate_at_ms: Option<u64>,
    opening: Option<AmdOpening>,
    /// Each observed turn, by `AmdTurn` index: when its decision is due, and the verdict.
    due: BTreeMap<u32, (u64, AmdVerdict)>,
    answered: usize,
    /// Completes after this many decisions, or never.
    completes_after: Option<usize>,
    /// The completion forbids replies (`machine-unavailable`, a stage the application handles).
    forbids: bool,
    last: AmdVerdict,
    detached: bool,
}

impl Answerer {
    pub(super) fn draw(seed: u64, realtime: bool) -> Self {
        let mut rng = Rng::with_seed(seed ^ 0x0000_0041_4d44_6765);
        let on = !realtime && rng.u8(0..100) < ATTACHES;
        let callee = *[Callee::Human, Callee::Screener, Callee::Voicemail, Callee::Hold]
            .get(rng.usize(0..4))
            .unwrap_or(&Callee::Human);
        let opening = match rng.u8(0..3) {
            0 => None,
            1 => Some(AmdOpening::Text),
            _ => Some(AmdOpening::Instructions),
        };
        let completes_after = (rng.u8(0..100) < 70).then(|| rng.usize(1..=4));
        let forbids = rng.u8(0..100) < 20;
        Answerer {
            rng,
            on,
            callee,
            attached: false,
            activate_at_ms: None,
            opening,
            due: BTreeMap::new(),
            answered: 0,
            completes_after,
            forbids,
            last: AmdVerdict::Allow,
            detached: false,
        }
    }
}

impl Talk {
    /// AMD's beat: attach, activate when due, schedule a decision for each turn it observed and
    /// send what is due, now and then a stale one, and complete when its script says.
    pub(super) async fn answer_amd(&mut self) -> Result<(), Halt> {
        if !self.amd.on || self.amd.detached {
            return Ok(());
        }
        let now = self.run.record.now().at_ms;
        if !self.amd.attached {
            self.amd.attached = true;
            // AMD starts listening once the callee answers: at once, or seconds in, after the
            // participant may already be speaking.
            let delay = if self.amd.rng.bool() { 0 } else { self.amd.rng.u64(100..3_000) };
            self.amd.activate_at_ms = Some(now + delay);
            return self.send(Input::AmdAttach {}).await;
        }
        if let Some(at) = self.amd.activate_at_ms
            && now >= at
        {
            self.amd.activate_at_ms = None;
            let opening = self.amd.opening;
            self.send(Input::AmdActivate { opening }).await?;
        }
        let turns = self.run.record.live().of(Kind::AmdTurn);
        for nth in &turns {
            if self.amd.due.contains_key(nth) {
                continue;
            }
            let delay =
                if self.amd.rng.u8(0..100) < AT_ONCE { 0 } else { self.amd.rng.u64(100..3_000) };
            let verdict = self.amd.callee.verdict(self.amd.due.len());
            self.amd.due.insert(*nth, (now + delay, verdict));
        }
        // An older turn's answer, late: the session must drop it.
        if turns.len() > 1 && self.amd.rng.u8(0..100) < STALE {
            let older = turns[self.amd.rng.usize(0..turns.len() - 1)];
            let verdict = self.amd.callee.verdict(0);
            self.send(Input::AmdDecide { turn: Ref::new(older), verdict }).await?;
        }
        let ready: Vec<(u32, AmdVerdict)> = self
            .amd
            .due
            .iter_mut()
            .filter(|(_, (at, _))| *at <= now)
            .map(|(nth, (at, verdict))| {
                *at = u64::MAX;
                (*nth, *verdict)
            })
            .collect();
        for (nth, verdict) in ready {
            self.amd.answered += 1;
            self.amd.last = verdict;
            self.send(Input::AmdDecide { turn: Ref::new(nth), verdict }).await?;
        }
        if self.amd.completes_after.is_some_and(|n| self.amd.answered >= n)
            // A completion lands any beat after: with a turn in its hook, or a reply held.
            && self.amd.rng.u8(0..100) < 20
        {
            self.amd.detached = true;
            let (category, forbid) = (self.amd.last, self.amd.forbids);
            self.send(Input::AmdDetach { category, forbid }).await?;
        }
        Ok(())
    }
}
