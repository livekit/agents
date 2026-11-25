//! The scripted AMD answerer. It plays AMD's side of the reply gate's protocol: it attaches,
//! activates, answers each turn it observed, and completes. Like every actor it acts only on what
//! it observed: a turn it can answer exists only once the session reported its commit. When the
//! setup carries AMD (`Setup::amd`), AMD's executor answers the gate instead, and the answerer
//! plays its models: it answers each request the executor made by the callee's script, late or
//! at once, sometimes with a failure, sometimes never, and now and then after the executor
//! abandoned it.
//!
//! The search chooses when it answers: at once (before the customer hook returns), late, for a
//! turn a newer one already superseded, twice for the latest turn, for a turn the session has not
//! committed or for an earlier run's turn, or by completing while a turn is still in its hook,
//! before its first decision or before it started listening. A completed run may be followed by a
//! second one.

use std::collections::BTreeMap;

use fastrand::Rng;

use super::{Halt, Talk};
use crate::{
    input::{AmdOpening, AmdSays, AmdVerdict, Input, Ref},
    record::{Kind, amd_run::AskKind},
};

/// Who answered the call, drawn once per run: the categories AMD's classifier would report turn
/// by turn (screener -> human -> human; voicemail -> message -> menu; hold -> `wait`).
#[derive(Clone, Copy, Debug)]
enum Callee {
    Human,
    Screener,
    Voicemail,
    Hold,
}

impl Callee {
    /// What AMD's classifier answers turn by turn (the proposal's DST personas).
    fn says(self, turn: usize) -> AmdSays {
        let script: &[AmdSays] = match self {
            Callee::Human => &[AmdSays::Human],
            Callee::Screener => &[AmdSays::Screening, AmdSays::Human, AmdSays::Human],
            Callee::Voicemail => &[AmdSays::Voicemail, AmdSays::Voicemail, AmdSays::Ivr],
            Callee::Hold => &[AmdSays::Wait, AmdSays::Wait, AmdSays::Human],
        };
        script[turn.min(script.len() - 1)]
    }

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

/// How often, per beat, AMD sends the latest turn's decision again: a duplicate.
const REPEAT: u8 = 3;

/// How often, per beat, AMD answers for a turn the session has not committed, or for an earlier
/// run's turn.
const STRAY: u8 = 2;

/// How often a completed run is followed by a second one.
const REATTACH: u8 = 30;

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
    /// Completes before it starts listening, once attached.
    detaches_pending: bool,
    last: AmdVerdict,
    detached: bool,
    /// A second run attaches once this one completed.
    reattaches: bool,
    /// Each request of AMD's executor seen, by `AmdAsk` index: when its answer is due, and what
    /// it says; `None` for one never answered.
    asks: BTreeMap<u32, Option<(u64, AmdSays)>>,
    /// Classifications answered so far: the callee's script goes by them.
    classified: usize,
}

impl Answerer {
    pub(super) fn draw(seed: u64) -> Self {
        let mut rng = Rng::with_seed(seed ^ 0x0000_0041_4d44_6765);
        let on = rng.u8(0..100) < ATTACHES;
        let callee = *[Callee::Human, Callee::Screener, Callee::Voicemail, Callee::Hold]
            .get(rng.usize(0..4))
            .unwrap_or(&Callee::Human);
        let opening = match rng.u8(0..3) {
            0 => None,
            1 => Some(AmdOpening::Text),
            _ => Some(AmdOpening::Instructions),
        };
        // Completes after this many decisions: none (before the first decision, or even
        // before it starts listening) up to four.
        let completes_after = (rng.u8(0..100) < 70).then(|| rng.usize(0..=4));
        let detaches_pending = completes_after == Some(0) && rng.bool();
        let forbids = rng.u8(0..100) < 20;
        let reattaches = rng.u8(0..100) < REATTACH;
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
            detaches_pending,
            last: AmdVerdict::Allow,
            detached: false,
            reattaches,
            asks: BTreeMap::new(),
            classified: 0,
        }
    }
}

impl Talk {
    /// AMD's beat: attach, activate when due, schedule a decision for each turn it observed and
    /// send what is due, now and then a stale one, and complete when its script says.
    pub(super) async fn answer_amd(&mut self) -> Result<(), Halt> {
        if self.run.setup.amd.is_some() {
            return self.answer_amd_models().await;
        }
        if !self.amd.on {
            return Ok(());
        }
        // A second run: attached afresh, with its own turns, after the first completed.
        if self.amd.detached {
            if !self.amd.reattaches || self.amd.rng.u8(0..100) >= 20 {
                return Ok(());
            }
            self.amd.reattaches = false;
            self.amd.attached = false;
            self.amd.detached = false;
            self.amd.due.clear();
            self.amd.answered = 0;
            self.amd.detaches_pending = false;
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
        // A completion before it ever listened: a callee that never answers.
        if self.amd.detaches_pending && self.amd.activate_at_ms.is_some() {
            if self.amd.rng.u8(0..100) < 30 {
                self.amd.detached = true;
                self.amd.activate_at_ms = None;
                let forbid = self.amd.forbids;
                return self.send(Input::AmdDetach { category: AmdVerdict::Allow, forbid }).await;
            }
            return Ok(());
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
        // The latest turn's decision again, once sent: the session drops the duplicate.
        if let Some(&latest) = turns.last()
            && self.amd.due.get(&latest).is_some_and(|(at, _)| *at == u64::MAX)
            && self.amd.rng.u8(0..100) < REPEAT
        {
            let verdict = self.amd.callee.verdict(self.amd.rng.usize(0..3));
            self.send(Input::AmdDecide { turn: Ref::new(latest), verdict }).await?;
        }
        // A decision for a turn the session has not committed, or for an earlier run's turn.
        if self.amd.rng.u8(0..100) < STRAY {
            let verdict = self.amd.callee.verdict(0);
            let earlier: Vec<u32> = (self.run.record.resources(Kind::AmdTurn))
                .filter(|t| !self.run.record.live().has(Kind::AmdTurn, t.nth()))
                .map(|t| t.nth())
                .collect();
            if earlier.is_empty() || self.amd.rng.bool() {
                self.send(Input::AmdDecideAhead { verdict }).await?;
            } else {
                let old = earlier[self.amd.rng.usize(0..earlier.len())];
                self.send(Input::AmdDecide { turn: Ref::new(old), verdict }).await?;
            }
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

    /// AMD's executor answers the gate; this beat plays its models. It enters once (again after
    /// a completion, now and then), answers each request when due, sometimes answers one the
    /// executor already abandoned, and now and then closes the run.
    async fn answer_amd_models(&mut self) -> Result<(), Halt> {
        let now = self.run.record.now().at_ms;
        if !self.amd.attached {
            self.amd.attached = true;
            return self.send(Input::AmdEnter {}).await;
        }
        let live = self.run.record.live().of(Kind::AmdAsk);
        let asks: Vec<(u32, AskKind)> = (self.run.record.amd_run().asks().into_iter())
            .map(|(nth, _, asked)| (nth, asked.kind))
            .collect();
        for (nth, kind) in &asks {
            if self.amd.asks.contains_key(nth) || !live.contains(nth) {
                continue;
            }
            let delay =
                if self.amd.rng.u8(0..100) < AT_ONCE { 0 } else { self.amd.rng.u64(100..3_000) };
            let says = match (kind, self.amd.rng.u8(0..100)) {
                // A request never answered: the executor's deadline decides.
                (_, 0..5) => None,
                (_, 5..9) => Some(AmdSays::Failed),
                (_, 9..12) => Some(AmdSays::Garbled),
                (_, 12) => Some(AmdSays::Crashed),
                (AskKind::Menu, n) => {
                    Some(if n % 2 == 0 { AmdSays::Menu } else { AmdSays::NoMenu })
                }
                (AskKind::Classify | AskKind::Choice, _) => {
                    self.amd.classified += 1;
                    Some(self.amd.callee.says(self.amd.classified - 1))
                }
            };
            self.amd.asks.insert(*nth, says.map(|says| (now + delay, says)));
        }
        let due: Vec<(u32, AmdSays)> = (self.amd.asks.iter_mut())
            .filter_map(|(nth, plan)| {
                let (at, says) = (*plan)?;
                (at <= now).then(|| {
                    *plan = None;
                    (*nth, says)
                })
            })
            .collect();
        for (nth, says) in due {
            self.send(Input::AmdReply { ask: Ref::new(nth), says }).await?;
        }
        // A request the executor abandoned, answered anyway: a classifier that ignores
        // cancellation.
        let gone: Vec<u32> =
            asks.iter().map(|(nth, _)| *nth).filter(|nth| !live.contains(nth)).collect();
        if !gone.is_empty() && self.amd.rng.u8(0..100) < STALE {
            let nth = gone[self.amd.rng.usize(0..gone.len())];
            let says = self.amd.callee.says(0);
            self.send(Input::AmdReply { ask: Ref::new(nth), says }).await?;
        }
        if self.amd.rng.u8(0..100) < 1 {
            self.send(Input::AmdClose {}).await?;
        }
        Ok(())
    }
}
