//! AMD on the call: the real run, or the scripted answerer.
//!
//! A real run (`Input::AmdEnter`) is the production executor and machine over fake models whose
//! categories follow a callee persona; it answers on its own, and the actor only enters it,
//! sometimes closes it, and sometimes enters a second one after it completed.
//!
//! The scripted answerer plays AMD's side of the reply gate's protocol where the real machine
//! would not go: it attaches, activates, answers each turn it observed, and completes. Like
//! every actor it acts only on what it observed: a turn it can answer exists only once the
//! session reported its commit.
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
    input::{AmdModel, AmdOpening, AmdPersona, AmdVerdict, Count, Input, Ref},
    record::Kind,
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

/// How often AMD on the call is a real run rather than the scripted answerer.
const REAL: u8 = 50;

/// How often, per beat, the host closes a real run (`aclose()`).
const CLOSE: u8 = 1;

/// A real run's draw: who answers, what classifies, and AMD's own STT.
#[derive(Clone, Copy, Debug)]
struct Real {
    persona: AmdPersona,
    model: AmdModel,
    own_stt: bool,
    stt_fails: bool,
    seed: u32,
    /// Runs entered so far.
    entered: u32,
    closed: bool,
}

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
    /// A real run instead of the scripted answers.
    real: Option<Real>,
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
        // Completes after this many decisions: none (before the first decision, or even
        // before it starts listening) up to four.
        let completes_after = (rng.u8(0..100) < 70).then(|| rng.usize(0..=4));
        let detaches_pending = completes_after == Some(0) && rng.bool();
        let forbids = rng.u8(0..100) < 20;
        let reattaches = rng.u8(0..100) < REATTACH;
        // Its own stream, so the scripted draws above stay as they were.
        let mut real = Rng::with_seed(seed ^ 0x0000_5245_414c_414d);
        let real = (real.u8(0..100) < REAL).then(|| Real {
            persona: *[
                AmdPersona::Human,
                AmdPersona::Screener,
                AmdPersona::Voicemail,
                AmdPersona::Hold,
                AmdPersona::Unavailable,
            ]
            .get(real.usize(0..5))
            .unwrap_or(&AmdPersona::Human),
            model: if real.u8(0..100) < 25 { AmdModel::Decision } else { AmdModel::Llm },
            own_stt: real.bool(),
            stt_fails: real.u8(0..100) < 10,
            seed: real.u32(..),
            entered: 0,
            closed: false,
        });
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
            real,
        }
    }
}

impl Talk {
    /// AMD's beat: attach, activate when due, schedule a decision for each turn it observed and
    /// send what is due, now and then a stale one, and complete when its script says.
    pub(super) async fn answer_amd(&mut self) -> Result<(), Halt> {
        if !self.amd.on {
            return Ok(());
        }
        if self.amd.real.is_some() {
            return self.enter_amd().await;
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
}

impl Talk {
    /// A real run's beat: enter it, now and then close it, and enter a second one after the
    /// first completed.
    async fn enter_amd(&mut self) -> Result<(), Halt> {
        let Some(mut real) = self.amd.real else { return Ok(()) };
        let detached = (self.run.record.amd().iter())
            .filter(|(_, _, o)| matches!(o, lk_agents_voice::amd::AmdObservation::Detached { .. }))
            .count();
        let second = real.entered == 1
            && detached >= 1
            && self.amd.reattaches
            && self.amd.rng.u8(0..100) < 20;
        if real.entered == 0 || second {
            real.entered += 1;
            real.closed = false;
            self.amd.real = Some(real);
            return self
                .send(Input::AmdEnter {
                    persona: real.persona,
                    model: real.model,
                    opening: self.amd.opening,
                    own_stt: real.own_stt,
                    stt_fails: real.stt_fails,
                    seed: Count(real.seed.wrapping_add(real.entered)),
                })
                .await;
        }
        if !real.closed && self.amd.rng.u8(0..100) < CLOSE {
            real.closed = true;
            self.amd.real = Some(real);
            return self.send(Input::AmdClose {}).await;
        }
        Ok(())
    }
}
