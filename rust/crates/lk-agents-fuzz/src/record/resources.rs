//! What the harness asks of a recording while a run is going: which resource a case's `Ref`
//! names, and what the driver has reached. Rules read none of this; they read the views.

use std::collections::BTreeSet;

use lk_agents_voice::{
    agent_session::HookCall,
    trace::{Step, step_names},
};

use super::{Applied, Detail, Host, Id, Kind, Observed, Opened, Recording};
use crate::input::Input;

/// One resource the run opened, with the input behind it.
#[derive(Clone, Copy)]
pub struct Resource<'a> {
    record: &'a Recording,
    opened: &'a Opened,
}

impl<'a> Resource<'a> {
    pub fn kind(&self) -> Kind {
        self.opened.kind
    }

    /// Which one of its kind, counting from zero: what a `Ref` in a case names.
    pub fn nth(&self) -> u32 {
        self.opened.nth
    }

    pub fn id(&self) -> Id {
        self.opened.id
    }

    pub fn detail(&self) -> &'a Detail {
        &self.opened.detail
    }

    pub fn opened_at(&self) -> Observed {
        self.opened.at
    }

    pub fn closed_at(&self) -> Option<Observed> {
        self.record
            .closed
            .iter()
            .find(|(_, kind, nth)| *kind == self.opened.kind && *nth == self.opened.nth)
            .map(|(at, ..)| *at)
    }

    /// Whether an input may still point at it.
    pub fn is_live(&self) -> bool {
        self.record.live.has(self.opened.kind, self.opened.nth)
    }

    /// The input that opened this, if the case opened it at all.
    ///
    /// Exact, not nearest: an input and everything it opens share one stamp. A resource the
    /// driver opened itself matches nothing and gives `None`.
    pub fn opened_by(&self) -> Option<&'a Applied> {
        self.record.applied.iter().find(|a| a.at == self.opened.at)
    }
}

impl Recording {
    /// The inputs the case applied that `pick` matches, with their stamps. The release phase
    /// asks this to know what the case has already released.
    pub fn sent<'a, T>(
        &'a self,
        pick: impl Fn(&'a Input) -> Option<T> + 'a,
    ) -> impl Iterator<Item = (Observed, T, bool)> + 'a {
        self.applied.iter().filter_map(move |a| pick(&a.input).map(|t| (a.at, t, a.accepted)))
    }

    /// Steps anywhere in the journal that `pick` matches. The release phase asks this for the
    /// driver's armed deadlines and the navigator's triggers.
    pub fn journal_where<'a, T>(
        &'a self,
        pick: impl Fn(&'a Step) -> Option<T> + 'a,
    ) -> impl Iterator<Item = (Observed, T)> + 'a {
        self.journal.iter().filter_map(move |(at, step)| pick(step).map(|t| (*at, t)))
    }

    /// The n-th resource of a kind, named the way a case names one.
    pub fn resource(&self, kind: Kind, nth: u32) -> Option<Resource<'_>> {
        self.opened
            .iter()
            .find(|o| o.kind == kind && o.nth == nth)
            .map(|opened| Resource { record: self, opened })
    }

    /// Every resource of a kind the run opened, live or not.
    pub fn resources(&self, kind: Kind) -> impl Iterator<Item = Resource<'_>> {
        self.opened
            .iter()
            .filter(move |o| o.kind == kind)
            .map(move |opened| Resource { record: self, opened })
    }

    /// Journal step variants this run produced, as `Family::Variant`.
    pub fn journal_names(&self) -> BTreeSet<String> {
        // The folded view is part of the vocabulary: `ByAudioActivityRepeated` is only ever produced by
        // folding a run of identical barge steps, so a journal that folds to it reached it. The
        // rules keep reading the raw steps; only the reach count reads the fold.
        let steps: Vec<Step> = self.journal.iter().map(|(_, step)| step.clone()).collect();
        lk_agents_voice::trace::folded(&steps).iter().map(step_name).collect()
    }

    /// Journal step variants the driver can produce that this run did not.
    ///
    /// The search steers by this, so it must not overstate itself: a name the vocabulary does not
    /// know means the two spellings have drifted, and it panics rather than quietly shrinking.
    pub fn journal_gaps(&self) -> Vec<String> {
        let known = step_names();
        let reached = self.journal_names();
        if let Some(stray) = reached.iter().find(|name| !known.contains(name)) {
            panic!(
                "step_name produced `{stray}`, which trace::step_names does not list: the two \
                 spellings of a journal variant have drifted apart"
            );
        }
        known.into_iter().filter(|name| !reached.contains(name)).collect()
    }
}

/// A journal step's variant path, spelled `Family::Variant`.
///
/// Read off `Debug`, because a `Step` is thirteen nested enums with no other way to ask which
/// variant it is. [`Recording::journal_gaps`] checks the spelling matches [`step_names`].
pub fn step_name(step: &Step) -> String {
    let cut = |s: &str| s.find(['(', '{', ' ', ')']).unwrap_or(s.len());
    // The two names lead the text: formatting stops once both are written, not at the payload's
    // end.
    struct Head<F>(String, F);
    impl<F: Fn(&str) -> usize> std::fmt::Write for Head<F> {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.0.push_str(s);
            let family = (self.1)(&self.0);
            let rest = self.0[family..].trim_start_matches('(');
            let done = family < self.0.len() && (self.1)(rest) < rest.len();
            if done { Err(std::fmt::Error) } else { Ok(()) }
        }
    }
    let mut head = Head(String::new(), cut);
    let _ = std::fmt::Write::write_fmt(&mut head, format_args!("{step:?}"));
    let text = head.0;
    let family = &text[..cut(&text)];
    let rest = text[family.len()..].trim_start_matches('(');
    match &rest[..cut(rest)] {
        "" => family.to_string(),
        variant => format!("{family}::{variant}"),
    }
}

/// The hook a call names, for the projection the recording keeps.
pub fn hook_of(call: &HookCall) -> Host {
    let ctx_ids = match call {
        HookCall::OnUserTurnCompleted { chat_ctx, .. } => {
            chat_ctx.iter().map(|i| i.id().to_owned()).collect()
        }
        _ => Vec::new(),
    };
    Host::Hook { id: call.id(), kind: call.kind(), speech: None, ctx_ids }
}
