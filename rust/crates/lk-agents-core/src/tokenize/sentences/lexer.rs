//! BlingFire's rule interpreter (`FALexTools_t::Process_int`): from each position, the longest
//! match of any rule (the earliest rule on a tie) acts on its span, and the scan resumes after it.
//!
//! Every decision reads at most `MAX_TOKEN_LEN` symbols ahead, so on text that may go on, a
//! decision is final once its automaton dies inside the text or its window ends before the last
//! character. The scan stops at the first decision that is not: everything before it stands,
//! and every end still to come lies at or after it.

use regex_automata::hybrid::dfa::Cache;
use regex_automata::{Anchored, Input};

use super::rules::{Action, END, OTHER, RuleSet, RuleSets, START, Set};

/// No match runs further than this many symbols from where it starts.
const MAX_TOKEN_LEN: usize = 300;
/// Rule sets call each other at most this deep, the top level included.
const MAX_DEPTH: usize = 3;

/// The lazy DFAs' state caches, one per rule set, kept between calls.
pub(super) struct Caches<'a> {
    sets: &'a RuleSets,
    caches: [Cache; 3],
    /// Every top-level match as `(rule, first, last)` character positions, and the
    /// `(set, rule)` of every rule that fired: what a test attributes a split to.
    #[cfg(test)]
    pub matches: Vec<(usize, usize, usize)>,
    #[cfg(test)]
    pub fired: Vec<(usize, usize)>,
}

impl<'a> Caches<'a> {
    pub(super) fn new(sets: &'a RuleSets) -> Self {
        Caches {
            sets,
            caches: sets.each_ref().map(|s| s.dfa.create_cache()),
            #[cfg(test)]
            matches: Vec::new(),
            #[cfg(test)]
            fired: Vec::new(),
        }
    }

    /// The positions (indices into `chars`) of the last character of each sentence, in scan
    /// order.
    pub(super) fn sentence_ends(&mut self, chars: &[char]) -> Vec<usize> {
        self.decided_ends(Text { chars, at_start: true, open: false }).0
    }

    /// The sentence ends of `text` decided whatever text follows it, and the symbol where the
    /// scan stopped: a scan resumes there, and every later end is at or after it.
    pub(super) fn decided_ends(&mut self, text: Text<'_>) -> (Vec<usize>, usize) {
        let mut ends = Vec::new();
        let stop = self.process(Set::Main, false, text, 0, 1, &mut ends);
        (ends, stop)
    }

    /// Returns the symbol the scan stopped at: past the text unless it is `open`.
    fn process(
        &mut self,
        which: Set,
        once: bool,
        text: Text<'_>,
        offset: usize,
        depth: usize,
        ends: &mut Vec<usize>,
    ) -> usize {
        let Text { chars, at_start, open } = text;
        let rules = &self.sets[which as usize];
        let lead = usize::from(at_start);
        let n = chars.len();
        if depth > MAX_DEPTH || n == 0 {
            return 0;
        }
        // The symbols, `^` text `$`, and the byte offset of each symbol in `symbols`.
        let mut symbols = String::with_capacity(n + 8);
        let mut at = Vec::with_capacity(n + 3);
        let start = at_start.then_some(START);
        let end = (!open).then_some(END);
        for c in start.into_iter().chain(chars.iter().map(|&c| symbol(c))).chain(end) {
            at.push(symbols.len());
            symbols.push(c);
        }
        at.push(symbols.len());
        let total = at.len() - 1;
        // Symbol `k` is `chars[k - lead]`; the last character is symbol `lead + n - 1`.
        let last_char = lead + n - 1;
        let char_of = |k: usize| k.saturating_sub(lead).min(n - 1);
        let mut s = 0;
        while s <= last_char {
            let within = s + MAX_TOKEN_LEN <= last_char;
            let window = if within { s + MAX_TOKEN_LEN } else { total };
            let (found, dead) =
                longest(rules, &mut self.caches[which as usize], &symbols[at[s]..at[window]]);
            if open && !within && !dead {
                return s;
            }
            let Some((matched, rule)) = found else {
                s += 1;
                continue;
            };
            let last = at.partition_point(|&b| b < at[s] + matched) - 1; // last matched symbol
            #[cfg(test)]
            {
                self.fired.push((which as usize, rule));
                if depth == 1 {
                    self.matches.push((rule, char_of(s) + offset, char_of(last) + offset));
                }
            }
            let rule = rules.rules[rule];
            // Spans are clamped to the text as BlingFire does.
            let from = char_of(s + rule.left);
            let to = char_of(last.saturating_sub(rule.right));
            match rule.action {
                Action::Eos => ends.push(to + offset),
                Action::EosAtFirst => ends.push(from + offset),
                Action::Call { set, once } if from <= to => {
                    let span = Text { chars: &chars[from..=to], at_start: true, open: false };
                    self.process(set, once, span, offset + from, depth + 1, ends);
                }
                Action::Call { .. } | Action::NoBreak => {}
            }
            if once {
                return total;
            }
            let resume = last as isize - rule.right as isize;
            s = if resume > s as isize { resume as usize + 1 } else { s + 1 };
        }
        s
    }
}

/// Characters to scan: `at_start` puts the `^` symbol before them; `open` text may go on, so it
/// has no `$` and a match that could still grow is not decided.
#[derive(Debug, Clone, Copy)]
pub(super) struct Text<'a> {
    pub chars: &'a [char],
    pub at_start: bool,
    pub open: bool,
}

/// The byte length and rule of the longest anchored match, the earliest rule on a tie, and
/// whether the automaton died before the end of `haystack`: then no longer haystack could
/// change the match.
fn longest(rules: &RuleSet, cache: &mut Cache, haystack: &str) -> (Option<(usize, usize)>, bool) {
    let dfa = &rules.dfa;
    let input = Input::new(haystack).anchored(Anchored::Yes);
    let mut sid = dfa.start_state_forward(cache, &input).expect("anchored start");
    let mut found = None;
    let mut dead = false;
    for (i, &b) in haystack.as_bytes().iter().enumerate() {
        sid = dfa.next_state(cache, sid, b).expect("the cache never gives up");
        if sid.is_tagged() {
            if sid.is_match() {
                found = Some((i, sid));
            } else if sid.is_dead() {
                dead = true;
                break;
            }
        }
    }
    if !dead {
        sid = dfa.next_eoi_state(cache, sid).expect("the cache never gives up");
        if sid.is_match() {
            found = Some((haystack.len(), sid));
        }
    }
    let found = found.map(|(len, sid)| {
        let rule = (0..dfa.match_len(cache, sid))
            .map(|k| dfa.match_pattern(cache, sid, k).as_usize())
            .min()
            .expect("a match state has a pattern");
        (len, rule)
    });
    (found, dead)
}

/// The input symbol for `c`: NUL is a space (as BlingFire replaces it), and the codepoints used
/// as anchors cannot be confused with them.
fn symbol(c: char) -> char {
    match c {
        '\0' => ' ',
        START | END | '\u{1}' | '\u{2}' => OTHER,
        c => c,
    }
}
