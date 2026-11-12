//! Mutation testing of the rules: which corpus windows notice a deleted or inverted rule, a
//! narrowed character set, or a word missing from a list. It chooses the CI corpus.

use std::io::Write as _;

use serde_json::{Value, json};

use super::lexer::Caches;
use super::rules::{Action, Patterns, Rule, RuleSets, Set, Sources, build};
use super::spans;

/// A rule set broken in one way, and the words a window must contain for the break to matter.
struct Mutant {
    name: String,
    sets: RuleSets,
    needs: Vec<String>,
}

fn compile(src: &Sources, edit: impl FnOnce(&mut Vec<(String, Rule)>)) -> RuleSets {
    let patterns = Patterns::new(src);
    let mut main = patterns.main();
    edit(&mut main);
    [build(main), patterns.last_dot(), patterns.mark_terminal()]
}

/// A character set or alternative that matches nothing.
const NOTHING: &str = "\u{FDD3}";

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn mutants() -> Vec<Box<dyn FnOnce() -> Mutant>> {
    let base = Sources::default();
    let rules = Patterns::new(&base).main().len();
    let mut out: Vec<Box<dyn FnOnce() -> Mutant>> = Vec::new();
    for i in 0..rules {
        out.push(Box::new(move || Mutant {
            name: format!("rule {i}: deleted"),
            sets: compile(&Sources::default(), |m| {
                m.remove(i);
            }),
            needs: Vec::new(),
        }));
        out.push(Box::new(move || Mutant {
            name: format!("rule {i}: inverted"),
            sets: compile(&Sources::default(), |m| {
                m[i].1.action = match m[i].1.action {
                    Action::NoBreak => Action::Call { set: Set::MarkTerminal, once: false },
                    _ => Action::NoBreak,
                };
            }),
            needs: Vec::new(),
        }));
    }
    let class = |name: &str, edit: fn(&mut Sources)| -> Box<dyn FnOnce() -> Mutant> {
        let name = name.to_owned();
        Box::new(move || {
            let mut src = Sources::default();
            edit(&mut src);
            Mutant { name, sets: compile(&src, |_| {}), needs: Vec::new() }
        })
    };
    out.push(class("no danda", |s| s.always = leak(s.always.replace(r"\x{964}\x{965}", ""))));
    out.push(class("no ideographic stop", |s| s.always = leak(s.always.replace(r"\x{3002}", ""))));
    out.push(class("no fullwidth ?!", |s| {
        s.always = leak(s.always.replace(r"\x{FF1F}\x{FF01}", ""))
    }));
    out.push(class("no Burmese/Armenian/Ethiopic stop", |s| {
        s.always = leak(s.always.replace(r"\x{104B}\x{589}\x{1362}", ""))
    }));
    out.push(class("no ellipsis terminator", |s| s.endp = leak(s.endp.replace(r"\x{2026}", ""))));
    out.push(class("no ! terminator", |s| s.endp = leak(s.endp.replacen('!', "", 1))));
    out.push(class("no Arabic/Urdu terminators", |s| {
        s.endp = leak(s.endp.replace(r"\x{6D4}\x{61F}", ""))
    }));
    out.push(class("no closers", |s| s.closers = NOTHING));
    out.push(class("no emoji", |s| s.emoji = NOTHING));
    out.push(class("capitals: uppercase only", |s| s.capital = r"[\p{Lu}]"));
    out.push(class("capitals: cased only", |s| s.capital = r"[\p{Lu}\p{Lt}]"));
    out.push(class("capitals: ASCII only", |s| s.capital = "[A-Z]"));

    // A word list with some entries left out; a window must contain one of them to notice.
    type Field = fn(&mut Sources) -> &mut Vec<&'static str>;
    let lists: [(&str, Field, usize); 6] = [
        ("abbreviation", |s| &mut s.abbreviations, 25),
        ("title", |s| &mut s.titles, 5),
        ("not-abbreviation", |s| &mut s.not_abbreviations, 1),
        ("start word", |s| &mut s.start_words, 8),
        ("ordinal noun", |s| &mut s.ordinal_nouns, 1),
        ("more abbreviation", |s| &mut s.abbreviations, 1),
    ];
    let more_from =
        base.abbreviations.len() - include_str!("data/more_abbreviations.txt").lines().count();
    for (kind, field, chunk) in lists {
        let mut probe = Sources::default();
        let all = field(&mut probe).clone();
        let range = match kind {
            "abbreviation" => 0..more_from,
            "more abbreviation" => more_from..all.len(),
            _ => 0..all.len(),
        };
        let indices: Vec<usize> = range.collect();
        for group in indices.chunks(chunk) {
            let gone: Vec<&'static str> = group.iter().map(|&i| all[i]).collect();
            out.push(Box::new(move || {
                let mut src = Sources::default();
                let list = field(&mut src);
                list.retain(|w| !gone.contains(w));
                if list.is_empty() {
                    list.push(NOTHING);
                }
                Mutant {
                    name: format!("{kind} missing: {}", gone.join(" ")),
                    sets: compile(&src, |_| {}),
                    needs: gone.iter().map(|w| (*w).to_owned()).collect(),
                }
            }));
        }
    }
    out
}

// Every mutant changes the split of at least one corpus window, so no rule, character set or
// list entry is untested.
#[test]
fn the_corpus_notices_every_mutant() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/sentences/corpus.jsonl");
    let texts: Vec<String> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap()["text"].as_str().unwrap().to_owned())
        .collect();
    let mut real = Caches::new(super::rules::rule_sets());
    let reference: Vec<_> = texts.iter().map(|t| spans(&mut real, t)).collect();
    let mut unnoticed = Vec::new();
    for make in mutants() {
        let mutant = make();
        let mut caches = Caches::new(&mutant.sets);
        let noticed = texts.iter().zip(&reference).any(|(text, want)| {
            (mutant.needs.is_empty() || mutant.needs.iter().any(|w| text.contains(w.as_str())))
                && &spans(&mut caches, text) != want
        });
        if !noticed {
            unnoticed.push(mutant.name);
        }
    }
    assert!(unnoticed.is_empty(), "no corpus window notices: {unnoticed:#?}");
}

// The experiment corpus's `windows.jsonl` (id, text, within, expect?) in, its `kills.jsonl` out:
// the current split of every window, then one line per mutant with the windows that notice it.
#[test]
#[ignore = "needs the experiment corpus"]
fn kill_matrix() {
    let windows: Vec<Value> =
        std::fs::read_to_string(super::sentences_tests::experiment_corpus().join("windows.jsonl"))
            .expect("windows.jsonl: sentence-corpus/tools/select_ci.sh cuts it before this runs")
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
    let mut out =
        std::fs::File::create(super::sentences_tests::experiment_corpus().join("kills.jsonl"))
            .unwrap();
    let texts: Vec<&str> = windows.iter().map(|w| w["text"].as_str().unwrap()).collect();

    let real_sets = super::rules::rule_sets();
    let mut real = Caches::new(real_sets);
    let mut decisions = Vec::new();
    let reference: Vec<Vec<(usize, usize)>> = texts
        .iter()
        .map(|t| {
            let got = spans(&mut real, t);
            decisions.push(std::mem::take(&mut real.matches));
            real.fired.clear();
            got
        })
        .collect();
    // A window cut from unlabelled text is used only if it splits as it did in its document.
    let usable: Vec<bool> = windows
        .iter()
        .zip(&reference)
        .map(|(w, got)| {
            if w.get("expect").is_some() {
                return true;
            }
            let chars: Vec<char> = w["text"].as_str().unwrap().chars().collect();
            let got: Vec<String> = got
                .iter()
                .map(|&(s, e)| chars[s..e].iter().collect::<String>().trim().to_owned())
                .collect();
            let within: Vec<String> = w["within"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap().trim().to_owned())
                .collect();
            got == within
        })
        .collect();
    for (i, (w, (got, ok))) in windows.iter().zip(reference.iter().zip(&usable)).enumerate() {
        if *ok {
            let line = json!({ "window": w["id"], "spans": got, "matches": decisions[i] });
            writeln!(out, "{line}").unwrap();
        }
    }
    let all = mutants();
    let total = all.len();
    for (k, make) in all.into_iter().enumerate() {
        let mutant = make();
        let mut caches = Caches::new(&mutant.sets);
        let mut kills = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            if !usable[i] {
                continue;
            }
            if !mutant.needs.is_empty() && !mutant.needs.iter().any(|w| text.contains(w.as_str())) {
                continue;
            }
            if spans(&mut caches, text) != reference[i] {
                kills.push(windows[i]["id"].clone());
            }
        }
        writeln!(out, "{}", json!({ "mutant": mutant.name, "kills": kills })).unwrap();
        eprintln!("{}/{total} {}: {} windows", k + 1, mutant.name, kills.len());
    }
}
