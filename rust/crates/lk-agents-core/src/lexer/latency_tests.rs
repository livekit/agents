//! The lexer must not slow the voice: prose is never held, a construct is held only until it
//! resolves, and the work per delta is linear in the delta.

use super::*;
use crate::tts::markup::Provider;

fn speech() -> Rules {
    Rules {
        model: Some(ModelSyntax::DEFAULT),
        ..Rules::speech(Vocabulary::of(Some(Provider::Cartesia)))
    }
}

/// A typical expressive reply, as an LLM streams it: a few characters per delta.
const REPLY: &str = "<expr type=\"expression\" label=\"warm\"/> Sure! Here's the **short** version:\n\
- Your order *ships* tomorrow, 3 < 5 days.\n\
- Track it at [the portal](https://x.y/track).\n\
Anything else? 😊";

fn deltas(text: &str, size: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < text.len() {
        let mut end = (at + size).min(text.len());
        while !text.is_char_boundary(end) {
            end += 1;
        }
        out.push(&text[at..end]);
        at = end;
    }
    out
}

#[test]
fn prose_is_held_at_most_one_character_and_only_at_a_special_one() {
    let prose =
        "Sure, your order ships tomorrow. It's 3 < 5 days, 50% off, a-b, x > y, 2*3*4, snake_case.";
    let mut lexer = Lexer::new(speech());
    for d in deltas(prose, 1) {
        lexer.push(d);
        let special = d.starts_with(['<', '*', '_', '~', '[', '!', '`']);
        assert!(lexer.held() <= usize::from(special), "held {} after {d:?}", lexer.held());
    }
}

#[test]
fn a_construct_is_held_only_until_it_resolves() {
    let mut lexer = Lexer::new(speech());
    let mut max = 0;
    for d in deltas(REPLY, 4) {
        lexer.push(d);
        max = max.max(lexer.held());
    }
    // the longest construct here is the opening marker
    let marker = "<expr type=\"expression\" label=\"warm\"/>";
    assert!(max < marker.len(), "held up to {max} bytes");
    lexer.finish();
    // and nothing is held between constructs
    let mut lexer = Lexer::new(speech());
    lexer.push("**short** and ");
    assert_eq!(lexer.held(), 0);
}

#[test]
fn spoken_words_leave_with_the_delta_that_completes_them() {
    let mut lexer = Lexer::new(speech());
    let mut said = Projector::new(View::Spoken { keep_markdown: false, keep_emoji: false });
    assert_eq!(said.text(&lexer.push("Sure! Here's ")), "Sure! Here's ");
    assert_eq!(said.text(&lexer.push("the **sho")), "the ");
    assert_eq!(said.text(&lexer.push("rt** version")), "short version");
}

/// `cargo test --release -p lk-agents-core --lib lexer::latency_tests::throughput -- --ignored --nocapture`
#[test]
#[ignore = "a measurement, run in release"]
#[allow(clippy::disallowed_methods, reason = "a measurement reads the clock; it decides nothing")]
fn throughput() {
    let text = REPLY.repeat(500);
    for size in [1, 4, 16] {
        let parts = deltas(&text, size);
        let start = std::time::Instant::now();
        let mut lexer = Lexer::new(speech());
        let mut said = Projector::new(View::Spoken { keep_markdown: false, keep_emoji: false });
        let mut sync = Units::new(
            View::Spoken { keep_markdown: false, keep_emoji: false },
            View::Display { keep_emoji: true },
        );
        let mut n = 0;
        for d in &parts {
            let tokens = lexer.push(d);
            n += said.text(&tokens).len();
            n += sync.push(&tokens).len();
        }
        let elapsed = start.elapsed();
        println!(
            "{size:>2}-byte deltas: {:.0} ns/delta, {:.1} MB/s ({n})",
            elapsed.as_nanos() as f64 / parts.len() as f64,
            text.len() as f64 / elapsed.as_secs_f64() / 1e6
        );
    }
}

#[test]
fn prose_around_near_misses_settles_as_one_token() {
    let text = "It's 3 < 5 days, 5 * 3, a_b, x~y, [a] and C# 50%.";
    let tokens = Lexer::all(speech(), text);
    assert_eq!(
        tokens,
        [Token { kind: Kind::Prose, raw: text.to_owned(), ctx: Context::default() }]
    );
}
