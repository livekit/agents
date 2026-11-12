use super::*;

// Catches: the first sentence held, a later batch sent before the audio played out, or a batch
// not joined up to the text length.
#[test]
fn first_sentence_is_immediate_then_wait_for_audio_and_batch() {
    let mut p = SentencePacer::new(5.0, 4, None);
    p.push("hello".into());
    p.push("你好".into());
    p.push("ab".into());
    p.end();
    assert_eq!(p.poll(100.0, 0.0).text.as_deref(), Some("hello"));
    assert_eq!(p.poll(101.0, 10.0).text, None);
    assert_eq!(p.poll(102.0, 10.0).text, None);
    assert_eq!(p.poll(106.0, 10.0).text.as_deref(), Some("你好 ab"));
    assert_eq!(p.poll(107.0, 10.0).delay, None);
}

// Catches: a paced batch joined past the provider's request limit.
#[test]
fn a_batch_is_never_joined_past_the_provider_limit() {
    let mut p = SentencePacer::new(5.0, 1000, Some(25));
    for sentence in ["First one.", "Second one.", "Third one.", "Fourth one."] {
        p.push(sentence.to_owned());
    }
    p.end();
    let first = p.poll(0.0, 0.0).text.expect("the first sentence goes at once");
    assert_eq!(first, "First one.");
    // the audio played out: the rest go, batched within the limit
    let mut batches = Vec::new();
    let mut now = 0.0;
    while p.holds_any() {
        now += 10.0;
        p.poll(now, 0.0);
        now += 10.0;
        if let Some(text) = p.poll(now, 0.0).text {
            batches.push(text);
        }
    }
    assert_eq!(batches, ["Second one. Third one.", "Fourth one."]);
    assert!(batches.iter().all(|b| b.chars().count() <= 25));
}
