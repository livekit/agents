use super::replace_words;

/// Replacements delete whole words without cascading across chunks. https://github.com/livekit/agents/pull/7407 https://github.com/livekit/agents/pull/7471 https://github.com/livekit/agents/pull/6439
#[test]
fn streamed_replacements_match_a_single_pass_for_every_partition() {
    let replacements = vec![
        ("cat".into(), "dog".into()),
        ("dog".into(), "wolf".into()),
        ("erase".into(), "".into()),
    ];
    let text = "cat! scatter erase dog? cat.";
    let expected = "dog! scatter  wolf? dog.";
    assert_eq!(replace_words(text, replacements.clone(), false).unwrap().0, expected);
    for width in 1..=text.len() {
        let mut pending = String::new();
        let mut out = String::new();
        for chunk in text.as_bytes().chunks(width) {
            pending.push_str(std::str::from_utf8(chunk).unwrap());
            if let Some((updated, processed)) = replace_words(&pending, replacements.clone(), true)
            {
                out.extend(updated.chars().take(processed));
                pending = updated.chars().skip(processed).collect();
            }
        }
        out.push_str(&replace_words(&pending, replacements.clone(), false).unwrap().0);
        assert_eq!(out, expected, "chunk width {width}");
    }
}
