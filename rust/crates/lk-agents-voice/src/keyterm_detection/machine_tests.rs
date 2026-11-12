use super::machine::{KeytermState, parse_record_keyterms};
use serde_json::json;

/// Keyterm tool fields that are not lists of strings leave the keyterms unchanged.
/// https://github.com/livekit/agents/pull/7468
#[test]
fn malformed_keyterm_lists_cannot_turn_strings_into_terms() {
    for malformed in [json!(null), json!("Acme"), json!({"Acme": true}), json!(7)] {
        let mut state = KeytermState::new(&["static".into()], None);
        state.apply(&parse_record_keyterms(Some(&json!({"confirm": ["known"]}))));
        let before = state.keyterms();
        let args = json!({"pending": malformed, "confirm": malformed, "remove": malformed});
        assert!(state.apply(&parse_record_keyterms(Some(&args))).is_none());
        assert_eq!(state.keyterms(), before);
    }
    let mut state = KeytermState::new(&["static".into()], None);
    let args = json!({"confirm": ["Acme", null, 7, "  ", "Acme"], "remove": null});
    assert_eq!(
        state.apply(&parse_record_keyterms(Some(&args))).unwrap().keyterms,
        ["static", "Acme"]
    );
}
