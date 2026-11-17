use super::*;
use lk_agents_abi::session::InputRequestRef;

fn candidates() -> Vec<InputCandidate> {
    ["first", "last"]
        .into_iter()
        .map(|id| InputCandidate {
            reference: InputRequestRef::Agent { agent_id: "desk".into(), request_id: id.into() },
            question: format!("Your {id} name?"),
            presentation: ChatMessage::text(
                id,
                lk_agents_abi::Role::Assistant,
                "Your first and last names?",
                std::time::UNIX_EPOCH,
            ),
        })
        .collect()
}

#[test]
fn verdicts_can_select_only_distinct_offered_indices_or_request_clarification() {
    let offered = candidates();
    assert_eq!(
        parse(r#"{"candidates":[1,0],"unclear":false}"#, &offered).unwrap(),
        InputRoute::Matched(vec![offered[1].reference.clone(), offered[0].reference.clone()])
    );
    assert_eq!(
        parse(r#"{"candidates":[],"unclear":false}"#, &offered).unwrap(),
        InputRoute::NoMatch
    );
    assert_eq!(
        parse(r#"{"candidates":[],"unclear":true}"#, &offered).unwrap(),
        InputRoute::Unclear
    );
    for invalid in [
        r#"{"candidates":[0,0],"unclear":false}"#,
        r#"{"candidates":[2],"unclear":false}"#,
        r#"{"candidates":[-1],"unclear":false}"#,
        r#"{"candidates":["0"],"unclear":false}"#,
        r#"{"candidates":[0],"unclear":true}"#,
        r#"{"candidates":[],"unclear":"false"}"#,
        r#"{"candidates":[],"unclear":false,"approve":true}"#,
    ] {
        assert!(parse(invalid, &offered).is_err(), "{invalid}");
    }
}
