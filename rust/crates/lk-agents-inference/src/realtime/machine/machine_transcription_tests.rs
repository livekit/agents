use super::*;
use crate::realtime::{
    XAI,
    protocol::{Incoming, TranscriptionCompleted},
};

#[derive(Debug, Clone, Copy)]
enum End {
    Failed,
    Reconnect,
    Close,
}

fn item() -> ChatItem {
    ChatItem::Message(lk_agents_abi::ChatMessage::new(
        "user",
        lk_agents_abi::Role::User,
        vec![],
        std::time::SystemTime::UNIX_EPOCH,
    ))
}

/// xAI sends cumulative revisions, not deltas. A failed ASR job or lost connection must settle
/// the latest correction once, rather than concatenating versions or dropping all received text.
#[test]
fn xai_cumulative_corrections_survive_asr_failure_reconnect_and_close() {
    for end in [End::Failed, End::Reconnect, End::Close] {
        let mut m = crate::realtime::testfix::machine(XAI);
        m.open(at(0));
        let generation = m.connection_generation();
        m.incoming_on(
            generation,
            at(0),
            Incoming::ItemAdded { previous_item_id: None, item: item() },
        );
        for text in ["book for Monday", "book for Tuesday"] {
            m.incoming_on(
                generation,
                at(1),
                Incoming::TranscriptionUpdated { item_id: "user".into(), transcript: text.into() },
            );
        }
        let step = match end {
            End::Failed => m.incoming_on(
                generation,
                at(2),
                Incoming::TranscriptionFailed { item_id: "user".into(), content_index: 0 },
            ),
            End::Reconnect => m.reconnect(at(2)),
            End::Close => m.close(at(2)),
        };
        let finals: Vec<_> = step
            .events
            .iter()
            .filter_map(|event| match event {
                Event::InputTranscription(t) if t.is_final => {
                    Some((t.item_id.as_str(), t.transcript.as_str()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(finals, [("user", "book for Tuesday")], "{end:?}");
        if matches!(end, End::Failed) {
            assert_eq!(
                m.chat_ctx().items[0].as_message().unwrap().raw_text_content().as_deref(),
                Some("book for Tuesday")
            );
        }
    }
}

/// Some compatible providers emit completed envelopes marked in_progress. They are still
/// revisable; if the connection ends before a settled result, the last received words survive.
#[test]
fn an_in_progress_completed_envelope_is_retained_when_the_connection_ends() {
    let mut m = crate::realtime::testfix::machine(XAI);
    m.open(at(0));
    let generation = m.connection_generation();
    m.incoming_on(generation, at(0), Incoming::ItemAdded { previous_item_id: None, item: item() });
    m.incoming_on(
        generation,
        at(1),
        Incoming::TranscriptionCompleted(TranscriptionCompleted {
            item_id: "user".into(),
            content_index: 0,
            transcript: "Tuesday".into(),
            confidence: None,
            in_progress: true,
            event_id: "partial".into(),
            usage: None,
        }),
    );
    let step = m.reconnect(at(2));
    assert!(step.events.iter().any(|e| matches!(e, Event::InputTranscription(t) if t.is_final && t.item_id == "user" && t.transcript == "Tuesday")));
}

#[test]
fn failing_a_held_xai_final_does_not_finalize_it_again_on_close() {
    let mut m = crate::realtime::testfix::machine(XAI);
    m.open(at(0));
    let generation = m.connection_generation();
    m.incoming_on(generation, at(0), Incoming::ItemAdded { previous_item_id: None, item: item() });
    m.incoming_on(
        generation,
        at(1),
        Incoming::TranscriptionCompleted(TranscriptionCompleted {
            item_id: "user".into(),
            content_index: 0,
            transcript: "Tuesday".into(),
            confidence: None,
            in_progress: false,
            event_id: "settled".into(),
            usage: None,
        }),
    );
    let failed = m.incoming_on(
        generation,
        at(2),
        Incoming::TranscriptionFailed { item_id: "user".into(), content_index: 0 },
    );
    assert_eq!(
        failed
            .events
            .iter()
            .filter(|e| matches!(e, Event::InputTranscription(t) if t.is_final))
            .count(),
        1
    );
    let closed = m.close(at(3));
    assert!(!closed.events.iter().any(|e| matches!(e, Event::InputTranscription(t) if t.is_final)));
}

/// The user's ASR job can finish after all assistant output. There may be no next output or
/// speech onset to flush it, so a settled late completion must reach history immediately.
#[test]
fn a_settled_xai_transcript_arriving_after_the_answer_is_not_held_until_another_turn() {
    let mut m = crate::realtime::testfix::machine(XAI);
    m.open(at(0));
    let generation = m.connection_generation();
    for event in [
        Incoming::ItemAdded { previous_item_id: None, item: item() },
        Incoming::ResponseCreated { response_id: "reply".into(), client_event_id: None },
        Incoming::OutputItemAdded {
            item_id: "answer".into(),
            kind: crate::realtime::protocol::OutputItemKind::Message,
            response_id: Some("reply".into()),
        },
        Incoming::AudioTranscriptDelta {
            item_id: "answer".into(),
            delta: "Tuesday".into(),
            start_time: None,
        },
        Incoming::AudioDelta {
            item_id: "answer".into(),
            frame: lk_agents_abi::AudioFrame::new(vec![0; 480], 24_000, 1, 240),
        },
        Incoming::ResponseDone(crate::realtime::protocol::ResponseDone {
            response_id: "reply".into(),
            status: GenerationStatus::Completed,
            error_type: None,
            error_code: None,
            usage: Default::default(),
        }),
    ] {
        m.incoming_on(generation, at(1), event);
    }
    let step = m.incoming_on(
        generation,
        at(10),
        Incoming::TranscriptionCompleted(TranscriptionCompleted {
            item_id: "user".into(),
            content_index: 0,
            transcript: "book for Tuesday".into(),
            confidence: None,
            in_progress: false,
            event_id: "late".into(),
            usage: None,
        }),
    );
    assert!(step.events.iter().any(|e| matches!(e, Event::InputTranscription(t) if t.is_final && t.item_id == "user" && t.transcript == "book for Tuesday")));
    assert!(
        !m.close(at(11))
            .events
            .iter()
            .any(|e| matches!(e, Event::InputTranscription(t) if t.is_final))
    );
}

// Catches: a cancelled response's message announced after the next response was created played
// as the new reply's. The cancelled reply still streams what it generated; when its
// `response.output_item.added` lands after the next `response.created`, only the response id it
// names tells it apart, and the session truncated both messages as the new reply's: one past the
// audio the server holds for it (DST seed 5025).
#[test]
fn a_cancelled_response_message_announced_late_is_not_the_next_replys() {
    use crate::realtime::{dialect::OPENAI, protocol::OutputItemKind};
    let mut m = crate::realtime::testfix::machine(OPENAI);
    m.open(at(0));
    let generation = m.connection_generation();
    let added = |item: &str, response: &str| Incoming::OutputItemAdded {
        item_id: item.into(),
        kind: OutputItemKind::Message,
        response_id: Some(response.into()),
    };
    let said = |item: &str| Incoming::AudioTranscriptDelta {
        item_id: item.into(),
        delta: "words ".into(),
        start_time: None,
    };
    let message = |id: &str| {
        ChatItem::Message(lk_agents_abi::ChatMessage::new(
            id,
            lk_agents_abi::Role::Assistant,
            vec![],
            std::time::SystemTime::UNIX_EPOCH,
        ))
    };
    let mut events = Vec::new();
    for event in [
        Incoming::ResponseCreated { response_id: "cancelled".into(), client_event_id: None },
        Incoming::ResponseCreated { response_id: "next".into(), client_event_id: None },
        added("tail", "cancelled"),
        said("tail"),
        added("answer", "next"),
        said("answer"),
        Incoming::OutputItemDone { item: message("tail") },
        Incoming::OutputItemDone { item: message("answer") },
    ] {
        events.extend(m.incoming_on(generation, at(1), event).events);
    }
    let told = |id: &str| {
        events.iter().any(|e| match e {
            Event::MessageStarted(s) => s.message_id == id,
            Event::TextDelta(d) => d.message_id == id,
            Event::MessageDone(d) => d.message_id == id,
            _ => false,
        })
    };
    assert!(told("answer"), "the next reply's message is told: {events:?}");
    assert!(!told("tail"), "the cancelled reply's message is nobody's: {events:?}");
}
