//! The realtime server never refuses a request the driver should have known it would refuse.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Breach, Kind, Recording},
};

pub struct RealtimeServerAcceptsRequests;

impl Rule for RealtimeServerAcceptsRequests {
    const CONTRACT: Contract = Contract {
        id: "realtime_server_accepts_requests",
        guarantee: "The realtime server (fakes/realtime/server.rs, refusing what the real one refuses) refuses no request the driver had already been told was invalid: no cancel of a response it heard was done, no reply asked over one it heard was created, no truncation past the audio it was sent. A refusal racing an event still on its way is the protocol's, not the driver's.",
        examples: &[raced_cancel],
        counterexamples: &[knowing_cancel, duplicate_item],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for (at, socket, breach) in r.inference_view().realtime_breaches() {
            if let Breach::Refused { event, code, why, knowing } = breach {
                check.expect(
                    Subject::Resource(Kind::Realtime, *socket),
                    !knowing,
                    format!("at {} ms the server refused {event} ({code}: {why}), which the client had been told it would", at.at_ms),
                );
            }
        }
    }
}

fn refused(event: &str, code: &'static str, knowing: bool) -> Breach {
    Breach::Refused { event: event.into(), code, why: "", knowing }
}

/// A cancel that crossed the response's `response.done` on the wire.
fn raced_cancel() -> Recording {
    Authored::new().breach(0, refused("response.cancel", "response_cancel_not_active", false)).done()
}

/// A cancel sent after the client had been told the response was done.
fn knowing_cancel() -> Recording {
    Authored::new().breach(0, refused("response.cancel", "response_cancel_not_active", true)).done()
}

/// An item created again under an id the conversation holds.
fn duplicate_item() -> Recording {
    let breach = refused("conversation.item.create", "item_create_duplicate_item_id", true);
    Authored::new().breach(0, breach).done()
}
