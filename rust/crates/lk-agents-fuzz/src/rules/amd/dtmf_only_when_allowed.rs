use lk_agents_voice::amd::SEND_DTMF_TOOL;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input, Ref},
    record::{Kind, Recording, amd::Answer},
};

// Spec §14 invariant 9, the AMD proposal "Reply payload: Tools": AMD's built-in
// `send_dtmf_events` is offered once in a reply whose decision set `dtmf`, and in no other
// request, a preemptive generation included. An agent that defines its own tool with that id
// keeps it in every request, so the tool appearing outside a DTMF reply is judged only when no
// request made outside one already carried it.

pub struct DtmfOnlyWhenAllowed;

impl Rule for DtmfOnlyWhenAllowed {
    const CONTRACT: Contract = Contract {
        id: "dtmf_only_when_allowed",
        guarantee: "AMD's DTMF tool is offered exactly once in a reply AMD allowed it for, and \
                    nowhere else.",
        examples: &[offered_to_the_ivr_reply],
        counterexamples: &[missing_from_the_ivr_reply, offered_twice, offered_to_a_plain_reply],
        probe: Some(an_ivr_menu),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        let mut ivr_requests = std::collections::BTreeSet::new();
        for turn in amd.gated_turns() {
            let Some(Answer::Decide { at, verdict: AmdVerdict::Ivr, .. }) = amd.deciding(&turn)
            else {
                continue;
            };
            for request in amd.requests_answering(&turn) {
                if amd.preemptive(&request) || request.opened < at {
                    continue;
                }
                ivr_requests.insert(request.chat);
                let offered = request.tools.iter().filter(|t| *t == SEND_DTMF_TOOL).count();
                check.expect_with(Subject::Resource(Kind::Chat, request.chat), offered == 1, || {
                    format!("request {} of an IVR reply offered the DTMF tool {offered} times", request.chat)
                });
            }
        }
        let carrying: Vec<_> = amd
            .requests()
            .into_iter()
            .filter(|q| q.tools.iter().any(|t| t == SEND_DTMF_TOOL))
            .collect();
        let agents_own = carrying.iter().any(|q| !ivr_requests.contains(&q.chat))
            && carrying.iter().any(|q| amd.attached_at(q.opened).is_none());
        if agents_own {
            return;
        }
        for request in carrying.iter().filter(|q| !ivr_requests.contains(&q.chat)) {
            check.expect_with(Subject::Resource(Kind::Chat, request.chat), false, || {
                format!("request {} offered AMD's DTMF tool outside an IVR reply", request.chat)
            });
        }
    }
}

fn ivr() -> Authored {
    Authored::new()
        .sent(Input::AmdAttach {})
        .user_turn(Some(1), 1, "press one for sales")
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Ivr })
}

fn offered_to_the_ivr_reply() -> Recording {
    ivr().request(10, 1, "press one for sales", &[], &[SEND_DTMF_TOOL]).done()
}

fn missing_from_the_ivr_reply() -> Recording {
    ivr().request(10, 1, "press one for sales", &[], &[]).done()
}

fn offered_twice() -> Recording {
    ivr().request(10, 1, "press one for sales", &[], &[SEND_DTMF_TOOL, SEND_DTMF_TOOL]).done()
}

fn offered_to_a_plain_reply() -> Recording {
    Authored::new()
        .sent(Input::AmdAttach {})
        .user_turn(Some(1), 1, "hello this is Sam")
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow })
        .request(10, 1, "hello this is Sam", &[], &[SEND_DTMF_TOOL])
        .done()
}

/// An IVR menu's turn is answered with the DTMF tool.
fn an_ivr_menu() -> crate::drive::Case {
    crate::fakes::amd::probe::one_turn("press one for appointments", AmdVerdict::Ivr)
}
