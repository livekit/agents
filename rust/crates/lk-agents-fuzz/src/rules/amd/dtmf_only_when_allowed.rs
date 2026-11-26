use crate::fakes::amd::SEND_DTMF_TOOL;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input, Ref},
    record::{Kind, Recording, amd::Answer},
};

// Spec §14 invariant 9: AMD's built-in
// `send_dtmf_events` is offered once in a reply whose decision set `dtmf`, and in no other
// request, a preemptive generation included. An agent that defines its own tool with that id
// keeps it in every request, so the tool appearing outside a DTMF reply is judged only when no
// request made outside one already carried it. A realtime response takes the session's tools, so
// the session sets them to the decision's tools for the response and restores them after
// (proposal, Realtime): every response dispatched for a DTMF reply's speech finds the tool among
// the tools the server last received, every other response does not, and once every DTMF reply's
// speech ended the live socket's last tools lack it.

pub struct DtmfOnlyWhenAllowed;

impl Rule for DtmfOnlyWhenAllowed {
    const CONTRACT: Contract = Contract {
        id: "dtmf_only_when_allowed",
        guarantee: "AMD's DTMF tool is offered exactly once in a reply AMD allowed it for, and \
                    nowhere else.",
        examples: &[offered_to_the_ivr_reply, realtime_tools_set_and_restored],
        counterexamples: &[
            missing_from_the_ivr_reply,
            offered_twice,
            offered_to_a_plain_reply,
            realtime_tools_never_set,
            realtime_tools_never_restored,
            realtime_tools_left_after_the_reply,
        ],
        probe: Some(an_ivr_menu),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        let mut ivr_requests = std::collections::BTreeSet::new();
        for turn in amd.gated_turns() {
            let Some(Answer::Decide { at, verdict, .. }) = amd.deciding(&turn) else {
                continue;
            };
            if !verdict.offers_dtmf() {
                continue;
            }
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
        let agents_own_tool = amd.agent_declared_at(SEND_DTMF_TOOL).is_some();
        let mut dtmf_responses = std::collections::BTreeSet::new();
        let mut dtmf_speeches = Vec::new();
        for turn in amd.gated_turns() {
            let Some(deciding) = amd.deciding(&turn) else { continue };
            if !amd.decision(&deciding).dtmf {
                continue;
            }
            dtmf_speeches.extend(amd.rt_reply_speeches(&turn));
            for response in amd.rt_responses_for(&turn) {
                dtmf_responses.insert(response.req_id.clone());
                let offered = (response.tools.iter().flatten())
                    .filter(|t| *t == SEND_DTMF_TOOL)
                    .count();
                check.expect_with(Subject::Session, offered == 1, || {
                    format!(
                        "realtime response {} of turn {}'s DTMF reply found the DTMF tool {offered} \
                         times among the session's tools ({:?})",
                        response.req_id, turn.turn, response.tools
                    )
                });
            }
        }
        if !agents_own_tool {
            restored(r, &dtmf_speeches, check);
            for response in amd.rt_responses() {
                if dtmf_responses.contains(&response.req_id) {
                    continue;
                }
                check.expect_with(Subject::Session, !response.offers(SEND_DTMF_TOOL), || {
                    format!(
                        "realtime response {} at {} ms found AMD's DTMF tool among the session's \
                         tools outside a DTMF reply",
                        response.req_id, response.at.at_ms
                    )
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

/// Once every DTMF reply's speech ended before the close, the last tools the live socket
/// received lack AMD's tool, after detach too. A socket a fatal error or a lost connection ended
/// holds nothing to restore.
fn restored(r: &Recording, dtmf_speeches: &[u64], check: &mut Check) {
    use crate::input::RtEvent;
    use crate::record::RtSent;
    if dtmf_speeches.is_empty() {
        return;
    }
    let close = r.close_begun().unwrap_or(crate::record::Observed::END);
    let ends: Vec<_> = (dtmf_speeches.iter())
        .map(|s| {
            let speech = r.host_view().speech(*s);
            speech.done_announcements().first().map(|(at, _)| *at).or(speech.done_at())
        })
        .collect();
    let Some(last_end) = ends.iter().copied().collect::<Option<Vec<_>>>().and_then(|e| e.into_iter().max())
    else {
        return;
    };
    let lost = r.applied().iter().any(|a| {
        a.accepted
            && matches!(
                a.input,
                Input::RealtimeEvent { event: RtEvent::Disconnected | RtEvent::FatalError, .. }
            )
    });
    if last_end >= close || lost {
        return;
    }
    let sent = r.rt_sent();
    let Some(socket) = sent.iter().map(|(_, socket, _)| *socket).max() else { return };
    let last = sent.iter().rev().find_map(|(_, s, sent)| match sent {
        RtSent::Tools(tools) if *s == socket => Some(tools),
        _ => None,
    });
    check.expect_with(Subject::Session, last.is_none_or(|t| !t.iter().any(|n| n == SEND_DTMF_TOOL)), || {
        format!(
            "every DTMF reply's speech ended by {} ms, and socket {socket}'s last tools still \
             offer AMD's DTMF tool: {last:?}",
            last_end.at_ms
        )
    });
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

fn realtime_ivr() -> Authored {
    use crate::record::RtSent;
    Authored::new()
        .rt_sent(RtSent::Tools(vec!["lookup".into()]))
        .sent(Input::AmdAttach {})
        .user_turn(Some(1), 1, "press one for sales")
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Ivr })
}

fn realtime_tools_set_and_restored() -> Recording {
    use crate::record::RtSent;
    realtime_ivr()
        .reply_created(1, 1, 7)
        .rt_sent(RtSent::Tools(vec!["lookup".into(), SEND_DTMF_TOOL.into()]))
        .rt_request("rt_1", 7, Some(crate::fakes::amd::STAGE))
        .later(2_000)
        .rt_sent(RtSent::Tools(vec!["lookup".into()]))
        .speech_done(7, lk_agents_voice::events::SpeechOutcome::Completed)
        .user_turn(Some(1), 2, "hello this is Sam")
        .sent(Input::AmdDecide { turn: Ref::new(1), verdict: AmdVerdict::Allow })
        .reply_created(1, 2, 8)
        .rt_request("rt_2", 8, None)
        .done()
}

/// `SpeechSpec.toolset` alone does not reach the server.
fn realtime_tools_never_set() -> Recording {
    realtime_ivr()
        .reply_created(1, 1, 7)
        .rt_request("rt_1", 7, Some(crate::fakes::amd::STAGE))
        .done()
}

fn realtime_tools_never_restored() -> Recording {
    use crate::record::RtSent;
    realtime_ivr()
        .reply_created(1, 1, 7)
        .rt_sent(RtSent::Tools(vec!["lookup".into(), SEND_DTMF_TOOL.into()]))
        .rt_request("rt_1", 7, Some(crate::fakes::amd::STAGE))
        .later(2_000)
        .speech_done(7, lk_agents_voice::events::SpeechOutcome::Completed)
        .user_turn(Some(1), 2, "hello this is Sam")
        .sent(Input::AmdDecide { turn: Ref::new(1), verdict: AmdVerdict::Allow })
        .reply_created(1, 2, 8)
        .rt_request("rt_2", 8, None)
        .done()
}

/// The DTMF reply ended and no other response followed: the server keeps the keypad.
fn realtime_tools_left_after_the_reply() -> Recording {
    use crate::record::RtSent;
    realtime_ivr()
        .reply_created(1, 1, 7)
        .rt_sent(RtSent::Tools(vec!["lookup".into(), SEND_DTMF_TOOL.into()]))
        .rt_request("rt_1", 7, Some(crate::fakes::amd::STAGE))
        .later(2_000)
        .speech_done(7, lk_agents_voice::events::SpeechOutcome::Completed)
        .later(2_000)
        .done()
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
