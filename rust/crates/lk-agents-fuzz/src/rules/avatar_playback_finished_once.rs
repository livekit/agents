//! The avatar reports each segment's playback finished once: when the generator yields its end,
//! or when a clear cuts it. Never a report for a segment that did not end, and never one missing
//! for a segment that did, unless the close came too soon after for the report to go out.
//!
//! A segment is a run of audio the runner received, closed by the receiver's segment end or by a
//! clear. The agent's speech waits on this report (`DataStreamAudioOutput`): one missing holds it
//! until the clear timeout, or for ever without one; one too many ends the next segment early.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{AvatarStep, Observed, Recording},
    rules::avatar_av_in_sync::started,
};

pub struct AvatarPlaybackFinishedOnce;

impl Rule for AvatarPlaybackFinishedOnce {
    const CONTRACT: Contract = Contract {
        id: "avatar_playback_finished_once",
        guarantee: "Each avatar segment that ends or is cut gets exactly one playback-finished report.",
        examples: &[one_each, both_cut_segments_reported],
        counterexamples: &[twice, never, one_cut_segment_unreported, cut_segment_reported_twice],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let Some((period, latency, _)) = started(r) else { return };
        let close = r
            .avatar()
            .iter()
            .find(|(_, s)| matches!(s, AvatarStep::CloseAsked))
            .map_or(Observed::END, |(at, _)| *at);
        // segments received and not yet due; when each that is due became so
        let mut open = std::collections::BTreeSet::new();
        let mut due: Vec<Observed> = Vec::new();
        let mut reports = 0usize;
        for (at, step) in r.avatar() {
            match step {
                AvatarStep::Received { segment } => {
                    open.insert(*segment);
                }
                AvatarStep::GeneratedEnd { segment } => {
                    if open.remove(segment) {
                        due.push(*at);
                    }
                }
                AvatarStep::Clear => {
                    // a clear cuts what is playing: everything received and not yet ended
                    due.extend(open.iter().map(|_| *at));
                    open.clear();
                }
                AvatarStep::ReportedFinished { .. } => {
                    reports += 1;
                    check.expect_with(Subject::Session, reports <= due.len(), || {
                        format!(
                            "report {reports} at {} ms, with {} segments ended",
                            at.at_ms,
                            due.len()
                        )
                    });
                }
                _ => {}
            }
        }
        // what ended long enough before the close must have been reported
        let owed = due.iter().filter(|at| at.at_ms + period + latency < close.at_ms).count();
        if !due.is_empty() {
            check.expect_with(Subject::Session, reports >= owed, || {
                format!("{reports} playback-finished reports for {owed} segments that ended")
            });
        }
    }
}

fn steps(finished: usize) -> Recording {
    let mut a = Authored::new()
        .avatar_at(0, AvatarStep::Started { period_ms: 40, latency_ms: 10, queue_ms: 100 })
        .avatar_at(10, AvatarStep::Received { segment: 0 })
        .avatar_at(200, AvatarStep::GeneratedEnd { segment: 0 });
    for _ in 0..finished {
        a = a.avatar_at(200, AvatarStep::ReportedFinished { interrupted: false });
    }
    a.avatar_at(300, AvatarStep::Received { segment: 1 })
        .avatar_at(400, AvatarStep::Clear)
        .avatar_at(400, AvatarStep::ReportedFinished { interrupted: true })
        .avatar_at(2000, AvatarStep::CloseAsked)
        .done()
}

fn one_each() -> Recording {
    steps(1)
}

fn twice() -> Recording {
    steps(2)
}

fn never() -> Recording {
    steps(0)
}

fn cut_segments(reports: usize) -> Recording {
    let mut a = Authored::new()
        .avatar_at(0, AvatarStep::Started { period_ms: 40, latency_ms: 10, queue_ms: 100 })
        .avatar_at(10, AvatarStep::Received { segment: 0 })
        .avatar_at(20, AvatarStep::ReceivedEnd)
        .avatar_at(30, AvatarStep::Received { segment: 1 })
        .avatar_at(40, AvatarStep::ReceivedEnd)
        .avatar_at(50, AvatarStep::Clear);
    for _ in 0..reports {
        a = a.avatar_at(50, AvatarStep::ReportedFinished { interrupted: true });
    }
    a.avatar_at(2000, AvatarStep::CloseAsked).done()
}

fn both_cut_segments_reported() -> Recording {
    cut_segments(2)
}

fn one_cut_segment_unreported() -> Recording {
    cut_segments(1)
}

fn cut_segment_reported_twice() -> Recording {
    cut_segments(3)
}
