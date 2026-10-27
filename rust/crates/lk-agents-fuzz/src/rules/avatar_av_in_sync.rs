//! The avatar's audio and video stay in sync: each frame period the generator lip-syncs to its
//! audio goes out with that audio, and they play within one synchronizer queue plus one frame of
//! each other.
//!
//! The bound is `rtc.AVSynchronizer`'s. Audio plays as soon as it is pushed behind at most
//! `queue_size_ms` already queued; video waits in a queue of `fps * queue_size_ms` frames and
//! shows on the frame clock, so up to one queue plus one frame period late. A runner that drops,
//! holds back or reorders one side drifts past it within a few frames.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{AvatarFrameLabel, AvatarStep, Observed, Recording},
};

pub struct AvatarAvInSync;

impl Rule for AvatarAvInSync {
    const CONTRACT: Contract = Contract {
        id: "avatar_av_in_sync",
        guarantee: "Each avatar frame's video and its audio are both pushed, and play within one synchronizer queue plus one frame period of each other.",
        examples: &[in_step],
        counterexamples: &[video_late, video_dropped],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let Some((period, latency, queue)) = started(r) else { return };
        let bound = queue + period;
        let close = r
            .avatar()
            .iter()
            .find(|(_, s)| matches!(s, AvatarStep::CloseAsked))
            .map_or(Observed::END, |(at, _)| *at);
        // per tick: when its frames were yielded, and when each plays
        let mut ticks =
            std::collections::BTreeMap::<u64, [(Option<Observed>, Option<u64>); 2]>::new();
        for (at, step) in r.avatar() {
            match step {
                AvatarStep::Generated(AvatarFrameLabel {
                    video, tick, segment: Some(_), ..
                }) => {
                    ticks.entry(*tick).or_default()[usize::from(!*video)].0 = Some(*at);
                }
                AvatarStep::Pushed {
                    frame: AvatarFrameLabel { video, tick, segment: Some(_), .. },
                    plays_ms,
                } => {
                    ticks.entry(*tick).or_default()[usize::from(!*video)].1 = Some(*plays_ms);
                }
                _ => {}
            }
        }
        for (tick, [(video_at, video_plays), (audio_at, audio_plays)]) in ticks {
            let (Some(video_at), Some(audio_at)) = (video_at, audio_at) else { continue };
            // the close may cut what was yielded too late to be pushed
            if video_at.max(audio_at).at_ms + bound + latency >= close.at_ms {
                continue;
            }
            let ok = match (video_plays, audio_plays) {
                (Some(v), Some(a)) => v.abs_diff(a) <= bound,
                _ => false,
            };
            check.expect_with(Subject::Session, ok, || {
                format!("tick {tick}: video plays at {video_plays:?} ms, its audio at {audio_plays:?} ms (bound {bound} ms)")
            });
        }
    }
}

/// The generator's frame period, latency and the synchronizer's audio queue.
pub(crate) fn started(r: &Recording) -> Option<(u64, u64, u64)> {
    r.avatar().iter().find_map(|(_, s)| match s {
        AvatarStep::Started { period_ms, latency_ms, queue_ms } => {
            Some((*period_ms, *latency_ms, *queue_ms))
        }
        _ => None,
    })
}

fn frame(video: bool, tick: u64) -> AvatarFrameLabel {
    AvatarFrameLabel { video, tick, segment: Some(0), from: Some(Observed::at(2, 10)) }
}

fn tick(video_plays: Option<u64>, audio_plays: u64) -> Recording {
    let mut a = Authored::new()
        .avatar_at(0, AvatarStep::Started { period_ms: 40, latency_ms: 10, queue_ms: 100 })
        .avatar_at(10, AvatarStep::Received { segment: 0 })
        .avatar_at(50, AvatarStep::Generated(frame(true, 1)))
        .avatar_at(50, AvatarStep::Generated(frame(false, 1)));
    if let Some(plays_ms) = video_plays {
        a = a.avatar_at(50, AvatarStep::Pushed { frame: frame(true, 1), plays_ms });
    }
    a.avatar_at(50, AvatarStep::Pushed { frame: frame(false, 1), plays_ms: audio_plays })
        .avatar_at(1000, AvatarStep::CloseAsked)
        .done()
}

fn in_step() -> Recording {
    tick(Some(80), 50)
}

/// The video of a frame shows 300 ms after its audio.
fn video_late() -> Recording {
    tick(Some(350), 50)
}

/// The audio went out, its video never did.
fn video_dropped() -> Recording {
    tick(None, 50)
}
