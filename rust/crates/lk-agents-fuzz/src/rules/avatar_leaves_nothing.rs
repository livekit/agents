//! Closing the avatar runner leaves nothing behind: no host frame held, no pull of the generator
//! after its media closed, the receiver closed, the synchronizer closed once, and nothing of the
//! runner's still running.
//!
//! A frame is the host's (`rtc.VideoFrame`, `rtc.AudioFrame`) and lives while the runner holds its
//! reference; one kept across an interruption or the close is a frame leaked per call.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{AvatarStep, Recording},
};

pub struct AvatarLeavesNothing;

impl Rule for AvatarLeavesNothing {
    const CONTRACT: Contract = Contract {
        id: "avatar_leaves_nothing",
        guarantee: "A closed avatar runner holds no frame, pulls nothing more, and has closed its receiver and its synchronizer once, with nothing left running.",
        examples: &[clean],
        counterexamples: &[a_frame_kept, still_pulling, media_left_open],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for (_, step) in r.avatar() {
            if let AvatarStep::Closed {
                frames_left,
                pulls_after_close,
                media_closes,
                receiver_closed,
                io_in_flight,
            } = step
            {
                let clean = *frames_left == 0
                    && *pulls_after_close == 0
                    && *media_closes == 1
                    && *receiver_closed
                    && !*io_in_flight;
                check.expect_with(Subject::Session, clean, || format!("closed with {step:?}"));
            }
        }
    }
}

fn closed(frames_left: usize, pulls_after_close: u32, media_closes: u32) -> Recording {
    Authored::new()
        .avatar_at(0, AvatarStep::CloseAsked)
        .avatar_at(
            1,
            AvatarStep::Closed {
                frames_left,
                pulls_after_close,
                media_closes,
                receiver_closed: true,
                io_in_flight: false,
            },
        )
        .done()
}

fn clean() -> Recording {
    closed(0, 0, 1)
}

fn a_frame_kept() -> Recording {
    closed(1, 0, 1)
}

fn still_pulling() -> Recording {
    closed(0, 2, 1)
}

fn media_left_open() -> Recording {
    closed(0, 0, 0)
}
