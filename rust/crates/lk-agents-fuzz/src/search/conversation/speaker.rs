//! The speaker: it plays the segments the agent flushed one at a time, reports their start, its
//! progress and their finish, and now and then loses a finish on the way to the tracker.
use super::*;

/// In a hundred finishes, how many the speaker's transport loses on the way to the tracker.
const FINISH_LOST: u8 = 2;

/// A segment the speaker started: when, and how much of it the speaker has reported played.
pub(super) struct Playing {
    pub(super) at: u64,
    pub(super) reported: u64,
}

impl Talk {
    /// Whether the person is hearing the agent right now: a segment the speaker reported started
    /// and has not finished.
    pub(super) fn hearing(&self) -> bool {
        self.run.record.live().of(Kind::Segment).iter().any(|s| self.heard.contains_key(s))
    }

    /// The speaker starts the oldest segment it was handed and has not started, once nothing is
    /// playing: one segment at a time, in order, as a speaker plays them. The moment a listener
    /// hears it begin, and the moment the driver marks the agent speaking.
    pub(super) async fn hear(&mut self) -> Result<(), Halt> {
        if self.hearing() {
            return Ok(());
        }
        let next = self
            .run
            .record
            .live()
            .of(Kind::Segment)
            .into_iter()
            .find(|s| !self.heard.contains_key(s));
        if let Some(segment) = next {
            let at = self.now_ms();
            self.heard.insert(segment, Playing { at, reported: 0 });
            self.send(Input::PlayoutStarted { segment: Ref::new(segment) }).await?;
        }
        Ok(())
    }

    /// The speaker reports how much of every segment it is playing has played by now, as a real
    /// output does at a discontinuity: the stretch since its last report, never past the segment.
    pub(super) async fn report_progress(&mut self) -> Result<(), Halt> {
        let now = self.now_ms();
        for segment in self.run.record.live().of(Kind::Segment) {
            let Some(length) = self.segment_ms(segment) else { continue };
            let Some(playing) = self.heard.get_mut(&segment) else { continue };
            let played = now.saturating_sub(playing.at).min(length);
            if played > playing.reported {
                let ms = Millis(played - playing.reported);
                playing.reported = played;
                self.send(Input::PlayoutProgressed { segment: Ref::new(segment), ms }).await?;
            }
        }
        Ok(())
    }

    /// The speaker finishes every segment that has played its length: what played since the last
    /// report, then the end. A segment still taking frames cannot have finished, and the output
    /// refuses the end; it is asked again on a later beat.
    pub(super) async fn finish_played(&mut self) -> Result<(), Halt> {
        let now = self.now_ms();
        for segment in self.run.record.live().of(Kind::Segment) {
            let Some(length) = self.segment_ms(segment) else { continue };
            if self.heard.get(&segment).is_none_or(|p| now < p.at + length) {
                continue;
            }
            self.report_progress().await?;
            let finish = self.finish(segment);
            self.send(finish).await?;
        }
        Ok(())
    }

    /// A segment's end as the speaker reports it: finished, or now and then finished with the
    /// report lost on the way.
    fn finish(&mut self, segment: u32) -> Input {
        let segment = Ref::new(segment);
        match self.rng.u8(0..100) < FINISH_LOST {
            true => Input::PlayoutFinishLost { segment },
            false => Input::PlayoutFinished { segment },
        }
    }

    /// Before time moves, the speaker finishes every flushed segment it reported played to the
    /// end, however that report came: a real output's finish follows its last progress at once,
    /// and the playout takes a finish missing for longer than its margin as lost.
    pub(super) async fn finish_drained(&mut self) -> Result<(), Halt> {
        for segment in self.run.record.live().of(Kind::Segment) {
            let Some(length) = self.segment_ms(segment) else { continue };
            let played: u64 = self
                .run
                .record
                .applied()
                .iter()
                .filter(|a| a.accepted)
                .filter_map(|a| match &a.input {
                    Input::PlayoutProgressed { segment: s, ms } if s.nth == segment => Some(ms.0),
                    _ => None,
                })
                .sum();
            if played >= length {
                let finish = self.finish(segment);
                self.apply_now(finish).await?;
            }
        }
        Ok(())
    }

    /// How long a segment plays: the audio captured for it, once flushed. A segment still taking
    /// frames has no length yet, and the speaker says nothing of it.
    pub(super) fn segment_ms(&self, segment: u32) -> Option<u64> {
        self.run.record.sink().iter().rev().find_map(|(_, sink)| match sink {
            Sink::Flush { segment: flushed, ms, .. } if *flushed == segment => Some(*ms),
            _ => None,
        })
    }
}
