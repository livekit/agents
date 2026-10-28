//! The successful speaker in workflow regressions: report real queued PCM as the clock advances.
use lk_agents_fuzz::drive::Run;
use lk_agents_fuzz::input::{Input, Millis, Ref};
use lk_agents_fuzz::record::{Kind, Sink};

/// The fixture holds a segment until it is flushed, then plays its accepted duration at normal
/// speed. Adversarial report-loss tests continue to drive these controls themselves.
pub async fn apply_with_playout(run: &mut Run, input: &Input) {
    let Input::Advance { ms } = input else {
        run.apply(input).await;
        return;
    };
    let mut remaining = ms.0;
    while remaining > 0 {
        let elapsed = remaining.min(100);
        remaining -= elapsed;
        run.apply(&Input::Advance { ms: Millis(elapsed) }).await;
        let now = run.record.now().at_ms;
        let pending = run.record.live().of(Kind::Segment);
        for segment in pending {
            let flushed = run.record.sink().iter().find_map(|(at, sink)| match sink {
                Sink::Flush { segment: id, ms, .. } if *id == segment => Some((at.at_ms, *ms)),
                _ => None,
            });
            let Some((at, duration)) = flushed else { continue };
            run.apply(&Input::PlayoutStarted { segment: Ref::new(segment) }).await;
            run.apply(&Input::PlayoutProgressed {
                segment: Ref::new(segment),
                ms: Millis(elapsed),
            })
            .await;
            if now >= at.saturating_add(duration) {
                run.apply(&Input::PlayoutFinished { segment: Ref::new(segment) }).await;
            }
        }
    }
}
