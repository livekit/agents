//! A prediction is judged against the detector's own threshold.
//!
//! The reference asks the detector stream for `unlikely_threshold(language)` once a prediction
//! lands: the user's overrides, the gateway's defaults and a fallback's rescale all live there.
//! The fake detector's table puts every language at [`EOT_THRESHOLD`], so each announced
//! prediction must carry it.
use lk_agents_voice::events::{EotPredictionEvent, SessionEvent};
use lk_agents_voice::trace::Step;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    fakes::EOT_THRESHOLD,
    record::{Kind, Recording},
};

pub struct EotJudgedByTheDetector;

impl Rule for EotJudgedByTheDetector {
    const CONTRACT: Contract = Contract {
        id: "eot_judged_by_the_detector",
        guarantee: "Every end-of-turn prediction is judged against the detector's own threshold.",
        examples: &[judged_by_the_detector],
        counterexamples: &[judged_by_the_stock_table],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let judged = r.journal_where(|step| match step {
            Step::Emitted(SessionEvent::EotPrediction(ev)) => Some(ev.threshold),
            _ => None,
        });
        for (at, threshold) in judged {
            check.expect(
                Subject::Resource(Kind::Eot, 0),
                threshold == EOT_THRESHOLD,
                format!(
                    "prediction at seq {} judged against {threshold}, the detector's is {EOT_THRESHOLD}",
                    at.seq
                ),
            );
        }
    }
}

fn announced(threshold: f64) -> Recording {
    Authored::new()
        .journal(Step::Emitted(SessionEvent::EotPrediction(EotPredictionEvent {
            probability: 0.45,
            threshold,
            inference_duration: 0.0,
            delay: std::time::Duration::from_millis(250),
        })))
        .done()
}

/// The detector's threshold.
fn judged_by_the_detector() -> Recording {
    announced(EOT_THRESHOLD)
}

/// The mini model's stock English threshold, the detector's own ignored.
fn judged_by_the_stock_table() -> Recording {
    announced(0.36)
}
