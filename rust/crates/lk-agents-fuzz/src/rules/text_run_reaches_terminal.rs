use crate::check::{Check, Subject};
use crate::text::{
    CancelBy, EXIT_BOUND_MS, RunKind, TextContract, TextRecording, TextRule, run_record,
};

pub struct TextRunReachesTerminal;

impl TextRule for TextRunReachesTerminal {
    const CONTRACT: TextContract = TextContract {
        id: "text_run_reaches_terminal",
        guarantee: "Every text run reports its final outcome and its driver exits: on its own while its fakes answer or panic, and within the exit bound once a client cancels it.",
        examples: &[finished, cancelled_while_hung],
        counterexamples: &[waited_on_a_panic, never_exited, exited_late],
    };

    fn on_complete(r: &TextRecording, check: &mut Check) {
        for run in &r.runs {
            let id = run.id;
            check.expect_with(Subject::Session, run.exited_ms.is_some(), || {
                format!("run {id} ({:?}): its driver never exited", run.kind)
            });
            match run.cancelled {
                Some((at, by)) => {
                    check.expect_with(Subject::Session, by == CancelBy::Client || run.hung, || {
                        format!("run {id} ({:?}): still running at the client's deadline with no fake left unanswered", run.kind)
                    });
                    let bound = at + EXIT_BOUND_MS;
                    check.expect_with(Subject::Session, run.exited_ms.is_none_or(|e| e <= bound), || {
                        format!("run {id}: exited {:?} ms, past the bound {bound} ms after its cancel", run.exited_ms)
                    });
                }
                None => {
                    let reported = run.terminal_ms.is_some_and(|t| run.exited_ms.is_none_or(|e| t <= e));
                    check.expect_with(Subject::Session, reported, || {
                        format!("run {id} ({:?}): no final outcome before its driver exited", run.kind)
                    });
                }
            }
        }
    }
}

fn with(f: impl FnOnce(&mut crate::text::RunRecord)) -> TextRecording {
    let mut run = run_record(RunKind::Direct);
    f(&mut run);
    TextRecording { runs: vec![run], ..TextRecording::default() }
}

fn finished() -> TextRecording {
    with(|_| {})
}

fn cancelled_while_hung() -> TextRecording {
    with(|r| {
        r.hung = true;
        r.terminal_ms = None;
        r.cancelled = Some((120_000, CancelBy::Deadline));
        r.exited_ms = Some(125_000);
    })
}

fn waited_on_a_panic() -> TextRecording {
    with(|r| {
        r.panicked = true;
        r.cancelled = Some((120_000, CancelBy::Deadline));
        r.exited_ms = Some(125_000);
    })
}

fn never_exited() -> TextRecording {
    with(|r| r.exited_ms = None)
}

fn exited_late() -> TextRecording {
    with(|r| {
        r.cancelled = Some((1_000, CancelBy::Client));
        r.exited_ms = Some(1_000 + EXIT_BOUND_MS + 1);
    })
}
