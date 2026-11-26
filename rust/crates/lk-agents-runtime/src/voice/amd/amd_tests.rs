use crate::voice::session::AmdTap;
use lk_agents_voice::amd::{AmdObservation, AmdRunId};

fn busy() -> AmdObservation {
    AmdObservation::AgentBusy { busy: true, at: lk_agents_abi::Ts::ZERO }
}

// Catches: one run hearing another's observations, or a detached run still fed.
#[test]
fn a_run_hears_only_its_own_observations_until_it_detaches() {
    let tap = AmdTap::default();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    tap.attach(AmdRunId(1), tx);
    tap.deliver(AmdRunId(2), &busy());
    tap.deliver(AmdRunId(1), &busy());
    assert_eq!(rx.try_recv().ok(), Some(busy()));
    assert!(rx.try_recv().is_err(), "run 2's observation went to run 1");
    tap.detach(AmdRunId(2));
    tap.deliver(AmdRunId(1), &busy());
    assert_eq!(rx.try_recv().ok(), Some(busy()), "another run's detach removed the route");
    tap.detach(AmdRunId(1));
    tap.deliver(AmdRunId(1), &busy());
    assert!(rx.try_recv().is_err());
}

// Catches: a run's route displaced by another run's attach, which ends the first run's
// observations and cancels it.
#[test]
fn another_runs_attach_leaves_a_runs_route_alone() {
    let tap = AmdTap::default();
    let (first, mut heard) = tokio::sync::mpsc::unbounded_channel();
    tap.attach(AmdRunId(1), first);
    let (second, _) = tokio::sync::mpsc::unbounded_channel();
    tap.attach(AmdRunId(2), second);
    tap.detach(AmdRunId(2));
    tap.deliver(AmdRunId(1), &busy());
    assert_eq!(heard.try_recv().ok(), Some(busy()), "run 1 is still routed");
}
