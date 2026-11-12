use lk_agents_abi::storage::StorageError;
use lk_agents_fuzz::{
    drive::{ConnSetup, Outcome, Run, Setup, StorageSetup, finish, simulate},
    input::{AgentText, Input, Millis, Ref},
    record::{Recording, StorageOperation, StorageStep},
};

#[derive(Clone, Copy)]
enum Reply {
    Complete,
    Fail,
}

fn reply(seed: u64, end: Reply) -> Recording {
    simulate(async {
        let setup = Setup {
            tts: false,
            storage: Some(StorageSetup { faults: true, ..StorageSetup::default() }),
            conn: Some(ConnSetup { llm: 0, ..ConnSetup::DEFAULT }),
            ..Setup::default()
        };
        let mut run = Run::start(&setup, seed, Default::default()).await;
        let finish = match end {
            Reply::Complete => Input::LlmEnd { chat: Ref::new(0) },
            Reply::Fail => Input::LlmFail { chat: Ref::new(0) },
        };
        for input in [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(2000) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("Hello.".into()) },
            finish,
            Input::Advance { ms: Millis(3000) },
        ] {
            run.apply(&input).await;
        }
        run.record
    })
}

#[test]
fn bound_storage_faults_close_the_call_and_successful_receipts_finish_the_speech() {
    // Storage's own abandon logs its failures and answers Ok: only acquires and commits fail.
    let mut unavailable = [false; 2];
    let mut slow = false;
    for seed in 0..1024 {
        for end in [Reply::Complete, Reply::Fail] {
            let recording = reply(seed, end);
            let host = recording.host_view();
            for (at, step) in host.storage() {
                let StorageStep::Answered { speech, operation, result, delivered: true } = step
                else {
                    continue;
                };
                let index = match operation {
                    StorageOperation::Acquire => 0,
                    StorageOperation::Settle => 1,
                    StorageOperation::Abandon => 2,
                    StorageOperation::ObserveLoss => continue,
                };
                if matches!(result, Err(StorageError::Unavailable(_))) && index < 2 {
                    unavailable[index] = true;
                }
                if *operation == StorageOperation::Acquire {
                    if result.is_err() {
                        assert!(
                            recording.inference_view().requests_for(*speech).is_empty(),
                            "seed {seed}: a failed durable acquire still started inference"
                        );
                        assert!(
                            recording.internal_view().error_close_requested_after(*at),
                            "seed {seed}: a failed durable acquire did not close the call"
                        );
                    } else {
                        assert!(!recording.inference_view().requests_for(*speech).is_empty());
                    }
                } else {
                    let requested = host.storage().iter().find_map(|(at, step)| {
                        matches!(step, StorageStep::Requested { speech: id, operation: op } if id == speech && op == operation).then_some(at.at_ms)
                    }).unwrap();
                    slow |= at.at_ms - requested >= 250;
                    if *operation == StorageOperation::Settle && result.is_err() {
                        assert!(
                            recording.internal_view().error_close_requested_after(*at),
                            "seed {seed}: a failed durable commit did not close the call"
                        );
                    }
                    assert!(
                        recording.internal_view().error_close_requested_after(*at)
                            || host
                                .speech(*speech)
                                .done_announcements()
                                .iter()
                                .any(|(done, _)| done > at),
                        "seed {seed}: {operation:?} receipt did not finish speech"
                    );
                }
            }
            let (failures, _) = lk_agents_fuzz::check::check_all(&recording);
            assert!(
                !failures.iter().any(|f| f.contract.starts_with("storage_")),
                "seed {seed}: {failures:?}"
            );
        }
        if unavailable.iter().all(|seen| *seen) && slow {
            return;
        }
    }
    panic!("seed window missed storage outcomes: unavailable={unavailable:?}, slow={slow}");
}

// Catches: a barge-in while the turn's lease take had applied but was unanswered: the
// interruption's watchdog dropped the speech with the acquire in flight, leaving the partition taken
// with no `Lease` to renew or release it, so the next turn waited out the lease's TTL and the
// agent was silent after the interruption.
#[test]
fn a_barge_in_during_the_lease_take_leaves_the_partition_free() {
    let outcome = simulate(async {
        let storage = StorageSetup { take_answer_ms: Some(6_000), ..StorageSetup::default() };
        let setup = Setup { tts: false, storage: Some(storage), ..Setup::default() };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        let reply = Input::GenerateReply { user_input: None, instructions: None, allow: None };
        run.apply(&reply).await;
        for _ in 0..100 {
            if run.lease_take_unanswered() {
                break;
            }
            run.apply(&Input::Advance { ms: Millis(1) }).await;
        }
        assert!(run.lease_take_unanswered(), "the turn never took the partition_key");
        // The watchdog gives up on the interrupted speech five seconds on, its acquire unanswered.
        run.apply(&Input::Interrupt { force: true, wait: false }).await;
        run.apply(&Input::Advance { ms: Millis(5_500) }).await;
        let answered = run.record.host_view().storage().iter().any(|(_, step)| {
            matches!(step, StorageStep::Answered { operation: StorageOperation::Acquire, .. })
        });
        assert!(!answered && !run.lease_take_unanswered(), "the acquire was not dropped");
        run.apply(&reply).await;
        run.apply(&Input::Advance { ms: Millis(12_000) }).await;
        finish(run, Outcome::default()).await
    });
    let partition_key: Vec<_> =
        outcome.failures.iter().filter(|f| f.contract.starts_with("storage_lane")).collect();
    assert!(partition_key.is_empty(), "{partition_key:#?}");
}

// Catches: the interruption watchdog ending a speech whose tools ended just before it fired, while
// the turn they reported was still being committed: the speech was announced done before its
// storage receipt (seed 21210, BasicPos).
#[test]
fn a_turn_reported_before_the_watchdog_fires_waits_for_its_receipt() {
    use lk_agents_fuzz::input::{ToolAnswer, ToolArgs};
    let mut straddled = false;
    for seed in 0..256 {
        let recording = simulate(async {
            let setup = Setup {
                tts: false,
                tools: vec!["lookup".into()],
                storage: Some(StorageSetup::default()),
                ..Setup::default()
            };
            let mut run = Run::start(&setup, seed, Default::default()).await;
            for input in [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(100) },
                Input::LlmToolCall {
                    chat: Ref::new(0),
                    name: "lookup".into(),
                    args: ToolArgs("{}".into()),
                },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(100) },
                Input::Interrupt { force: true, wait: false },
                // One millisecond short of the watchdog the interruption armed.
                Input::Advance { ms: Millis(4_999) },
                Input::ToolAnswered {
                    tool: Ref::new(0),
                    answer: ToolAnswer::Value,
                    text: AgentText("Found it.".into()),
                },
                Input::Advance { ms: Millis(3_000) },
            ] {
                run.apply(&input).await;
            }
            run.record
        });
        let host = recording.host_view();
        let committed = host.storage().iter().find_map(|(at, step)| {
            matches!(step, StorageStep::Requested { operation: StorageOperation::Settle, .. })
                .then_some(at.at_ms)
        });
        let received = host.storage().iter().find_map(|(at, step)| {
            matches!(step, StorageStep::Answered { operation: StorageOperation::Settle, .. })
                .then_some(at.at_ms)
        });
        straddled |= committed.zip(received).is_some_and(|(c, r)| r > c);
        let (failures, _) = lk_agents_fuzz::check::check_all(&recording);
        assert!(
            !failures.iter().any(|f| f.contract.starts_with("storage_")),
            "seed {seed}: {failures:?}"
        );
    }
    assert!(straddled, "no seed committed the turn across the watchdog");
}

// Catches: a realtime reply whose server audio was already arriving held back until its speech
// reloaded the voice partition, so the person waited on storage for the first word
// (`first_audio_is_prompt`, campaign seeds 9850, 27056 and 27623, with storage lost).
#[test]
fn a_realtime_reply_plays_while_its_partition_reloads() {
    for seed in [9850, 27056, 27623] {
        let setup = lk_agents_fuzz::search::sample_setup(&mut fastrand::Rng::with_seed(seed));
        let outcome = lk_agents_fuzz::search::explore(setup, seed, 3000);
        let late = outcome.failures.iter().any(|f| f.contract == "first_audio_is_prompt");
        assert!(!late, "seed {seed}: the reply's first audio waited on the partition_key");
    }
}

// Catches the harness judging the partition by itself, not by the real backend (launch-gate seeds
// 1239, 4416, 2692, 2634, 2990, 3368): a database worker a lease's renewal respawned off the task
// bridge waited for virtual time to move until the lease lapsed; another worker's steal read the
// lease it replaced before the call's take applied, and its take was observed before it was noted
// speaking; a turn the close released at its bound reported afterwards, as designed.
#[test]
fn the_partition_is_judged_by_the_real_backend() {
    for seed in [1239, 4416, 2692, 2634, 2990, 3368] {
        let setup = lk_agents_fuzz::search::sample_setup(&mut fastrand::Rng::with_seed(seed));
        let outcome = lk_agents_fuzz::search::explore(setup, seed, 3000);
        let storage: Vec<_> =
            outcome.failures.iter().filter(|f| f.contract.starts_with("storage_")).collect();
        assert!(storage.is_empty(), "seed {seed}: {storage:#?}");
    }
}
