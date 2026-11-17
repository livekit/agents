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
        ] {
            run.apply(&input).await;
        }
        // A save asked is answered on storage's own clock, resent until its deadline at most: each
        // round moves the clock to the furthest deadline the driver or the store waits on.
        let owed = |record: &Recording| {
            let storage = record.host_view().storage();
            let save = |op: &StorageOperation| {
                matches!(op, StorageOperation::Settle | StorageOperation::Abandon)
            };
            let count = |answered: bool| {
                storage
                    .iter()
                    .filter(|(_, step)| match step {
                        StorageStep::Requested { operation, .. } => !answered && save(operation),
                        StorageStep::Answered { operation, .. } => answered && save(operation),
                        _ => false,
                    })
                    .count()
            };
            count(false) > count(true)
        };
        for _ in 0..16 {
            let round = run.release_advance();
            run.apply(&round).await;
            if !owed(&run.record) {
                return run.record;
            }
        }
        panic!("seed {seed}: the turn's save was never answered");
    })
}

#[test]
fn bound_storage_faults_close_the_call_and_successful_receipts_finish_the_speech() {
    // Storage's own abandon logs its failures and answers Ok: only acquires and commits fail.
    let mut failed = [false; 2];
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
                    StorageOperation::ObserveLoss | StorageOperation::Checkpoint => continue,
                };
                // Resend can turn an I/O refusal into lease loss before the final receipt.
                // Cover failed acquires and commits independent of that intermediate kind.
                if result.is_err() && index < 2 {
                    failed[index] = true;
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
                        matches!(step, StorageStep::Requested { speech: id, operation: op } if id == speech && op == operation).then_some(*at)
                    }).unwrap();
                    slow |= at.at_ms - requested.at_ms >= 250;
                    // A commit resent through an outage may outlast the lease: its loss closes the
                    // call before the failure is answered.
                    // A lease-loss observation can already have requested close before the
                    // pending speech reports its failed settle; it does not request it twice.
                    let acquired = host.storage().iter().find_map(|(at,step)| {
                        matches!(step,StorageStep::Requested {speech:id,operation:StorageOperation::Acquire} if id==speech).then_some(*at)
                    }).unwrap();
                    let closed = recording.internal_view().error_close_requested_after(acquired);
                    if *operation == StorageOperation::Settle && result.is_err() {
                        assert!(
                            closed,
                            "seed {seed}: a failed durable commit did not close the call: {result:?}"
                        );
                    }
                    assert!(
                        closed
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
        if failed.iter().all(|seen| *seen) && slow {
            return;
        }
    }
    panic!("seed window missed storage outcomes: failed={failed:?}, slow={slow}");
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
// (`first_audio_is_prompt`, with storage slow to answer).
#[test]
fn a_realtime_reply_plays_while_its_partition_reloads() {
    use lk_agents_fuzz::{drive::RealtimeSetup, record::Sink};
    let recording = simulate(async {
        let storage = StorageSetup { take_answer_ms: Some(3_000), ..StorageSetup::default() };
        let setup = Setup {
            realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
            storage: Some(storage),
            ..Setup::default()
        };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(5_000) },
        ] {
            run.apply(&input).await;
        }
        run.record
    });
    let acquired = recording.host_view().storage().iter().find_map(|(at, step)| {
        matches!(step, StorageStep::Answered { operation: StorageOperation::Acquire, .. })
            .then_some(at.at_ms)
    });
    let played = recording
        .sink()
        .iter()
        .find_map(|(at, sink)| matches!(sink, Sink::Audio { .. }).then_some(at.at_ms));
    // The storage answers the take three seconds on; the server's audio arrives long before.
    assert!(
        played.zip(acquired).is_some_and(|(played, acquired)| played < acquired),
        "the reply's first audio waited on the partition: played {played:?}, acquired {acquired:?}"
    );
    let (failures, _) = lk_agents_fuzz::check::check_all(&recording);
    assert!(failures.is_empty(), "{failures:#?}");
}

// Catches: an interruption landing on a speech whose turn was already reported, its commit sent
// again through an outage: the interruption armed the watchdog anew and it ended the speech, so
// the speech was announced done before its storage receipt (seed 634, TaskBridge).
#[test]
fn an_interruption_after_the_turn_is_reported_waits_for_its_receipt() {
    let recording = simulate(async {
        // The database goes down between the reply's checkpoint and its commit, past the watchdog.
        let storage = StorageSetup { outage: Some((2_001, 9_000)), ..StorageSetup::default() };
        let setup = Setup { tts: false, storage: Some(storage), ..Setup::default() };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(2_000) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("Hello.".into()) },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
            Input::Interrupt { force: true, wait: false },
            Input::Advance { ms: Millis(12_000) },
        ] {
            run.apply(&input).await;
        }
        run.record
    });
    let host = recording.host_view();
    let settle = |answered: bool| {
        host.storage().iter().find_map(|(at, step)| {
            let op = match step {
                StorageStep::Requested { operation, .. } if !answered => operation,
                StorageStep::Answered { operation, .. } if answered => operation,
                _ => return None,
            };
            (*op == StorageOperation::Settle).then_some(at.at_ms)
        })
    };
    // The interruption lands at 2.5 s, the commit already sent; its watchdog would fire at 7.5 s.
    let (committed, received) = (settle(false), settle(true));
    assert!(
        committed.is_some_and(|c| c < 2_500) && received.is_none_or(|r| r > 7_500),
        "the commit did not straddle the watchdog: sent {committed:?}, answered {received:?}"
    );
    let (failures, _) = lk_agents_fuzz::check::check_all(&recording);
    let storage: Vec<_> = failures.iter().filter(|f| f.contract.starts_with("storage_")).collect();
    assert!(storage.is_empty(), "{storage:#?}");
}

// Catches (seed 1284): an interrupted reply whose turn waited on its checkpoint, the
// checkpoint's receipt withheld after its write, ended by the interruption watchdog,
// read as a reply stuck on a wait the interrupt should cut: the recording never showed the
// checkpoint, the turn's own wait on its partition.
#[test]
fn an_interrupted_turn_waits_on_its_checkpoint() {
    use lk_agents_fuzz::input::{ToolAnswer, ToolArgs};
    let recording = simulate(async {
        let storage =
            StorageSetup { checkpoint_answer_ms: Some((1, 20_000)), ..StorageSetup::default() };
        let setup = Setup {
            tts: false,
            tools: vec!["lookup".into()],
            storage: Some(storage),
            ..Setup::default()
        };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(300) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Value,
                text: AgentText("Found it.".into()),
            },
            Input::Advance { ms: Millis(50) },
            Input::Interrupt { force: true, wait: false },
            Input::Advance { ms: Millis(6_000) },
        ] {
            run.apply(&input).await;
        }
        run.record
    });
    assert!(checkpoint_at_watchdog(&recording), "no watchdog fired over an unanswered checkpoint");
    let rule = "interrupt_ends_the_reply";
    let (failures, _) = lk_agents_fuzz::check::check_all(&recording);
    let failed: Vec<_> = failures.iter().filter(|f| f.contract == rule).collect();
    assert!(failed.is_empty(), "{failed:#?}");
}

/// Whether a speech's interruption watchdog fired while its turn's checkpoint was unanswered.
fn checkpoint_at_watchdog(recording: &Recording) -> bool {
    let storage = recording.host_view().storage();
    recording.host_view().speeches().iter().any(|speech| {
        let Some(fired) = recording.internal_view().speech(speech.id).timed_out_at() else {
            return false;
        };
        let checkpoint = |answered: bool| {
            storage
                .iter()
                .filter(|(at, step)| {
                    *at < fired
                        && match step {
                            StorageStep::Requested { speech: s, operation } => {
                                !answered
                                    && *s == speech.id
                                    && *operation == StorageOperation::Checkpoint
                            }
                            StorageStep::Answered { speech: s, operation, .. } => {
                                answered
                                    && *s == speech.id
                                    && *operation == StorageOperation::Checkpoint
                            }
                            _ => false,
                        }
                })
                .count()
        };
        checkpoint(false) > checkpoint(true)
    })
}

// Catches (seed 1626): a write the database refused, its resend still waiting out storage's own
// backoff on the clock, read as a quiet run: the case settled and closed with the speech still
// owed its receipt, and `speech_settles` found it done only after everything was answered.
#[test]
fn a_write_refused_and_resent_is_owed_until_its_answer() {
    let outcome = simulate(async {
        let storage = StorageSetup { outage: Some((16_100, 5_000)), ..StorageSetup::default() };
        let setup = Setup { tts: false, storage: Some(storage), ..Setup::default() };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            // Past the first away timer, so no deadline of the driver's outlasts the resends.
            Input::Advance { ms: Millis(16_000) },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("Hello.".into()) },
            Input::LlmEnd { chat: Ref::new(0) },
            // The commit is refused from here on, and resent on storage's own backoff.
            Input::Advance { ms: Millis(1_000) },
        ] {
            run.apply(&input).await;
        }
        finish(run, Outcome::default()).await
    });
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}
