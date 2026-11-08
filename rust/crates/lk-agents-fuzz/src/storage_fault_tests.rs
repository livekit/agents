//! Seeded storage errors and slow commits exercised through the real voice executor.
use lk_agents_abi::storage::StorageError;
use lk_agents_fuzz::{
    drive::{ConnSetup, Run, Setup, simulate},
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
            storage_lost_at: Some(u32::MAX),
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
fn storage_faults_and_slow_commits_finish_the_speech() {
    let mut unavailable = [false; 3];
    let mut conflict = [false; 3];
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
                match result {
                    Err(StorageError::Unavailable(_)) => unavailable[index] = true,
                    Err(StorageError::VersionConflict { .. }) => conflict[index] = true,
                    _ => {}
                }
                if *operation == StorageOperation::Acquire {
                    assert!(
                        !recording.inference_view().requests_for(*speech).is_empty(),
                        "seed {seed}: failed acquire prevented the model request"
                    );
                } else {
                    let requested = host.storage().iter().find_map(|(at, step)| {
                        matches!(step, StorageStep::Requested { speech: id, operation: op } if id == speech && op == operation).then_some(at.at_ms)
                    }).unwrap();
                    slow |= at.at_ms - requested >= 250;
                    assert!(
                        host.speech(*speech).done_announcements().iter().any(|(done, _)| done > at),
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
        if unavailable.iter().all(|seen| *seen) && conflict.iter().all(|seen| *seen) && slow {
            return;
        }
    }
    panic!(
        "seed window missed storage outcomes: unavailable={unavailable:?}, conflict={conflict:?}, slow={slow}"
    );
}
