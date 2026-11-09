use crate::check::{Check, Subject};
use crate::text::{TaskRecord, TextContract, TextRecording, TextRule};
use lk_agents_abi::storage::TaskState;

pub struct TextA2aTaskNotOrphaned;

impl TextRule for TextA2aTaskNotOrphaned {
    const CONTRACT: TextContract = TextContract {
        id: "a2a_task_not_orphaned",
        guarantee: "Once every turn ended and the lane ran its next turn, a task whose client disconnected, cancelled or aborted is settled (completed, failed, canceled, rejected or waiting on input), never left submitted or working.",
        examples: &[cancelled, completed],
        counterexamples: &[left_working],
    };

    fn on_complete(r: &TextRecording, check: &mut Check) {
        for task in r.tasks.iter().filter(|t| t.left) {
            let open = matches!(task.status, TaskState::Submitted | TaskState::Working);
            check.expect_with(Subject::Session, !open, || {
                format!("task {} was left {:?}", task.task_id, task.status)
            });
        }
    }
}

fn task(status: TaskState) -> TextRecording {
    let t = TaskRecord { task_id: "AST_1".into(), status, left: true };
    TextRecording { tasks: vec![t], ..TextRecording::default() }
}

fn cancelled() -> TextRecording {
    task(TaskState::Canceled)
}

fn completed() -> TextRecording {
    task(TaskState::Completed)
}

fn left_working() -> TextRecording {
    task(TaskState::Working)
}
