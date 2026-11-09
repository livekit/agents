use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use livekit_protocol::agent::chat_context::chat_item;
use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
use livekit_protocol::agent::{ChatMessage, ChatRole};
use serde_json::json;

use crate::backends::sqlite::FileConnection;
use crate::sql::{BatchResult, Row, Statement};
use crate::testing::Backend;
use crate::thread::db;
use crate::thread::run::{ItemRow, TaskState};
use crate::thread::schema::TaskData;
use crate::thread::{LeaseTiming, SchemaCell};
use crate::{CommitOptions, RunOptions, TaskWrite, Thread, ThreadDbConnection};
use crate::{RunOutcome, StorageError};

pub(crate) const LANE: &str = "front-desk";

pub(crate) fn message(id: &str, role: ChatRole, text: &str) -> ItemRow {
    ItemRow::new(chat_item::Item::Message(ChatMessage {
        id: id.into(),
        role: role as i32,
        content: vec![ChatContent { payload: Some(Payload::Text(text.into())) }],
        ..Default::default()
    }))
}

pub(crate) fn submitted(task_id: &str) -> TaskWrite {
    TaskWrite {
        task_id: task_id.into(),
        status: TaskState::Submitted,
        message_id: Some(format!("m_{task_id}")),
        name: None,
        data: TaskData::default(),
        internal: None,
    }
}

struct Diagnostics {
    primary: Arc<FileConnection>,
    contender: Arc<FileConnection>,
    reword: AtomicBool,
    install_race: AtomicBool,
    contender_version: Option<i64>,
    /// The database becomes unreachable once the contender installed its schema.
    lost_after_race: bool,
    unreachable: AtomicBool,
}

impl Diagnostics {
    fn error(&self, error: StorageError) -> StorageError {
        match error {
            StorageError::Invalid(_) if self.reword.load(Ordering::Relaxed) => {
                StorageError::Invalid("backend diagnostic was reworded".into())
            }
            other => other,
        }
    }
}

#[async_trait]
impl ThreadDbConnection for Diagnostics {
    async fn query(&self, statement: Statement) -> Result<Vec<Row>, StorageError> {
        if self.unreachable.load(Ordering::Relaxed) {
            return Err(StorageError::Unavailable("database unreachable".into()));
        }
        self.primary.query(statement).await.map_err(|error| self.error(error))
    }

    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError> {
        if self.install_race.swap(false, Ordering::Relaxed) {
            self.contender.batch(db::bootstrap()).await?;
            if let Some(version) = self.contender_version {
                self.contender
                    .batch(vec![Statement::raw(format!(
                        "UPDATE schema_version SET version = {version}"
                    ))])
                    .await?;
            }
            self.unreachable.store(self.lost_after_race, Ordering::Relaxed);
        }
        self.primary.batch(statements).await.map_err(|error| self.error(error))
    }

    async fn close(&self) {
        self.primary.close().await;
        self.contender.close().await;
    }
}

async fn initialize(reword_initial: bool, install_race: bool, version: Option<i64>) {
    let store = Backend::Sqlite.store(LeaseTiming::default()).await;
    let file = store.dir.as_ref().unwrap().join("schema-probe.sqlite");
    let diagnostics = Arc::new(Diagnostics {
        primary: Arc::new(
            FileConnection::open(file.clone(), true, crate::testing::threads()).await.unwrap(),
        ),
        contender: Arc::new(
            FileConnection::open(file, false, crate::testing::threads()).await.unwrap(),
        ),
        reword: AtomicBool::new(reword_initial),
        install_race: AtomicBool::new(install_race),
        contender_version: version,
        lost_after_race: false,
        unreachable: AtomicBool::new(false),
    });
    let result = async {
        let thread = Thread::new(
            diagnostics.clone(),
            "AT_schema_probe",
            LeaseTiming::default(),
            SchemaCell::default(),
        )
        .await?;
        diagnostics.reword.store(true, Ordering::Relaxed);
        let lane = thread.lane("desk");
        let mut run = lane
            .open(RunOptions::new("schema-run", "AST_schema"), &crate::Holder::new("test"))
            .await?;
        run.set_state(json!({"saved": true}));
        run.commit(CommitOptions::default()).await?;
        let mut restored = lane
            .open(RunOptions::new("schema-read", "AST_read"), &crate::Holder::new("test"))
            .await?;
        if restored.state() != Some(&json!({"saved": true})) {
            return Err(StorageError::Unavailable(
                "the initialized schema lost the committed state".into(),
            ));
        }
        restored.abandon(RunOutcome::Completed, "verified").await?;
        thread.close().await;
        Ok(())
    }
    .await;
    diagnostics.close().await;
    store.close().await;
    if version.is_some() {
        assert!(matches!(result, Err(StorageError::Invalid(_))), "{result:?}");
    } else {
        result.unwrap();
    }
}

#[tokio::test]
async fn opening_a_new_database_does_not_depend_on_error_prose() {
    initialize(true, false, None).await;
}

#[tokio::test]
async fn a_schema_install_race_reconciles_reworded_errors() {
    initialize(false, true, None).await;
}

#[tokio::test]
async fn a_schema_install_race_refuses_an_incompatible_version() {
    initialize(false, true, Some(99)).await;
}

#[tokio::test]
async fn a_lost_install_race_reports_why_the_schema_could_not_be_checked() {
    let store = Backend::Sqlite.store(LeaseTiming::default()).await;
    let file = store.dir.as_ref().unwrap().join("schema-probe.sqlite");
    let diagnostics = Arc::new(Diagnostics {
        primary: Arc::new(
            FileConnection::open(file.clone(), true, crate::testing::threads()).await.unwrap(),
        ),
        contender: Arc::new(
            FileConnection::open(file, false, crate::testing::threads()).await.unwrap(),
        ),
        reword: AtomicBool::new(false),
        install_race: AtomicBool::new(true),
        contender_version: None,
        lost_after_race: true,
        unreachable: AtomicBool::new(false),
    });
    let thread = Thread::new(
        diagnostics.clone(),
        "AT_schema_probe",
        LeaseTiming::default(),
        SchemaCell::default(),
    )
    .await
    .unwrap();
    let opened = thread
        .lane("desk")
        .open(RunOptions::new("schema-run", "AST_schema"), &crate::Holder::new("test"))
        .await;
    assert!(matches!(opened, Err(StorageError::Unavailable(_))), "{:?}", opened.err());
    diagnostics.close().await;
    store.close().await;
}

/// Items page by insertion position with an optional lane filter; the cursor is opaque,
/// bound to its filter, and pages neither skip nor repeat.
async fn items_page_by_position_with_a_filter_bound_cursor(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread().await;
    for (lane, n) in [("lane-a", 0), ("lane-b", 1), ("lane-a", 2)] {
        let mut run = s
            .lane(lane)
            .open(RunOptions::new(format!("r{n}"), ""), &crate::Holder::new("test"))
            .await
            .unwrap();
        run.history_mut().push(message(&format!("i{n}"), ChatRole::User, "hi"));
        run.commit(CommitOptions::default()).await.unwrap();
    }
    let ids = |p: &crate::thread::db::ItemPage| -> Vec<String> {
        p.items.iter().map(|i| format!("{}/{}", i.lane, i.item.id)).collect()
    };
    let first = s.list_items(None, None, 2).await.unwrap();
    assert_eq!(ids(&first), vec!["lane-a/i0", "lane-b/i1"]);
    let rest = s.list_items(None, first.next.as_deref(), 2).await.unwrap();
    assert_eq!((ids(&rest), rest.next), (vec!["lane-a/i2".to_owned()], None));
    let a = s.list_items(Some("lane-a"), None, 10).await.unwrap();
    assert_eq!(ids(&a), vec!["lane-a/i0", "lane-a/i2"]);
    let crossed = s.list_items(Some("lane-a"), first.next.as_deref(), 2).await;
    assert!(matches!(crossed, Err(StorageError::Invalid(_))), "a cursor of another filter");
    fx.cleanup(&[&s]).await;
}

/// A2A ListTasks pages by last update (ties by task id) without skipping or
/// repeating, filters by status, and refuses a page token made under other filters.
async fn tasks_page_by_last_update_with_a_filter_bound_token(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread().await;
    let lane = s.lane(LANE);
    for id in ["AST_1", "AST_2", "AST_3", "AST_4", "AST_5"] {
        assert!(lane.submit(&submitted(id)).await.unwrap());
    }
    tokio::time::sleep(Duration::from_millis(5)).await;
    lane.cancel_task("AST_2", Default::default()).await.unwrap();
    let query = |status, page_token| crate::thread::db::TaskQuery {
        status,
        page_size: 2,
        page_token,
        ..Default::default()
    };
    let (mut seen, mut token) = (Vec::new(), None);
    loop {
        let page = s.list_tasks(&query(None, token)).await.unwrap();
        assert_eq!(page.total_size, 5);
        seen.extend(page.tasks.iter().map(|t| (t.updated_at, t.task_id.clone())));
        if page.next_page_token.is_empty() {
            break;
        }
        token = Some(page.next_page_token);
    }
    let mut ordered = seen.clone();
    ordered.sort_by(|a, b| b.cmp(a));
    ordered.dedup();
    assert_eq!(seen, ordered, "last update first, ties by id, nothing repeated");
    assert_eq!(seen.len(), 5);
    assert_eq!(seen[0].1, "AST_2", "the canceled task was updated last");

    let canceled = s.list_tasks(&query(Some(TaskState::Canceled), None)).await.unwrap();
    assert_eq!(canceled.tasks.iter().map(|t| t.task_id.as_str()).collect::<Vec<_>>(), ["AST_2"]);
    let submitted = s.list_tasks(&query(Some(TaskState::Submitted), None)).await.unwrap();
    assert_eq!(submitted.total_size, 4);
    let crossed = s.list_tasks(&query(None, Some(submitted.next_page_token))).await;
    assert!(matches!(crossed, Err(StorageError::Invalid(_))), "a token of another filter");
    fx.cleanup(&[&s]).await;
}

/// A thread's tasks browse in insertion order, of one lane when asked, whatever was updated
/// since; the token is bound to that order and lane.
async fn tasks_browse_in_insertion_order_by_lane(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread().await;
    for (lane, id) in [(LANE, "AST_1"), ("expert", "AST_2"), (LANE, "AST_3"), (LANE, "AST_4")] {
        assert!(s.lane(lane).submit(&submitted(id)).await.unwrap());
    }
    tokio::time::sleep(Duration::from_millis(5)).await;
    s.lane(LANE).cancel_task("AST_1", Default::default()).await.unwrap();
    let query = |page_token| crate::thread::db::TaskQuery {
        lane: Some(LANE.into()),
        order: crate::thread::db::TaskOrder::Inserted,
        page_size: 2,
        page_token,
        ..Default::default()
    };
    let (mut seen, mut token) = (Vec::new(), None);
    loop {
        let page = s.list_tasks(&query(token)).await.unwrap();
        assert_eq!(page.total_size, 3);
        seen.extend(page.tasks.into_iter().map(|t| t.task_id));
        if page.next_page_token.is_empty() {
            break;
        }
        token = Some(page.next_page_token);
    }
    assert_eq!(seen, ["AST_1", "AST_3", "AST_4"], "oldest first, the lane's only");
    let first = s.list_tasks(&query(None)).await.unwrap();
    let crossed = s
        .list_tasks(&crate::thread::db::TaskQuery {
            page_token: Some(first.next_page_token),
            ..Default::default()
        })
        .await;
    assert!(matches!(crossed, Err(StorageError::Invalid(_))), "a token of another listing");
    fx.cleanup(&[&s]).await;
}

crate::contract!(
    items_page_by_position_with_a_filter_bound_cursor,
    tasks_page_by_last_update_with_a_filter_bound_token,
    tasks_browse_in_insertion_order_by_lane,
);
