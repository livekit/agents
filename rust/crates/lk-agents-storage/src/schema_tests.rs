//! Schema initialization reconciles authoritative metadata, independent of server diagnostic text.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use lk_agents_storage::engine::{LeaseTiming, SchemaCell};
use lk_agents_storage::sql::{BatchResult, Row, SqlValue, Statement};
use lk_agents_storage::sqlite::FileConnection;
use lk_agents_storage::testing::Backend;
use lk_agents_storage::{CommitOptions, RunOptions, Thread, ThreadDbConnection};
use lk_agents_storage::{StorageError, db, run::RunOutcome};
use serde_json::json;

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
                    .batch(vec![Statement {
                        sql: "UPDATE schema_version SET version = ?".into(),
                        params: vec![SqlValue::Int(version)],
                    }])
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
        primary: Arc::new(FileConnection::open(file.clone(), true).await.unwrap()),
        contender: Arc::new(FileConnection::open(file, false).await.unwrap()),
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
            "schema-worker",
        )
        .await?;
        diagnostics.reword.store(true, Ordering::Relaxed);
        let lane = thread.lane("desk");
        let mut run = lane.open(RunOptions::new("schema-run", "AST_schema")).await?;
        run.state = Some(json!({"saved": true}));
        run.commit(CommitOptions::default()).await?;
        let mut restored = lane.open(RunOptions::new("schema-read", "AST_read")).await?;
        if restored.state != Some(json!({"saved": true})) {
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
        primary: Arc::new(FileConnection::open(file.clone(), true).await.unwrap()),
        contender: Arc::new(FileConnection::open(file, false).await.unwrap()),
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
        "schema-worker",
    )
    .await
    .unwrap();
    let opened = thread.lane("desk").open(RunOptions::new("schema-run", "AST_schema")).await;
    assert!(matches!(opened, Err(StorageError::Unavailable(_))), "{:?}", opened.err());
    diagnostics.close().await;
    store.close().await;
}
